"""
Biological Knowledge Graph Construction Module

This module builds sentence-specific biological knowledge graphs by:
1. Querying biological databases (KEGG, STRING, Reactome, GO)
2. Constructing subgraphs with entity relationships
3. Adding hallmark-specific pathway nodes
"""

import logging
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
import networkx as nx
import requests
import asyncio
import aiohttp
from collections import defaultdict
import json
from concurrent.futures import ThreadPoolExecutor
import pickle
import os

from .bio_entity_extractor import BioEntity

logger = logging.getLogger(__name__)


@dataclass
class KGNode:
    """Represents a node in the biological knowledge graph"""
    node_id: str
    node_type: str  # gene, protein, pathway, go_term, hallmark
    name: str
    properties: Dict[str, Any]
    original_text: Optional[str] = None  # Original text before normalization
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        return self.node_id == other.node_id


@dataclass
class KGEdge:
    """Represents an edge in the biological knowledge graph"""
    source: str
    target: str
    edge_type: str  # interacts, regulates, pathway_member, associated_with
    properties: Dict[str, Any]
    confidence: float


class BiologicalKGBuilder:
    """
    Constructs biological knowledge graphs for input sentences by integrating
    information from multiple biological databases.
    """
    
    # Cancer hallmark to pathway mappings (simplified version)
    HALLMARK_PATHWAYS = {
        'evading_growth_suppressors': ['hsa04110', 'hsa04310', 'hsa04115'],  # Cell cycle, Wnt, p53
        'tumor_promoting_inflammation': ['hsa04620', 'hsa04668', 'hsa04064'],  # Toll-like, TNF, NF-kB
        'enabling_replicative_immortality': ['hsa04310', 'hsa04350'],  # Wnt, TGF-beta
        'cellular_energetics': ['hsa00010', 'hsa00020', 'hsa04152'],  # Glycolysis, TCA, AMPK
        'resisting_cell_death': ['hsa04210', 'hsa04215', 'hsa04115'],  # Apoptosis, p53
        'activating_invasion_metastasis': ['hsa04510', 'hsa04810', 'hsa04670'],  # Focal adhesion, cytoskeleton
        'genomic_instability': ['hsa03430', 'hsa03440', 'hsa04115'],  # DNA repair, p53
        'inducing_angiogenesis': ['hsa04370', 'hsa04066'],  # VEGF, HIF-1
        'sustaining_proliferative_signaling': ['hsa04010', 'hsa04012', 'hsa04151'],  # MAPK, ErbB, PI3K-Akt
        'avoiding_immune_destruction': ['hsa04650', 'hsa04660', 'hsa04672']  # NK cell, T cell, B cell
    }
    
    def __init__(self, config: Dict):
        """
        Initialize the KG builder with database configurations.
        
        Args:
            config: Configuration dictionary with KG construction settings
        """
        self.config = config
        self.databases = config.get('databases', [])
        self.max_hops = config.get('graph_construction', {}).get('max_hops', 2)
        self.max_neighbors = config.get('graph_construction', {}).get('max_neighbors', 50)
        self.include_pathways = config.get('graph_construction', {}).get('include_pathways', True)
        self.include_go_terms = config.get('graph_construction', {}).get('include_go_terms', True)
        
        # Initialize database clients
        self._init_database_clients()
        
        # Cache for API responses
        self.cache_dir = "cache/kg_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.api_cache = self._load_cache()
        
        # Pending edges to add after nodes are created
        self.pending_edges = []
        
    def _init_database_clients(self):
        """Initialize API clients for biological databases"""
        self.db_configs = {}
        
        for db in self.databases:
            if db['enabled']:
                self.db_configs[db['name']] = db
                
        # STRING specific settings
        if 'STRING' in self.db_configs:
            self.string_confidence = self.db_configs['STRING'].get('confidence_threshold', 700)
    
    def _load_cache(self) -> Dict:
        """Load API response cache"""
        cache_file = os.path.join(self.cache_dir, "api_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save API response cache"""
        cache_file = os.path.join(self.cache_dir, "api_cache.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.api_cache, f)
    
    async def build_knowledge_graph(self, entities: List[BioEntity], 
                                  hallmarks: Optional[List[str]] = None) -> nx.MultiDiGraph:
        """
        Build a biological knowledge graph for the given entities.
        
        Args:
            entities: List of extracted biological entities
            hallmarks: Optional list of cancer hallmarks to include pathways for
            
        Returns:
            NetworkX MultiDiGraph representing the biological knowledge graph
        """
        kg = nx.MultiDiGraph()
        
        # Add entity nodes
        entity_nodes = self._create_entity_nodes(entities)
        for node in entity_nodes:
            kg.add_node(node.node_id, **node.__dict__)
        
        # Fetch relationships from databases
        edges = await self._fetch_relationships(entity_nodes)
        for edge in edges:
            kg.add_edge(edge.source, edge.target, 
                       edge_type=edge.edge_type,
                       confidence=edge.confidence,
                       **edge.properties)
        
        # Log edge status
        logger.info(f"After API calls: {kg.number_of_edges()} edges")
        
        # Add co-occurrence edges if no edges found
        if kg.number_of_edges() == 0 and len(entity_nodes) > 1:
            logger.warning("No edges from databases, adding co-occurrence edges")
            for i, node1 in enumerate(entity_nodes):
                for node2 in entity_nodes[i+1:]:
                    # Add edges between any entities (not just genes/proteins)
                    kg.add_edge(node1.node_id, node2.node_id,
                               edge_type='co_occurrence',
                               confidence=0.5,
                               source='text')
            logger.info(f"After co-occurrence: {kg.number_of_edges()} edges")
        
        # Add pathway nodes if requested
        if self.include_pathways:
            pathway_nodes = await self._fetch_pathway_nodes(entity_nodes)
            for node in pathway_nodes:
                kg.add_node(node.node_id, **node.__dict__)
            
            # Add pending edges (e.g., gene-pathway edges)
            for edge in self.pending_edges:
                kg.add_edge(edge.source, edge.target,
                           edge_type=edge.edge_type,
                           confidence=edge.confidence,
                           **edge.properties)
            self.pending_edges = []  # Clear pending edges
        
        # Add hallmark-specific pathways
        if hallmarks:
            hallmark_nodes = self._add_hallmark_pathways(hallmarks)
            for node in hallmark_nodes:
                kg.add_node(node.node_id, **node.__dict__)
        
        # Expand graph based on max_hops
        if self.max_hops > 0:
            kg = await self._expand_graph(kg, entity_nodes, self.max_hops)
        
        # Prune graph to max_neighbors
        kg = self._prune_graph(kg, entity_nodes)
        
        # Save cache
        self._save_cache()
        
        return kg
    
    def _create_entity_nodes(self, entities: List[BioEntity]) -> List[KGNode]:
        """Create KG nodes from extracted entities"""
        nodes = []
        
        for entity in entities:
            # Use primary database ID as node ID
            primary_db = self._get_primary_database(entity.entity_type)
            node_id = None
            
            if primary_db in entity.normalized_ids:
                node_id = f"{primary_db}:{entity.normalized_ids[primary_db]}"
            elif entity.normalized_ids:
                # Use first available ID
                db, id_val = next(iter(entity.normalized_ids.items()))
                node_id = f"{db}:{id_val}"
            else:
                # Use entity text as fallback
                node_id = f"ENTITY:{entity.text}"
            
            # Normalize gene/protein names for API calls
            normalized_name = entity.text.upper()
            if entity.entity_type in ['GENE', 'PROTEIN']:
                # Handle common variations
                if normalized_name.startswith('P') and len(normalized_name) > 1 and normalized_name[1:].isdigit():
                    normalized_name = 'TP' + normalized_name[1:]  # p53 -> TP53
            
            node = KGNode(
                node_id=node_id,
                node_type=entity.entity_type.lower(),
                name=normalized_name,
                original_text=entity.text,
                properties={
                    'normalized_ids': entity.normalized_ids,
                    'confidence': entity.confidence,
                    'context': entity.context
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _get_primary_database(self, entity_type: str) -> str:
        """Get primary database for entity type"""
        db_map = {
            'GENE': 'HGNC',
            'PROTEIN': 'UNIPROT',
            'CHEMICAL': 'CHEBI',
            'DISEASE': 'DO'
        }
        return db_map.get(entity_type, 'UNKNOWN')
    
    async def _fetch_relationships(self, nodes: List[KGNode]) -> List[KGEdge]:
        """Fetch relationships between entities from biological databases"""
        edges = []
        
        # Prepare async tasks for different databases
        tasks = []
        
        if 'STRING' in self.db_configs and self.db_configs['STRING']['enabled']:
            tasks.append(self._fetch_string_interactions(nodes))
        
        if 'KEGG' in self.db_configs and self.db_configs['KEGG']['enabled']:
            tasks.append(self._fetch_kegg_pathways(nodes))
        
        if 'Reactome' in self.db_configs and self.db_configs['Reactome']['enabled']:
            tasks.append(self._fetch_reactome_pathways(nodes))
        
        # Execute all tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    edges.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error fetching relationships: {result}")
        
        return edges
    
    async def _fetch_string_interactions(self, nodes: List[KGNode]) -> List[KGEdge]:
        """Fetch protein-protein interactions from STRING database"""
        edges = []
        
        # Get protein identifiers
        protein_ids = []
        id_to_node = {}
        
        for node in nodes:
            if node.node_type in ['gene', 'protein']:
                # Try to get UniProt or gene ID
                if 'UNIPROT' in node.properties.get('normalized_ids', {}):
                    protein_id = node.properties['normalized_ids']['UNIPROT']
                    protein_ids.append(protein_id)
                    id_to_node[protein_id] = node
                elif 'NCBI_GENE' in node.properties.get('normalized_ids', {}):
                    gene_id = node.properties['normalized_ids']['NCBI_GENE']
                    protein_ids.append(f"9606.ENSP{gene_id}")  # Human proteins
                    id_to_node[f"9606.ENSP{gene_id}"] = node
        
        if not protein_ids:
            return edges
        
        # Check cache
        cache_key = f"string_interactions_{'-'.join(sorted(protein_ids))}"
        if cache_key in self.api_cache:
            cached_data = self.api_cache[cache_key]
            return self._parse_string_response(cached_data, id_to_node)
        
        # Query STRING API
        try:
            async with aiohttp.ClientSession() as session:
                # STRING expects protein names or identifiers
                string_ids = []
                for node in nodes:
                    if node.node_type in ['gene', 'protein']:
                        # Map node to both with and without species prefix
                        string_ids.append(node.name.upper())
                        id_to_node[node.name.upper()] = node
                        id_to_node[f"9606.{node.name.upper()}"] = node
                
                if not string_ids:
                    logger.warning("No valid STRING identifiers found")
                    return edges
                
                url = "https://string-db.org/api/json/network"
                # STRING expects gene names without species prefix for the identifiers parameter
                gene_names = [node.name.upper() for node in nodes if node.node_type in ['gene', 'protein']]
                params = {
                    'identifiers': '%0d'.join(gene_names[:10]),  # Just gene names, limit to 10
                    'species': 9606,  # Human
                    'required_score': 400,  # Lower threshold
                    'network_type': 'functional',
                    'caller_identity': 'biokg_biobert'
                }
                
                logger.debug(f"STRING API call with {len(gene_names)} identifiers")
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        # STRING returns text/json, not application/json, so we need to parse manually
                        text = await response.text()
                        try:
                            data = json.loads(text)
                            logger.info(f"STRING returned {len(data)} interactions")
                            self.api_cache[cache_key] = data
                            edges = self._parse_string_response(data, id_to_node)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse STRING response: {e}")
                            logger.debug(f"Response text: {text[:500]}")
                    else:
                        error_text = await response.text()
                        logger.error(f"STRING API returned status {response.status}: {error_text[:200]}")
                        
        except Exception as e:
            logger.error(f"Error fetching STRING interactions: {e}")
        
        return edges
    
    def _parse_string_response(self, data: List[Dict], id_to_node: Dict) -> List[KGEdge]:
        """Parse STRING API response"""
        edges = []
        
        logger.debug(f"Parsing STRING response with {len(data)} items")
        
        for interaction in data:
            # STRING API returns preferredName_A and preferredName_B
            protein_a = None
            protein_b = None
            
            # Try different field names
            if 'preferredName_A' in interaction and 'preferredName_B' in interaction:
                protein_a = interaction['preferredName_A']
                protein_b = interaction['preferredName_B']
            elif 'stringId_A' in interaction and 'stringId_B' in interaction:
                protein_a = interaction['stringId_A'].split('.')[-1]
                protein_b = interaction['stringId_B'].split('.')[-1]
            
            if protein_a and protein_b:
                # Find matching nodes
                node_a = None
                node_b = None
                
                for key, node in id_to_node.items():
                    if protein_a.upper() in key.upper() or protein_a.upper() == node.name.upper():
                        node_a = node
                    if protein_b.upper() in key.upper() or protein_b.upper() == node.name.upper():
                        node_b = node
                
                if node_a and node_b and node_a != node_b:
                    edge = KGEdge(
                        source=node_a.node_id,
                        target=node_b.node_id,
                        edge_type='interacts',
                        properties={
                            'score': interaction.get('score', 0),
                            'database': 'STRING'
                        },
                        confidence=interaction.get('score', 0) / 1000.0
                    )
                    edges.append(edge)
                    logger.debug(f"Added edge: {node_a.name} -> {node_b.name}")
        
        logger.info(f"Parsed {len(edges)} edges from STRING")
        return edges
    
    async def _fetch_kegg_pathways(self, nodes: List[KGNode]) -> List[KGEdge]:
        """Fetch pathway memberships from KEGG"""
        edges = []
        
        # Get gene identifiers
        gene_ids = []
        id_to_node = {}
        
        for node in nodes:
            if node.node_type == 'gene' and 'NCBI_GENE' in node.properties.get('normalized_ids', {}):
                gene_id = node.properties['normalized_ids']['NCBI_GENE']
                gene_ids.append(gene_id)
                id_to_node[gene_id] = node
        
        if not gene_ids:
            return edges
        
        # Query KEGG for each gene
        async with aiohttp.ClientSession() as session:
            for gene_id in gene_ids:
                cache_key = f"kegg_gene_{gene_id}"
                
                if cache_key in self.api_cache:
                    pathways = self.api_cache[cache_key]
                else:
                    try:
                        url = f"{self.db_configs['KEGG']['api_endpoint']}/get/hsa:{gene_id}"
                        async with session.get(url) as response:
                            if response.status == 200:
                                text = await response.text()
                                pathways = self._parse_kegg_gene(text)
                                self.api_cache[cache_key] = pathways
                            else:
                                pathways = []
                    except Exception as e:
                        logger.error(f"Error fetching KEGG data for gene {gene_id}: {e}")
                        pathways = []
                
                # Create pathway membership edges
                for pathway_id in pathways:
                    edge = KGEdge(
                        source=id_to_node[gene_id].node_id,
                        target=f"KEGG:{pathway_id}",
                        edge_type='pathway_member',
                        properties={'database': 'KEGG'},
                        confidence=1.0
                    )
                    edges.append(edge)
        
        return edges
    
    def _parse_kegg_gene(self, kegg_text: str) -> List[str]:
        """Parse KEGG gene entry to extract pathway IDs"""
        pathways = []
        in_pathway_section = False
        
        for line in kegg_text.split('\n'):
            if line.startswith('PATHWAY'):
                in_pathway_section = True
                # Extract pathway ID from first line
                parts = line.split()
                if len(parts) > 1:
                    pathways.append(parts[1])
            elif in_pathway_section and line.startswith(' '):
                # Continuation of pathway section
                parts = line.strip().split()
                if parts:
                    pathways.append(parts[0])
            elif in_pathway_section and not line.startswith(' '):
                # End of pathway section
                break
        
        return pathways
    
    async def _fetch_reactome_pathways(self, nodes: List[KGNode]) -> List[KGEdge]:
        """Fetch pathway information from Reactome"""
        edges = []
        
        # Get protein identifiers
        protein_ids = []
        id_to_node = {}
        
        for node in nodes:
            if 'UNIPROT' in node.properties.get('normalized_ids', {}):
                protein_id = node.properties['normalized_ids']['UNIPROT']
                protein_ids.append(protein_id)
                id_to_node[protein_id] = node
        
        if not protein_ids:
            return edges
        
        # Query Reactome API
        cache_key = f"reactome_pathways_{'-'.join(sorted(protein_ids))}"
        
        if cache_key in self.api_cache:
            pathways = self.api_cache[cache_key]
        else:
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"{self.db_configs['Reactome']['api_endpoint']}/identifiers/projection"
                    headers = {'Content-Type': 'text/plain'}
                    data = '\n'.join(protein_ids)
                    
                    async with session.post(url, data=data, headers=headers) as response:
                        if response.status == 200:
                            pathways = await response.json()
                            self.api_cache[cache_key] = pathways
                        else:
                            pathways = []
            except Exception as e:
                logger.error(f"Error fetching Reactome pathways: {e}")
                pathways = []
        
        # Parse response and create edges
        for pathway in pathways:
            if 'identifier' in pathway and pathway['identifier'] in id_to_node:
                for reaction in pathway.get('pathways', []):
                    edge = KGEdge(
                        source=id_to_node[pathway['identifier']].node_id,
                        target=f"REACTOME:{reaction['stId']}",
                        edge_type='pathway_member',
                        properties={
                            'pathway_name': reaction.get('displayName', ''),
                            'database': 'Reactome'
                        },
                        confidence=1.0
                    )
                    edges.append(edge)
        
        return edges
    
    async def _fetch_pathway_nodes(self, entity_nodes: List[KGNode]) -> List[KGNode]:
        """Fetch pathway nodes connected to entities"""
        pathway_nodes = []
        pathway_ids = set()
        
        # Collect unique pathway IDs from edges
        # This is simplified - in practice would query the graph
        
        # Add KEGG pathway nodes
        for node in entity_nodes:
            if node.node_type == 'gene':
                # Add common cancer pathways based on gene names
                gene_pathway_map = {
                    'TP53': ['hsa04115'],  # p53 signaling
                    'P53': ['hsa04115'],
                    'EGFR': ['hsa04010', 'hsa04012'],  # MAPK, ErbB signaling
                    'VEGF': ['hsa04370'],  # VEGF signaling
                    'VEGFA': ['hsa04370'],
                    'BRCA1': ['hsa04110', 'hsa04115'],  # Cell cycle, p53
                    'BRCA2': ['hsa04110', 'hsa04115'],
                    'KRAS': ['hsa04010'],  # MAPK
                    'PIK3CA': ['hsa04151'],  # PI3K-Akt
                    'AKT1': ['hsa04151'],
                    'MYC': ['hsa04110'],  # Cell cycle
                    'PTEN': ['hsa04151']  # PI3K-Akt
                }
                
                gene_name = node.name.upper()
                if gene_name in gene_pathway_map:
                    pathway_ids.update(gene_pathway_map[gene_name])
        
        # Create pathway nodes
        for pathway_id in pathway_ids:
            if pathway_id.startswith('hsa'):
                # KEGG pathway
                node = KGNode(
                    node_id=f"KEGG:{pathway_id}",
                    node_type='pathway',
                    name=self._get_pathway_name(pathway_id),
                    properties={'database': 'KEGG', 'pathway_id': pathway_id}
                )
                pathway_nodes.append(node)
                
                # Add edges from genes to pathways
                for entity_node in entity_nodes:
                    if entity_node.node_type == 'gene' and entity_node.name.upper() in gene_pathway_map:
                        if pathway_id in gene_pathway_map[entity_node.name.upper()]:
                            edge = KGEdge(
                                source=entity_node.node_id,
                                target=f"KEGG:{pathway_id}",
                                edge_type='pathway_member',
                                properties={'database': 'KEGG'},
                                confidence=0.8
                            )
                            self.pending_edges.append(edge)
        
        return pathway_nodes
    
    def _get_pathway_name(self, pathway_id: str) -> str:
        """Get human-readable pathway name"""
        # Simplified mapping - in practice would query KEGG
        pathway_names = {
            'hsa04115': 'p53 signaling pathway',
            'hsa04010': 'MAPK signaling pathway',
            'hsa04210': 'Apoptosis',
            'hsa04310': 'Wnt signaling pathway',
            'hsa04151': 'PI3K-Akt signaling pathway',
            'hsa04110': 'Cell cycle',
            'hsa04066': 'HIF-1 signaling pathway',
            'hsa04370': 'VEGF signaling pathway'
        }
        return pathway_names.get(pathway_id, pathway_id)
    
    def _add_hallmark_pathways(self, hallmarks: List[str]) -> List[KGNode]:
        """Add cancer hallmark-specific pathway nodes"""
        hallmark_nodes = []
        
        for hallmark in hallmarks:
            # Create hallmark node
            hallmark_node = KGNode(
                node_id=f"HALLMARK:{hallmark}",
                node_type='hallmark',
                name=hallmark.replace('_', ' ').title(),
                properties={'hallmark': hallmark}
            )
            hallmark_nodes.append(hallmark_node)
            
            # Add associated pathways
            if hallmark in self.HALLMARK_PATHWAYS:
                for pathway_id in self.HALLMARK_PATHWAYS[hallmark]:
                    pathway_node = KGNode(
                        node_id=f"KEGG:{pathway_id}",
                        node_type='pathway',
                        name=self._get_pathway_name(pathway_id),
                        properties={
                            'database': 'KEGG',
                            'pathway_id': pathway_id,
                            'hallmark_associated': hallmark
                        }
                    )
                    hallmark_nodes.append(pathway_node)
        
        return hallmark_nodes
    
    async def _expand_graph(self, kg: nx.MultiDiGraph, seed_nodes: List[KGNode], 
                          max_hops: int) -> nx.MultiDiGraph:
        """Expand graph by fetching neighbors up to max_hops"""
        # This is a simplified implementation
        # In practice, would iteratively fetch neighbors from databases
        
        current_nodes = {node.node_id for node in seed_nodes}
        
        for hop in range(max_hops):
            # Get neighbors of current nodes
            new_nodes = set()
            
            for node_id in current_nodes:
                # Fetch neighbors from databases
                neighbors = await self._fetch_node_neighbors(node_id)
                new_nodes.update(neighbors)
            
            # Add new nodes to graph
            for node in new_nodes:
                if node.node_id not in kg:
                    kg.add_node(node.node_id, **node.__dict__)
            
            current_nodes = new_nodes
        
        return kg
    
    async def _fetch_node_neighbors(self, node_id: str) -> List[KGNode]:
        """Fetch neighbors of a node from databases"""
        # Simplified implementation
        neighbors = []
        
        # Would query appropriate database based on node type
        # For now, return empty list
        
        return neighbors
    
    def _prune_graph(self, kg: nx.MultiDiGraph, seed_nodes: List[KGNode]) -> nx.MultiDiGraph:
        """Prune graph to keep only most relevant neighbors"""
        if len(kg) <= self.max_neighbors:
            return kg
        
        # Calculate relevance scores for nodes
        seed_ids = {node.node_id for node in seed_nodes}
        relevance_scores = {}
        
        for node in kg.nodes():
            if node in seed_ids:
                relevance_scores[node] = 1.0
            else:
                # Score based on connectivity to seed nodes
                paths_to_seeds = 0
                for seed_id in seed_ids:
                    if nx.has_path(kg, node, seed_id):
                        paths_to_seeds += 1
                
                relevance_scores[node] = paths_to_seeds / len(seed_ids)
        
        # Keep top nodes by relevance
        sorted_nodes = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        nodes_to_keep = {node for node, _ in sorted_nodes[:self.max_neighbors]}
        
        # Create pruned graph
        pruned_kg = kg.subgraph(nodes_to_keep).copy()
        
        return pruned_kg