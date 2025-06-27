"""
Fixed Biological Knowledge Graph Construction Module v2

Additional fixes:
- STRING API now queries each protein individually or in correct batch format
- Reactome API endpoint corrected
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
import time

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
    
    # Common gene name mappings
    GENE_NAME_MAPPINGS = {
        'P53': 'TP53',
        'P21': 'CDKN1A',
        'P16': 'CDKN2A',
        'P27': 'CDKN1B',
        'HER2': 'ERBB2',
        'CMYC': 'MYC',
        'C-MYC': 'MYC',
        'N-MYC': 'MYCN',
        'L-MYC': 'MYCL',
        'BCL-2': 'BCL2',
        'BCL-XL': 'BCL2L1',
        'NF-KB': 'NFKB1',
        'NFKB': 'NFKB1',
        'P-AKT': 'AKT1',
        'PAKT': 'AKT1',
        'HIF1A': 'HIF1A',
        'HIF-1A': 'HIF1A',
        'HIF-1ALPHA': 'HIF1A',
        'HIF1-ALPHA': 'HIF1A',
        'VEGF-A': 'VEGFA',
        'TGF-B': 'TGFB1',
        'TGFB': 'TGFB1',
        'TNF-A': 'TNF',
        'TNFA': 'TNF',
        'IL-6': 'IL6',
        'IL6': 'IL6',
        'STAT-3': 'STAT3',
        'PDL1': 'CD274',
        'PD-L1': 'CD274',
        'PDL-1': 'CD274',
        'PD1': 'PDCD1',
        'PD-1': 'PDCD1',
        'CTLA4': 'CTLA4',
        'CTLA-4': 'CTLA4'
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
        
        # Rate limiting for KEGG
        self.last_kegg_call = 0
        self.kegg_delay = 0.35  # ~3 calls per second
        
    def _init_database_clients(self):
        """Initialize API clients for biological databases"""
        self.db_configs = {}
        
        for db in self.databases:
            if db['enabled']:
                self.db_configs[db['name']] = db
                
        # STRING specific settings
        if 'STRING' in self.db_configs:
            self.string_confidence = self.db_configs['STRING'].get('confidence_threshold', 400)
    
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
    
    def _normalize_gene_name(self, name: str) -> str:
        """Normalize gene names to standard symbols"""
        name = name.upper().strip()
        
        # Check mapping table first
        if name in self.GENE_NAME_MAPPINGS:
            return self.GENE_NAME_MAPPINGS[name]
        
        # Handle p-prefixed proteins
        if name.startswith('P') and len(name) > 1 and name[1:].isdigit():
            # p53 -> TP53, p21 -> CDKN1A, etc.
            if name == 'P53':
                return 'TP53'
            elif name == 'P21':
                return 'CDKN1A'
            elif name == 'P16':
                return 'CDKN2A'
            elif name == 'P27':
                return 'CDKN1B'
        
        # Remove hyphens for some genes
        if '-' in name:
            no_hyphen = name.replace('-', '')
            if no_hyphen in self.GENE_NAME_MAPPINGS:
                return self.GENE_NAME_MAPPINGS[no_hyphen]
        
        return name
    
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
        seen_names = set()  # Track seen gene names to avoid duplicates
        
        for entity in entities:
            # Normalize gene/protein names
            normalized_name = entity.text
            if entity.entity_type in ['GENE', 'PROTEIN']:
                normalized_name = self._normalize_gene_name(entity.text)
            
            # Skip if we've already seen this normalized name
            if normalized_name in seen_names:
                continue
            seen_names.add(normalized_name)
            
            # Use normalized name as primary identifier
            node_id = f"{entity.entity_type}:{normalized_name}"
            
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
        
        # Get gene/protein names
        gene_names = []
        name_to_node = {}
        
        for node in nodes:
            if node.node_type in ['gene', 'protein']:
                gene_names.append(node.name)
                name_to_node[node.name] = node
                # Also map lowercase version
                name_to_node[node.name.lower()] = node
        
        if not gene_names:
            logger.debug("No gene/protein nodes for STRING query")
            return edges
        
        # STRING works better with newline-separated identifiers in POST requests
        # or with the network endpoint using specific formatting
        
        # Check cache
        cache_key = f"string_interactions_v2_{'-'.join(sorted(gene_names))}"
        if cache_key in self.api_cache:
            cached_data = self.api_cache[cache_key]
            return self._parse_string_response(cached_data, name_to_node)
        
        # Query STRING API - use POST method for multiple proteins
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://string-db.org/api/tsv/network"
                
                # Try POST method with form data
                data = {
                    'identifiers': '\r'.join(gene_names),  # Carriage return separated
                    'species': '9606',  # Human
                    'required_score': str(self.string_confidence),
                    'network_type': 'functional',
                    'caller_identity': 'biokg_biobert_v2'
                }
                
                logger.info(f"STRING API POST call with {len(gene_names)} identifiers")
                logger.debug(f"STRING identifiers: {gene_names}")
                
                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        text = await response.text()
                        # Parse TSV response
                        lines = text.strip().split('\n')
                        if len(lines) > 1:  # Has header + data
                            data_list = []
                            headers = lines[0].split('\t')
                            for line in lines[1:]:
                                values = line.split('\t')
                                if len(values) == len(headers):
                                    data_list.append(dict(zip(headers, values)))
                            
                            logger.info(f"STRING returned {len(data_list)} interactions")
                            self.api_cache[cache_key] = data_list
                            edges = self._parse_string_response(data_list, name_to_node)
                        else:
                            logger.warning("STRING returned no interactions")
                    else:
                        # If POST fails, try individual queries
                        logger.warning(f"STRING POST failed with status {response.status}, trying individual queries")
                        edges = await self._fetch_string_individual(gene_names, name_to_node)
                        
        except Exception as e:
            logger.error(f"Error fetching STRING interactions: {e}", exc_info=True)
            # Fallback to individual queries
            edges = await self._fetch_string_individual(gene_names, name_to_node)
        
        return edges
    
    async def _fetch_string_individual(self, gene_names: List[str], name_to_node: Dict) -> List[KGEdge]:
        """Fetch STRING interactions one protein at a time"""
        all_interactions = {}
        edges = []
        
        async with aiohttp.ClientSession() as session:
            for gene_name in gene_names[:10]:  # Limit to 10 to avoid too many requests
                try:
                    url = "https://string-db.org/api/tsv/interaction_partners"
                    params = {
                        'identifiers': gene_name,
                        'species': '9606',
                        'required_score': str(self.string_confidence),
                        'limit': '20'  # Limit partners per protein
                    }
                    
                    logger.debug(f"STRING individual query for {gene_name}")
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            text = await response.text()
                            lines = text.strip().split('\n')
                            if len(lines) > 1:
                                headers = lines[0].split('\t')
                                for line in lines[1:]:
                                    values = line.split('\t')
                                    if len(values) == len(headers):
                                        interaction = dict(zip(headers, values))
                                        # Check if partner is in our node list
                                        partner = interaction.get('preferredName_B', '')
                                        if partner in name_to_node:
                                            key = tuple(sorted([gene_name, partner]))
                                            if key not in all_interactions:
                                                all_interactions[key] = interaction
                        
                    await asyncio.sleep(0.1)  # Brief delay between requests
                    
                except Exception as e:
                    logger.error(f"Error fetching STRING data for {gene_name}: {e}")
        
        # Convert to edges
        for (gene1, gene2), interaction in all_interactions.items():
            node1 = name_to_node.get(gene1)
            node2 = name_to_node.get(gene2)
            
            if node1 and node2:
                score = float(interaction.get('score', 0))
                edge = KGEdge(
                    source=node1.node_id,
                    target=node2.node_id,
                    edge_type='interacts',
                    properties={
                        'score': score,
                        'database': 'STRING',
                        'method': 'individual_query'
                    },
                    confidence=score / 1000.0
                )
                edges.append(edge)
        
        logger.info(f"STRING individual queries found {len(edges)} interactions")
        return edges
    
    def _parse_string_response(self, data: List[Dict], name_to_node: Dict) -> List[KGEdge]:
        """Parse STRING API response"""
        edges = []
        
        logger.debug(f"Parsing STRING response with {len(data)} items")
        
        for interaction in data:
            # STRING TSV format has these columns
            protein_a = interaction.get('preferredName_A', '').strip()
            protein_b = interaction.get('preferredName_B', '').strip()
            score = float(interaction.get('score', 0))
            
            if protein_a and protein_b and score > 0:
                # Find matching nodes
                node_a = name_to_node.get(protein_a) or name_to_node.get(protein_a.lower())
                node_b = name_to_node.get(protein_b) or name_to_node.get(protein_b.lower())
                
                if node_a and node_b and node_a != node_b:
                    edge = KGEdge(
                        source=node_a.node_id,
                        target=node_b.node_id,
                        edge_type='interacts',
                        properties={
                            'score': score,
                            'database': 'STRING',
                            'experimental_score': float(interaction.get('experimental', 0)),
                            'database_score': float(interaction.get('database', 0)),
                            'text_mining_score': float(interaction.get('textmining', 0))
                        },
                        confidence=score / 1000.0  # STRING scores are 0-1000
                    )
                    edges.append(edge)
                    logger.debug(f"Added edge: {node_a.name} <-> {node_b.name} (score: {score})")
        
        logger.info(f"Parsed {len(edges)} edges from STRING")
        return edges
    
    async def _fetch_kegg_pathways(self, nodes: List[KGNode]) -> List[KGEdge]:
        """Fetch pathway memberships from KEGG"""
        edges = []
        
        # KEGG gene ID mapping for common cancer genes
        gene_to_kegg_id = {
            'TP53': '7157',
            'EGFR': '1956',
            'ERBB2': '2064',
            'KRAS': '3845',
            'BRAF': '673',
            'PIK3CA': '5290',
            'PTEN': '5728',
            'AKT1': '207',
            'MYC': '4609',
            'VEGFA': '7422',
            'BCL2': '596',
            'BRCA1': '672',
            'BRCA2': '675',
            'CDKN2A': '1029',
            'CDKN1A': '1026',
            'CDKN1B': '1027',
            'RB1': '5925',
            'MDM2': '4193',
            'NFKB1': '4790',
            'STAT3': '6774',
            'HIF1A': '3091',
            'TGFB1': '7040',
            'TNF': '7124',
            'IL6': '3569',
            'CD274': '29126',  # PD-L1
            'PDCD1': '5133',   # PD-1
            'CTLA4': '1493'
        }
        
        # Process each gene node
        for node in nodes:
            if node.node_type == 'gene':
                gene_name = node.name.upper()
                kegg_id = gene_to_kegg_id.get(gene_name)
                
                if not kegg_id:
                    # Try to get from normalized IDs
                    if 'NCBI_GENE' in node.properties.get('normalized_ids', {}):
                        kegg_id = node.properties['normalized_ids']['NCBI_GENE']
                
                if kegg_id:
                    cache_key = f"kegg_gene_{kegg_id}"
                    
                    if cache_key in self.api_cache:
                        pathways = self.api_cache[cache_key]
                    else:
                        # Rate limiting for KEGG
                        current_time = time.time()
                        time_since_last = current_time - self.last_kegg_call
                        if time_since_last < self.kegg_delay:
                            await asyncio.sleep(self.kegg_delay - time_since_last)
                        
                        try:
                            async with aiohttp.ClientSession() as session:
                                url = f"https://rest.kegg.jp/get/hsa:{kegg_id}"
                                logger.debug(f"KEGG API call for gene {gene_name} (ID: {kegg_id})")
                                
                                async with session.get(url) as response:
                                    self.last_kegg_call = time.time()
                                    
                                    if response.status == 200:
                                        text = await response.text()
                                        pathways = self._parse_kegg_gene(text)
                                        self.api_cache[cache_key] = pathways
                                        logger.debug(f"KEGG found {len(pathways)} pathways for {gene_name}")
                                    else:
                                        logger.warning(f"KEGG API returned status {response.status} for gene {kegg_id}")
                                        pathways = []
                        except Exception as e:
                            logger.error(f"Error fetching KEGG data for gene {kegg_id}: {e}")
                            pathways = []
                    
                    # Create pathway membership edges
                    for pathway_id in pathways:
                        pathway_node_id = f"KEGG:{pathway_id}"
                        
                        # Create pathway node
                        pathway_node = KGNode(
                            node_id=pathway_node_id,
                            node_type='pathway',
                            name=self._get_pathway_name(pathway_id),
                            properties={'database': 'KEGG', 'pathway_id': pathway_id}
                        )
                        self.pending_edges.append(pathway_node)
                        
                        # Create edge
                        edge = KGEdge(
                            source=node.node_id,
                            target=pathway_node_id,
                            edge_type='pathway_member',
                            properties={'database': 'KEGG'},
                            confidence=1.0
                        )
                        edges.append(edge)
        
        logger.info(f"KEGG pathways: found {len(edges)} pathway memberships")
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
        """Fetch pathway information from Reactome using ContentService API"""
        edges = []
        
        # Collect gene names
        gene_names = []
        name_to_node = {}
        
        for node in nodes:
            if node.node_type in ['gene', 'protein']:
                gene_names.append(node.name)
                name_to_node[node.name] = node
        
        if not gene_names:
            return edges
        
        # Query Reactome ContentService API
        cache_key = f"reactome_pathways_v3_{'-'.join(sorted(gene_names))}"
        
        if cache_key in self.api_cache:
            all_pathways = self.api_cache[cache_key]
        else:
            all_pathways = {}
            try:
                async with aiohttp.ClientSession() as session:
                    # Search for each gene individually to get Reactome entities
                    for gene_name in gene_names[:10]:  # Limit to avoid too many requests
                        # Step 1: Search for the gene/protein
                        search_url = f"https://reactome.org/ContentService/search/query?query={gene_name}&species=Homo%20sapiens"
                        
                        logger.debug(f"Reactome search for {gene_name}")
                        
                        async with session.get(search_url, headers={'Accept': 'application/json'}) as search_response:
                            if search_response.status == 200:
                                search_data = await search_response.json()
                                
                                # Find protein/gene entities from search results
                                entity_ids = []
                                for result in search_data.get('results', []):
                                    if result.get('typeName') in ['Protein', 'Gene', 'EntityWithAccessionedSequence', 'ReferenceGeneProduct']:
                                        if 'stId' in result:
                                            entity_ids.append(result['stId'])
                                        elif 'dbId' in result:
                                            entity_ids.append(str(result['dbId']))
                                
                                # Step 2: Get pathways for each entity
                                pathways_for_gene = []
                                for entity_id in entity_ids[:3]:  # Limit entities per gene
                                    pathways_url = f"https://reactome.org/ContentService/data/pathways/low/entity/{entity_id}?species=9606"
                                    
                                    async with session.get(pathways_url, headers={'Accept': 'application/json'}) as pathway_response:
                                        if pathway_response.status == 200:
                                            pathways = await pathway_response.json()
                                            logger.debug(f"Reactome found {len(pathways)} pathways for {gene_name} (entity: {entity_id})")
                                            
                                            # Add unique pathways
                                            seen_ids = set()
                                            for pathway in pathways:
                                                if pathway.get('stId') not in seen_ids:
                                                    seen_ids.add(pathway.get('stId'))
                                                    pathways_for_gene.append({
                                                        'stId': pathway.get('stId', ''),
                                                        'displayName': pathway.get('displayName', ''),
                                                        'dbId': pathway.get('dbId', '')
                                                    })
                                
                                if pathways_for_gene:
                                    all_pathways[gene_name] = pathways_for_gene[:20]  # Limit pathways per gene
                            else:
                                logger.warning(f"Reactome search failed for {gene_name}: status {search_response.status}")
                        
                        # Brief delay between searches
                        await asyncio.sleep(0.1)
                            
                self.api_cache[cache_key] = all_pathways
                
            except Exception as e:
                logger.error(f"Error fetching Reactome pathways: {e}", exc_info=True)
        
        # Create edges for pathways
        for gene_name, pathways in all_pathways.items():
            if gene_name in name_to_node:
                node = name_to_node[gene_name]
                
                for pathway in pathways:
                    pathway_id = pathway.get('stId', '')
                    pathway_name = pathway.get('displayName', '')
                    
                    if pathway_id:
                        pathway_node_id = f"REACTOME:{pathway_id}"
                        
                        # Create pathway node
                        pathway_node = KGNode(
                            node_id=pathway_node_id,
                            node_type='pathway',
                            name=pathway_name,
                            properties={
                                'database': 'Reactome',
                                'pathway_id': pathway_id,
                                'species': 'Homo sapiens'
                            }
                        )
                        self.pending_edges.append(pathway_node)
                        
                        # Create edge
                        edge = KGEdge(
                            source=node.node_id,
                            target=pathway_node_id,
                            edge_type='pathway_member',
                            properties={
                                'database': 'Reactome',
                                'pathway_name': pathway_name
                            },
                            confidence=1.0
                        )
                        edges.append(edge)
        
        logger.info(f"Reactome pathways: found {len(edges)} pathway memberships")
        return edges
    
    async def _fetch_pathway_nodes(self, entity_nodes: List[KGNode]) -> List[KGNode]:
        """Return pathway nodes that were created during edge fetching"""
        # The pending_edges list contains pathway nodes that need to be added
        pathway_nodes = [edge for edge in self.pending_edges if isinstance(edge, KGNode)]
        
        # Clear pathway nodes from pending edges
        self.pending_edges = [edge for edge in self.pending_edges if isinstance(edge, KGEdge)]
        
        # Also add some cancer-specific pathways based on extracted genes
        gene_names = [node.name.upper() for node in entity_nodes if node.node_type == 'gene']
        
        # Map of genes to their key pathways
        gene_pathway_map = {
            'TP53': ['hsa04115'],  # p53 signaling
            'EGFR': ['hsa04010', 'hsa04012'],  # MAPK, ErbB signaling
            'VEGFA': ['hsa04370'],  # VEGF signaling
            'BRCA1': ['hsa04110', 'hsa04115'],  # Cell cycle, p53
            'BRCA2': ['hsa04110', 'hsa04115'],
            'KRAS': ['hsa04010'],  # MAPK
            'PIK3CA': ['hsa04151'],  # PI3K-Akt
            'AKT1': ['hsa04151'],
            'MYC': ['hsa04110'],  # Cell cycle
            'PTEN': ['hsa04151'],  # PI3K-Akt
            'HIF1A': ['hsa04066'],  # HIF-1 signaling
            'NFKB1': ['hsa04064'],  # NF-kappa B signaling
            'BCL2': ['hsa04210'],  # Apoptosis
            'TGFB1': ['hsa04350'],  # TGF-beta signaling
            'TNF': ['hsa04668'],  # TNF signaling
            'IL6': ['hsa04630'],  # JAK-STAT signaling
            'STAT3': ['hsa04630'],
            'CD274': ['hsa04514'],  # Cell adhesion molecules
            'PDCD1': ['hsa04514'],
            'CTLA4': ['hsa04514']
        }
        
        added_pathways = set()
        for gene_name in gene_names:
            if gene_name in gene_pathway_map:
                for pathway_id in gene_pathway_map[gene_name]:
                    if pathway_id not in added_pathways:
                        added_pathways.add(pathway_id)
                        
                        pathway_node = KGNode(
                            node_id=f"KEGG:{pathway_id}",
                            node_type='pathway',
                            name=self._get_pathway_name(pathway_id),
                            properties={'database': 'KEGG', 'pathway_id': pathway_id}
                        )
                        pathway_nodes.append(pathway_node)
                        
                        # Add edge from gene to pathway
                        for entity_node in entity_nodes:
                            if entity_node.node_type == 'gene' and entity_node.name.upper() == gene_name:
                                edge = KGEdge(
                                    source=entity_node.node_id,
                                    target=f"KEGG:{pathway_id}",
                                    edge_type='pathway_member',
                                    properties={'database': 'KEGG', 'inferred': True},
                                    confidence=0.8
                                )
                                self.pending_edges.append(edge)
        
        return pathway_nodes
    
    def _get_pathway_name(self, pathway_id: str) -> str:
        """Get human-readable pathway name"""
        # Extended pathway name mapping
        pathway_names = {
            'hsa04115': 'p53 signaling pathway',
            'hsa04010': 'MAPK signaling pathway',
            'hsa04210': 'Apoptosis',
            'hsa04310': 'Wnt signaling pathway',
            'hsa04151': 'PI3K-Akt signaling pathway',
            'hsa04110': 'Cell cycle',
            'hsa04066': 'HIF-1 signaling pathway',
            'hsa04370': 'VEGF signaling pathway',
            'hsa04012': 'ErbB signaling pathway',
            'hsa04350': 'TGF-beta signaling pathway',
            'hsa04630': 'JAK-STAT signaling pathway',
            'hsa04064': 'NF-kappa B signaling pathway',
            'hsa04668': 'TNF signaling pathway',
            'hsa04620': 'Toll-like receptor signaling pathway',
            'hsa04514': 'Cell adhesion molecules',
            'hsa04650': 'Natural killer cell mediated cytotoxicity',
            'hsa04660': 'T cell receptor signaling pathway',
            'hsa04672': 'Intestinal immune network for IgA production',
            'hsa04510': 'Focal adhesion',
            'hsa04810': 'Regulation of actin cytoskeleton',
            'hsa04670': 'Leukocyte transendothelial migration',
            'hsa03430': 'Mismatch repair',
            'hsa03440': 'Homologous recombination',
            'hsa04152': 'AMPK signaling pathway',
            'hsa00010': 'Glycolysis / Gluconeogenesis',
            'hsa00020': 'Citrate cycle (TCA cycle)',
            'hsa04215': 'Apoptosis - multiple species'
        }
        return pathway_names.get(pathway_id, f"KEGG pathway {pathway_id}")
    
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