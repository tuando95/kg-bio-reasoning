"""
Simplified Knowledge Graph Builder for Development/Testing

This version creates mock knowledge graphs to ensure the pipeline works
while we debug the actual API integrations.
"""

import networkx as nx
import torch
import logging
from typing import List, Dict, Set, Optional
import random

from .bio_entity_extractor import BioEntity

logger = logging.getLogger(__name__)


class SimpleKGBuilder:
    """
    Simplified KG builder that creates mock graphs for testing.
    
    This ensures the pipeline works end-to-end while we debug
    the actual biological database integrations.
    """
    
    # Common cancer-related pathways
    CANCER_PATHWAYS = {
        'p53_signaling': 'hsa04115',
        'mapk_signaling': 'hsa04010',
        'pi3k_akt_signaling': 'hsa04151',
        'cell_cycle': 'hsa04110',
        'apoptosis': 'hsa04210',
        'vegf_signaling': 'hsa04370',
        'wnt_signaling': 'hsa04310',
        'tgf_beta_signaling': 'hsa04350'
    }
    
    # Gene to pathway mappings
    GENE_PATHWAYS = {
        'TP53': ['p53_signaling', 'apoptosis', 'cell_cycle'],
        'EGFR': ['mapk_signaling', 'pi3k_akt_signaling'],
        'VEGF': ['vegf_signaling', 'mapk_signaling'],
        'VEGFA': ['vegf_signaling', 'mapk_signaling'],
        'BRCA1': ['cell_cycle', 'p53_signaling'],
        'BRCA2': ['cell_cycle', 'p53_signaling'],
        'KRAS': ['mapk_signaling', 'pi3k_akt_signaling'],
        'PIK3CA': ['pi3k_akt_signaling', 'mapk_signaling'],
        'AKT1': ['pi3k_akt_signaling', 'apoptosis'],
        'MYC': ['cell_cycle', 'apoptosis'],
        'PTEN': ['pi3k_akt_signaling', 'cell_cycle'],
        'CDK4': ['cell_cycle'],
        'CDK6': ['cell_cycle'],
        'RB': ['cell_cycle'],
        'RB1': ['cell_cycle'],
        'BCL2': ['apoptosis'],
        'BAX': ['apoptosis'],
        'CASP3': ['apoptosis'],
        'CASP9': ['apoptosis']
    }
    
    # Protein-protein interactions
    INTERACTIONS = {
        'TP53': ['MDM2', 'CDKN1A', 'BAX', 'BBC3'],
        'EGFR': ['SRC', 'GRB2', 'SOS1', 'KRAS'],
        'VEGFA': ['FLT1', 'KDR', 'NRP1'],
        'BRCA1': ['BRCA2', 'RAD51', 'PALB2'],
        'KRAS': ['BRAF', 'RAF1', 'PIK3CA'],
        'AKT1': ['MTOR', 'GSK3B', 'MDM2'],
        'MYC': ['MAX', 'MYCN'],
        'PTEN': ['PIK3CA', 'AKT1']
    }
    
    def __init__(self, config: Dict):
        """Initialize the simplified KG builder."""
        self.config = config
        self.max_neighbors = config.get('graph_construction', {}).get('max_neighbors', 20)
        logger.info("Initialized Simplified KG Builder (for testing)")
    
    async def build_knowledge_graph(self, entities: List[BioEntity], 
                                  hallmarks: Optional[List[str]] = None) -> nx.MultiDiGraph:
        """
        Build a simplified knowledge graph for testing.
        
        Args:
            entities: List of extracted biological entities
            hallmarks: Optional list of cancer hallmarks
            
        Returns:
            NetworkX MultiDiGraph
        """
        kg = nx.MultiDiGraph()
        
        # Add entity nodes
        entity_genes = []
        for i, entity in enumerate(entities):
            node_id = f"ENTITY_{i}_{entity.text.upper()}"
            
            # Simplified gene name extraction
            gene_name = entity.text.upper()
            # Handle common variations
            if gene_name.startswith('P') and gene_name[1:].isdigit():
                gene_name = 'TP' + gene_name[1:]
            
            kg.add_node(
                node_id,
                node_type='gene' if entity.entity_type in ['GENE', 'PROTEIN'] else entity.entity_type.lower(),
                name=entity.text,
                gene_name=gene_name,
                properties={'confidence': entity.confidence}
            )
            
            if entity.entity_type in ['GENE', 'PROTEIN']:
                entity_genes.append((node_id, gene_name))
        
        # Add pathway nodes based on genes
        added_pathways = set()
        for node_id, gene_name in entity_genes:
            if gene_name in self.GENE_PATHWAYS:
                for pathway_name in self.GENE_PATHWAYS[gene_name]:
                    pathway_id = f"KEGG:{self.CANCER_PATHWAYS.get(pathway_name, pathway_name)}"
                    
                    if pathway_id not in added_pathways:
                        kg.add_node(
                            pathway_id,
                            node_type='pathway',
                            name=pathway_name.replace('_', ' ').title(),
                            properties={'database': 'KEGG'}
                        )
                        added_pathways.add(pathway_id)
                    
                    # Add gene-pathway membership edge
                    kg.add_edge(
                        node_id, pathway_id,
                        edge_type='pathway_member',
                        confidence=0.9,
                        properties={'source': 'KEGG'}
                    )
        
        # Add protein-protein interactions
        gene_to_node = {gene: node for node, gene in entity_genes}
        
        for node_id, gene_name in entity_genes:
            if gene_name in self.INTERACTIONS:
                for interactor in self.INTERACTIONS[gene_name]:
                    # Check if interactor is in our entities
                    if interactor in gene_to_node:
                        target_node = gene_to_node[interactor]
                        kg.add_edge(
                            node_id, target_node,
                            edge_type='interacts',
                            confidence=0.8,
                            properties={'source': 'STRING', 'score': 800}
                        )
                    else:
                        # Add interactor as new node
                        interactor_id = f"PROTEIN_{interactor}"
                        if interactor_id not in kg:
                            kg.add_node(
                                interactor_id,
                                node_type='protein',
                                name=interactor,
                                properties={'source': 'STRING'}
                            )
                        kg.add_edge(
                            node_id, interactor_id,
                            edge_type='interacts',
                            confidence=0.7,
                            properties={'source': 'STRING', 'score': 700}
                        )
        
        # Add hallmark nodes if provided
        if hallmarks:
            for hallmark in hallmarks:
                hallmark_id = f"HALLMARK:{hallmark}"
                kg.add_node(
                    hallmark_id,
                    node_type='hallmark',
                    name=hallmark.replace('_', ' ').title(),
                    properties={'hallmark': hallmark}
                )
                
                # Connect hallmark to relevant pathways
                hallmark_pathways = {
                    'resisting_cell_death': ['apoptosis', 'p53_signaling'],
                    'sustaining_proliferative_signaling': ['mapk_signaling', 'pi3k_akt_signaling'],
                    'inducing_angiogenesis': ['vegf_signaling'],
                    'evading_growth_suppressors': ['cell_cycle', 'p53_signaling'],
                    'activating_invasion_metastasis': ['tgf_beta_signaling', 'wnt_signaling']
                }
                
                if hallmark in hallmark_pathways:
                    for pathway_name in hallmark_pathways[hallmark]:
                        pathway_id = f"KEGG:{self.CANCER_PATHWAYS.get(pathway_name, pathway_name)}"
                        if pathway_id in kg:
                            kg.add_edge(
                                hallmark_id, pathway_id,
                                edge_type='associated_with',
                                confidence=0.9,
                                properties={'source': 'literature'}
                            )
        
        # Add some GO terms for completeness
        go_terms = {
            'apoptosis': 'GO:0006915',
            'cell_proliferation': 'GO:0008283',
            'angiogenesis': 'GO:0001525',
            'cell_cycle': 'GO:0007049'
        }
        
        for name, go_id in go_terms.items():
            if random.random() < 0.3:  # Add some randomness
                go_node_id = f"GO:{go_id}"
                kg.add_node(
                    go_node_id,
                    node_type='go_term',
                    name=name,
                    properties={'ontology': 'biological_process'}
                )
                
                # Connect to relevant genes
                for node_id, gene_name in entity_genes[:2]:  # Limit connections
                    if random.random() < 0.5:
                        kg.add_edge(
                            node_id, go_node_id,
                            edge_type='associated_with',
                            confidence=0.6,
                            properties={'source': 'GO'}
                        )
        
        logger.info(f"Built simplified KG with {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges")
        return kg