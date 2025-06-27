"""
Biological Knowledge Integration Pipeline

Main pipeline that orchestrates:
1. Entity extraction from text
2. Knowledge graph construction
3. Integration with downstream models
"""

import logging
from typing import List, Dict, Tuple, Optional
import torch
import networkx as nx
from dataclasses import dataclass
import asyncio

from .bio_entity_extractor import BioEntityExtractor, BioEntity
from .kg_builder import BiologicalKGBuilder, KGNode, KGEdge
from .simple_kg_builder import SimpleKGBuilder

logger = logging.getLogger(__name__)


@dataclass
class KGPipelineOutput:
    """Output of the knowledge graph pipeline"""
    text: str
    entities: List[BioEntity]
    knowledge_graph: nx.MultiDiGraph
    node_embeddings: Optional[torch.Tensor] = None
    graph_embedding: Optional[torch.Tensor] = None
    entity_to_token_mapping: Optional[Dict[int, List[int]]] = None


class BiologicalKGPipeline:
    """
    Main pipeline for biological knowledge graph construction and integration.
    
    This pipeline:
    1. Extracts biological entities from text
    2. Constructs sentence-specific knowledge graphs
    3. Prepares graph data for downstream model integration
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration dictionary containing KG settings
        """
        self.config = config
        
        # Initialize components
        logger.info("Initializing Biological Knowledge Graph Pipeline...")
        
        # Entity extractor
        self.entity_extractor = BioEntityExtractor(
            config.get('entity_extraction', {})
        )
        
        # Knowledge graph builder
        use_simple_builder = config.get('use_simple_kg_builder', True)  # Default to simple for now
        if use_simple_builder:
            logger.info("Using simplified KG builder for testing")
            self.kg_builder = SimpleKGBuilder(config)
        else:
            logger.info("Using full biological KG builder with API calls")
            self.kg_builder = BiologicalKGBuilder(config)
        
        # Cache for processed examples
        self.cache = {}
        
        logger.info("Pipeline initialization complete")
    
    def process_text(self, text: str, 
                    hallmarks: Optional[List[str]] = None,
                    use_cache: bool = True) -> KGPipelineOutput:
        """
        Process a text through the complete pipeline.
        
        Args:
            text: Input text to process
            hallmarks: Optional list of cancer hallmarks for pathway inclusion
            use_cache: Whether to use cached results
            
        Returns:
            KGPipelineOutput containing entities and knowledge graph
        """
        # Check cache
        cache_key = self._get_cache_key(text, hallmarks)
        if use_cache and cache_key in self.cache:
            logger.debug(f"Using cached result for text: {text[:50]}...")
            return self.cache[cache_key]
        
        # Extract entities
        logger.debug("Extracting biological entities...")
        entities = self.entity_extractor.extract_entities(text)
        logger.info(f"Extracted {len(entities)} entities from text")
        
        # Build knowledge graph
        logger.debug("Building knowledge graph...")
        kg = asyncio.run(self.kg_builder.build_knowledge_graph(entities, hallmarks))
        logger.info(f"Built knowledge graph with {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges")
        
        # Create output
        output = KGPipelineOutput(
            text=text,
            entities=entities,
            knowledge_graph=kg
        )
        
        # Cache result
        if use_cache:
            self.cache[cache_key] = output
        
        return output
    
    def process_batch(self, texts: List[str],
                     hallmarks_list: Optional[List[List[str]]] = None,
                     use_cache: bool = True) -> List[KGPipelineOutput]:
        """
        Process a batch of texts through the pipeline.
        
        Args:
            texts: List of input texts
            hallmarks_list: Optional list of hallmark lists for each text
            use_cache: Whether to use cached results
            
        Returns:
            List of KGPipelineOutput objects
        """
        outputs = []
        
        if hallmarks_list is None:
            hallmarks_list = [None] * len(texts)
        
        for text, hallmarks in zip(texts, hallmarks_list):
            output = self.process_text(text, hallmarks, use_cache)
            outputs.append(output)
        
        return outputs
    
    def map_entities_to_tokens(self, entities: List[BioEntity], 
                             tokenized_text: Dict) -> Dict[int, List[int]]:
        """
        Map biological entities to token positions in tokenized text.
        
        Args:
            entities: List of extracted entities
            tokenized_text: Tokenized text with offset mapping
            
        Returns:
            Dictionary mapping entity index to token indices
        """
        entity_to_tokens = {}
        
        # Get offset mapping from tokenizer output
        offset_mapping = tokenized_text.get('offset_mapping', [])
        
        for idx, entity in enumerate(entities):
            token_indices = []
            
            # Find tokens that overlap with entity span
            for token_idx, (start, end) in enumerate(offset_mapping):
                if start is None or end is None:
                    continue
                    
                # Check if token overlaps with entity
                if (start >= entity.start and start < entity.end) or \
                   (end > entity.start and end <= entity.end) or \
                   (start <= entity.start and end >= entity.end):
                    token_indices.append(token_idx)
            
            if token_indices:
                entity_to_tokens[idx] = token_indices
        
        return entity_to_tokens
    
    def prepare_graph_features(self, kg: nx.MultiDiGraph) -> Dict[str, torch.Tensor]:
        """
        Prepare graph features for GNN processing.
        
        Args:
            kg: Knowledge graph
            
        Returns:
            Dictionary containing node features, edge indices, and edge attributes
        """
        # Create node index mapping
        node_to_idx = {node: idx for idx, node in enumerate(kg.nodes())}
        
        # Initialize node features (placeholder - would use actual embeddings)
        num_nodes = len(node_to_idx)
        node_features = torch.zeros(num_nodes, 768)  # Using BERT dimension
        
        # Create edge index tensor
        edge_list = []
        edge_types = []
        edge_weights = []
        
        for source, target, data in kg.edges(data=True):
            source_idx = node_to_idx[source]
            target_idx = node_to_idx[target]
            
            edge_list.append([source_idx, target_idx])
            edge_types.append(data.get('edge_type', 'unknown'))
            edge_weights.append(data.get('confidence', 1.0))
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        # Create edge type encoding
        unique_edge_types = list(set(edge_types))
        edge_type_to_idx = {etype: idx for idx, etype in enumerate(unique_edge_types)}
        edge_type_indices = torch.tensor([edge_type_to_idx[etype] for etype in edge_types])
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_weights': edge_weights,
            'edge_types': edge_type_indices,
            'node_to_idx': node_to_idx,
            'edge_type_to_idx': edge_type_to_idx
        }
    
    def get_entity_subgraph(self, kg: nx.MultiDiGraph, 
                           entity: BioEntity) -> nx.MultiDiGraph:
        """
        Extract subgraph centered around a specific entity.
        
        Args:
            kg: Full knowledge graph
            entity: Entity to center subgraph around
            
        Returns:
            Subgraph containing entity and its neighbors
        """
        # Find entity node in graph
        entity_node = None
        for node in kg.nodes():
            node_data = kg.nodes[node]
            if node_data.get('name') == entity.text:
                entity_node = node
                break
        
        if entity_node is None:
            return nx.MultiDiGraph()
        
        # Get k-hop neighbors
        k_hop = self.config.get('graph_construction', {}).get('max_hops', 2)
        neighbors = set([entity_node])
        
        for _ in range(k_hop):
            new_neighbors = set()
            for node in neighbors:
                new_neighbors.update(kg.predecessors(node))
                new_neighbors.update(kg.successors(node))
            neighbors.update(new_neighbors)
        
        # Create subgraph
        subgraph = kg.subgraph(neighbors).copy()
        
        return subgraph
    
    def get_pathway_relevance_scores(self, kg: nx.MultiDiGraph, 
                                   hallmarks: List[str]) -> Dict[str, float]:
        """
        Calculate relevance scores for pathways based on hallmarks.
        
        Args:
            kg: Knowledge graph
            hallmarks: List of cancer hallmarks
            
        Returns:
            Dictionary mapping pathway IDs to relevance scores
        """
        pathway_scores = {}
        
        # Get hallmark-associated pathways
        hallmark_pathways = set()
        for hallmark in hallmarks:
            if hallmark in BiologicalKGBuilder.HALLMARK_PATHWAYS:
                hallmark_pathways.update(BiologicalKGBuilder.HALLMARK_PATHWAYS[hallmark])
        
        # Score pathways based on connectivity and hallmark association
        for node in kg.nodes():
            node_data = kg.nodes[node]
            if node_data.get('node_type') == 'pathway':
                pathway_id = node_data.get('properties', {}).get('pathway_id')
                
                if pathway_id:
                    # Base score from hallmark association
                    base_score = 1.0 if pathway_id in hallmark_pathways else 0.5
                    
                    # Connectivity score
                    degree = kg.degree(node)
                    connectivity_score = min(degree / 10.0, 1.0)
                    
                    # Combined score
                    pathway_scores[node] = base_score * 0.7 + connectivity_score * 0.3
        
        return pathway_scores
    
    def _get_cache_key(self, text: str, hallmarks: Optional[List[str]]) -> str:
        """Generate cache key for text and hallmarks"""
        hallmark_str = '-'.join(sorted(hallmarks)) if hallmarks else 'none'
        return f"{hash(text)}_{hallmark_str}"
    
    def visualize_knowledge_graph(self, kg: nx.MultiDiGraph, 
                                 output_path: str,
                                 highlight_entities: Optional[List[str]] = None):
        """
        Visualize the knowledge graph (placeholder for actual implementation).
        
        Args:
            kg: Knowledge graph to visualize
            output_path: Path to save visualization
            highlight_entities: Optional list of entity names to highlight
        """
        # This would be implemented with proper graph visualization
        # Using libraries like plotly or matplotlib
        logger.info(f"Visualization would be saved to {output_path}")
    
    def get_statistics(self, kg: nx.MultiDiGraph) -> Dict[str, any]:
        """
        Get statistics about the knowledge graph.
        
        Args:
            kg: Knowledge graph
            
        Returns:
            Dictionary with graph statistics
        """
        stats = {
            'num_nodes': kg.number_of_nodes(),
            'num_edges': kg.number_of_edges(),
            'density': nx.density(kg),
            'node_types': {},
            'edge_types': {},
            'avg_degree': 0
        }
        
        # Count node types
        for node in kg.nodes():
            node_type = kg.nodes[node].get('node_type', 'unknown')
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        # Count edge types
        for _, _, data in kg.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            stats['edge_types'][edge_type] = stats['edge_types'].get(edge_type, 0) + 1
        
        # Average degree
        if kg.number_of_nodes() > 0:
            stats['avg_degree'] = sum(dict(kg.degree()).values()) / kg.number_of_nodes()
        
        return stats