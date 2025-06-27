"""
Cached Dataset Module for Cancer Hallmarks Classification

This module loads pre-processed knowledge graphs from cache instead of
building them on-the-fly, significantly speeding up training.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import pickle
import json
import networkx as nx
from torch_geometric.data import Data as GraphData

from ..kg_construction import BioEntity

logger = logging.getLogger(__name__)


class CachedHallmarksDataset(Dataset):
    """
    Dataset that loads pre-cached knowledge graphs.
    
    This is much faster than building KGs on-the-fly during training.
    """
    
    HALLMARK_LABELS = {
        0: "evading_growth_suppressors",
        1: "tumor_promoting_inflammation",
        2: "enabling_replicative_immortality",
        3: "cellular_energetics",
        4: "resisting_cell_death",
        5: "activating_invasion_metastasis",
        6: "genomic_instability",
        7: "none",
        8: "inducing_angiogenesis",
        9: "sustaining_proliferative_signaling",
        10: "avoiding_immune_destruction"
    }
    
    def __init__(self,
                 split: str,
                 config: Dict,
                 tokenizer: AutoTokenizer,
                 cache_dir: str = "cache/kg_preprocessed",
                 max_length: int = 512):
        """
        Initialize cached dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            config: Configuration dictionary
            tokenizer: Tokenizer for text encoding
            cache_dir: Directory containing cached KGs
            max_length: Maximum sequence length
        """
        self.split = split
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load cache index
        self.cache_dir = Path(cache_dir) / split
        cache_index_path = self.cache_dir / "index.json"
        
        if not cache_index_path.exists():
            raise FileNotFoundError(
                f"Cache index not found at {cache_index_path}. "
                "Please run preprocess_kg_data.py first."
            )
        
        with open(cache_index_path, 'r') as f:
            self.cache_index = json.load(f)
        
        # Convert to list for indexing
        self.sample_ids = sorted([int(idx) for idx in self.cache_index.keys()])
        
        logger.info(f"Loaded cached dataset for {split} split with {len(self.sample_ids)} samples")
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample with cached knowledge graph.
        
        Returns:
            Dictionary containing all necessary tensors
        """
        # Get actual sample ID
        sample_id = self.sample_ids[idx]
        
        # Load cached data
        cache_info = self.cache_index[str(sample_id)]
        cache_file = self.cache_dir / cache_info['file']
        
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Extract components
        text = cached_data['text']
        labels = self._encode_labels(cached_data['labels'])
        
        # Reconstruct entities
        entities = self._deserialize_entities(cached_data['entities'])
        
        # Reconstruct knowledge graph
        kg = self._deserialize_graph(cached_data['knowledge_graph'])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # Prepare graph data for PyTorch Geometric
        graph_data = self._prepare_graph_data(kg)
        
        # Map entities to tokens
        entity_mapping = self._map_entities_to_tokens(
            entities,
            encoding['offset_mapping'].squeeze(0).tolist()
        )
        
        # Prepare entity information
        entity_types, entity_confidences = self._prepare_entity_tensors(entities)
        
        # Prepare biological context
        biological_context = self._prepare_biological_context(
            kg, entities, cached_data['hallmarks']
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
            'graph_data': graph_data,
            'entity_mapping': entity_mapping,
            'entity_types': entity_types,
            'entity_confidences': entity_confidences,
            'biological_context': biological_context,
            'idx': sample_id
        }
    
    def _encode_labels(self, label_list: List[int]) -> torch.Tensor:
        """Encode multi-label targets."""
        labels = torch.zeros(11, dtype=torch.float)
        for label_idx in label_list:
            labels[label_idx] = 1.0
        return labels
    
    def _deserialize_entities(self, entity_data: List[Dict]) -> List[BioEntity]:
        """Reconstruct BioEntity objects from cached data."""
        entities = []
        for e_data in entity_data:
            entity = BioEntity(
                text=e_data['text'],
                start=e_data['start'],
                end=e_data['end'],
                entity_type=e_data['entity_type'],
                normalized_ids=e_data['normalized_ids'],
                confidence=e_data['confidence'],
                context=e_data.get('context')
            )
            entities.append(entity)
        return entities
    
    def _deserialize_graph(self, graph_data: Dict) -> nx.MultiDiGraph:
        """Reconstruct NetworkX graph from cached data."""
        kg = nx.MultiDiGraph()
        
        # Add nodes
        for node_id, node_attrs in graph_data['nodes']:
            kg.add_node(node_id, **node_attrs)
        
        # Add edges
        for source, target, edge_attrs in graph_data['edges']:
            kg.add_edge(source, target, **edge_attrs)
        
        # Add graph attributes
        kg.graph.update(graph_data.get('graph', {}))
        
        return kg
    
    def _prepare_graph_data(self, kg) -> GraphData:
        """Prepare knowledge graph for PyTorch Geometric."""
        # Create node features (placeholder - should match pipeline output)
        num_nodes = kg.number_of_nodes()
        node_features = torch.randn(num_nodes, 768)  # BioBERT dimension
        
        # Create edge index
        edge_list = []
        edge_types = []
        
        node_to_idx = {node: idx for idx, node in enumerate(kg.nodes())}
        
        for source, target, data in kg.edges(data=True):
            source_idx = node_to_idx[source]
            target_idx = node_to_idx[target]
            edge_list.append([source_idx, target_idx])
            edge_types.append(self._edge_type_to_id(data.get('edge_type', 'unknown')))
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros(0, dtype=torch.long)
        
        # Create node types
        node_types = []
        for node in kg.nodes():
            node_data = kg.nodes[node]
            node_types.append(self._node_type_to_id(node_data.get('node_type', 'unknown')))
        
        # Create PyTorch Geometric Data object
        data = GraphData(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_types=torch.tensor(node_types, dtype=torch.long)
        )
        
        return data
    
    def _node_type_to_id(self, node_type: str) -> int:
        """Convert node type to ID."""
        node_type_map = {
            'gene': 0, 'protein': 1, 'pathway': 2,
            'go_term': 3, 'hallmark': 4, 'unknown': 5
        }
        return node_type_map.get(node_type, 5)
    
    def _edge_type_to_id(self, edge_type: str) -> int:
        """Convert edge type to ID."""
        edge_type_map = {
            'interacts': 0, 'regulates': 1, 'pathway_member': 2,
            'associated_with': 3, 'unknown': 4
        }
        return edge_type_map.get(edge_type, 4)
    
    def _map_entities_to_tokens(self, entities, offset_mapping) -> Dict[int, List[int]]:
        """Map entities to token positions."""
        entity_to_tokens = {}
        
        for idx, entity in enumerate(entities):
            token_indices = []
            
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
    
    def _prepare_entity_tensors(self, entities) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare entity type and confidence tensors."""
        entity_type_map = {
            'GENE': 0, 'PROTEIN': 1, 'CHEMICAL': 2, 'DISEASE': 3, 'UNKNOWN': 4
        }
        
        if not entities:
            return torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.float)
        
        entity_types = []
        entity_confidences = []
        
        for entity in entities:
            entity_types.append(entity_type_map.get(entity.entity_type, 4))
            entity_confidences.append(entity.confidence)
        
        return (torch.tensor(entity_types, dtype=torch.long),
                torch.tensor(entity_confidences, dtype=torch.float))
    
    def _prepare_biological_context(self, kg, entities, hallmarks) -> Dict[str, torch.Tensor]:
        """Prepare biological context for attention mechanism."""
        # Simplified version - in practice would compute actual embeddings
        num_entities = len(entities) if entities else 1
        num_pathways = sum(1 for n in kg.nodes() if kg.nodes[n].get('node_type') == 'pathway')
        
        if num_pathways == 0:
            num_pathways = 1
        
        return {
            'entity_embeddings': torch.randn(num_entities, 768),
            'pathway_embeddings': torch.randn(num_pathways, 768),
            'pathway_relevance_scores': torch.rand(num_pathways)
        }