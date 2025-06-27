"""
Dataset Module for Cancer Hallmarks Classification

This module handles data loading, preprocessing, and batching for the
Hallmarks of Cancer (HoC) dataset with biological knowledge graph integration.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as GraphData, Batch as GraphBatch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict
import asyncio

from ..kg_construction import BiologicalKGPipeline
from .cached_dataset import CachedHallmarksDataset

logger = logging.getLogger(__name__)


class HallmarksOfCancerDataset(Dataset):
    """
    Dataset for cancer hallmarks classification with knowledge graph integration.
    
    Handles:
    - Text tokenization
    - Entity extraction
    - Knowledge graph construction
    - Multi-label encoding
    """
    
    # Hallmark label mapping (exact names from the dataset)
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
                 kg_pipeline: BiologicalKGPipeline,
                 max_length: int = 512,
                 cache_graphs: bool = True):
        """
        Initialize dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            config: Configuration dictionary
            tokenizer: Tokenizer for text encoding
            kg_pipeline: Knowledge graph construction pipeline
            max_length: Maximum sequence length
            cache_graphs: Whether to cache constructed knowledge graphs
        """
        self.split = split
        self.config = config
        self.tokenizer = tokenizer
        self.kg_pipeline = kg_pipeline
        self.max_length = max_length
        self.cache_graphs = cache_graphs
        
        # Load dataset
        logger.info(f"Loading HoC dataset split: {split}")
        self.dataset = load_dataset("qanastek/HoC", split=split)
        
        # Cache for knowledge graphs
        self.kg_cache = {} if cache_graphs else None
        
        logger.info(f"Loaded {len(self.dataset)} samples for {split} split")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with text and knowledge graph.
        
        Returns:
            Dictionary containing:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - labels: Multi-label targets
                - graph_data: Knowledge graph data
                - entity_mapping: Entity to token mapping
                - biological_context: Context for bio attention
        """
        sample = self.dataset[idx]
        
        # Extract text and labels
        text = sample['text']
        labels = self._encode_labels(sample['label'])  # 'label' not 'labels' in HoC dataset
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # Get knowledge graph (from cache or construct)
        if self.cache_graphs and idx in self.kg_cache:
            kg_output = self.kg_cache[idx]
        else:
            # Extract hallmarks for this sample (convert to lowercase with underscores for KG processing)
            hallmarks = [self.HALLMARK_LABELS[i].lower().replace(' ', '_').replace('and_', '') 
                        for i in range(11) if labels[i] == 1 and i != 7]
            
            # Process through KG pipeline
            kg_output = self.kg_pipeline.process_text(text, hallmarks)
            
            if self.cache_graphs:
                self.kg_cache[idx] = kg_output
        
        # Prepare graph data for PyTorch Geometric
        graph_data = self._prepare_graph_data(kg_output.knowledge_graph, kg_output.entities)
        
        # Map entities to tokens
        entity_mapping = self.kg_pipeline.map_entities_to_tokens(
            kg_output.entities,
            {'offset_mapping': encoding['offset_mapping'].squeeze(0).tolist()}
        )
        
        # Prepare entity information
        entity_types, entity_confidences = self._prepare_entity_tensors(kg_output.entities)
        
        # Prepare biological context for attention
        biological_context = self._prepare_biological_context(
            kg_output.knowledge_graph,
            kg_output.entities,
            hallmarks
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
            'idx': idx
        }
    
    def _encode_labels(self, label_list: List[int]) -> torch.Tensor:
        """
        Encode multi-label targets.
        
        Args:
            label_list: List of active label indices
            
        Returns:
            Binary tensor of shape [num_labels]
        """
        labels = torch.zeros(11, dtype=torch.float)
        for label_idx in label_list:
            labels[label_idx] = 1.0
        return labels
    
    def _prepare_graph_data(self, kg, entities) -> GraphData:
        """
        Prepare knowledge graph for PyTorch Geometric.
        
        Args:
            kg: NetworkX knowledge graph
            entities: List of biological entities
            
        Returns:
            PyTorch Geometric Data object
        """
        # Get graph features from pipeline
        graph_features = self.kg_pipeline.prepare_graph_features(kg)
        
        # Create node type tensor
        node_types = []
        node_type_map = {
            'gene': 0, 'protein': 1, 'pathway': 2, 
            'go_term': 3, 'hallmark': 4, 'unknown': 5
        }
        
        for node in kg.nodes():
            node_data = kg.nodes[node]
            node_type = node_data.get('node_type', 'unknown')
            node_types.append(node_type_map.get(node_type, 5))
        
        # Create PyTorch Geometric Data object
        data = GraphData(
            x=graph_features['node_features'],
            edge_index=graph_features['edge_index'],
            edge_attr=graph_features['edge_types'],
            node_types=torch.tensor(node_types, dtype=torch.long)
        )
        
        return data
    
    def _prepare_entity_tensors(self, entities) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare entity type and confidence tensors.
        
        Returns:
            Tuple of (entity_types, entity_confidences)
        """
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
        """
        Prepare biological context for attention mechanism.
        
        Returns:
            Dictionary with entity/pathway embeddings and relevance scores
        """
        # Get pathway relevance scores
        pathway_scores = self.kg_pipeline.get_pathway_relevance_scores(kg, hallmarks)
        
        # Extract entity and pathway embeddings from graph
        entity_embeddings = []
        pathway_embeddings = []
        pathway_relevance = []
        
        node_to_idx = {node: idx for idx, node in enumerate(kg.nodes())}
        
        # Get entity node indices
        for entity in entities:
            # Find corresponding node in graph
            entity_node = None
            for node in kg.nodes():
                if kg.nodes[node].get('name') == entity.text:
                    entity_node = node
                    break
            
            if entity_node and entity_node in node_to_idx:
                # Placeholder embedding (would come from GNN in practice)
                entity_embeddings.append(torch.randn(768))
            else:
                entity_embeddings.append(torch.zeros(768))
        
        # Get pathway nodes
        for node in kg.nodes():
            if kg.nodes[node].get('node_type') == 'pathway':
                # Placeholder embedding
                pathway_embeddings.append(torch.randn(768))
                relevance = pathway_scores.get(node, 0.5)
                pathway_relevance.append(relevance)
        
        # Convert to tensors
        if entity_embeddings:
            entity_embeddings = torch.stack(entity_embeddings)
        else:
            entity_embeddings = torch.zeros(1, 768)
        
        if pathway_embeddings:
            pathway_embeddings = torch.stack(pathway_embeddings)
            pathway_relevance = torch.tensor(pathway_relevance)
        else:
            pathway_embeddings = torch.zeros(1, 768)
            pathway_relevance = torch.zeros(1)
        
        return {
            'entity_embeddings': entity_embeddings,
            'pathway_embeddings': pathway_embeddings,
            'pathway_relevance_scores': pathway_relevance
        }


class HoCDataModule:
    """
    Data module for managing datasets and dataloaders.
    """
    
    def __init__(self, config: Dict, use_cached_kg: bool = True):
        """
        Initialize data module.
        
        Args:
            config: Configuration dictionary
            use_cached_kg: Whether to use pre-cached knowledge graphs
        """
        self.config = config
        self.use_cached_kg = use_cached_kg
        
        # Initialize tokenizer
        model_name = config['model']['base_model']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize KG pipeline (only if not using cache)
        if not use_cached_kg:
            self.kg_pipeline = BiologicalKGPipeline(config['knowledge_graph'])
        
        # Dataset parameters
        self.batch_size = config['training']['batch_size']
        self.max_length = config['dataset']['max_seq_length']
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup datasets for all splits."""
        logger.info(f"Setting up datasets (use_cached_kg={self.use_cached_kg})...")
        
        if self.use_cached_kg:
            # Use cached datasets
            DatasetClass = CachedHallmarksDataset
            dataset_kwargs = {
                'config': self.config,
                'tokenizer': self.tokenizer,
                'max_length': self.max_length
            }
        else:
            # Use on-the-fly KG construction
            DatasetClass = HallmarksOfCancerDataset
            dataset_kwargs = {
                'config': self.config,
                'tokenizer': self.tokenizer,
                'kg_pipeline': self.kg_pipeline,
                'max_length': self.max_length
            }
        
        self.train_dataset = DatasetClass(split='train', **dataset_kwargs)
        self.val_dataset = DatasetClass(split='validation', **dataset_kwargs)
        self.test_dataset = DatasetClass(split='test', **dataset_kwargs)
        
        logger.info(f"Dataset sizes - Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to handle graphs and variable-length data.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Batched data dictionary
        """
        # Stack regular tensors
        input_ids = torch.stack([sample['input_ids'] for sample in batch])
        attention_mask = torch.stack([sample['attention_mask'] for sample in batch])
        labels = torch.stack([sample['labels'] for sample in batch])
        
        # Batch graphs
        graph_list = [sample['graph_data'] for sample in batch]
        batched_graph = GraphBatch.from_data_list(graph_list)
        
        # Prepare entity mappings (per-batch)
        entity_mappings = {}
        for batch_idx, sample in enumerate(batch):
            entity_mappings[batch_idx] = sample['entity_mapping']
        
        # Stack entity tensors (pad to max entities in batch)
        entity_types_list = [sample['entity_types'] for sample in batch]
        entity_confidences_list = [sample['entity_confidences'] for sample in batch]
        
        max_entities = max(len(et) for et in entity_types_list)
        
        entity_types = torch.zeros(len(batch), max_entities, dtype=torch.long)
        entity_confidences = torch.zeros(len(batch), max_entities, dtype=torch.float)
        
        for i, (et, ec) in enumerate(zip(entity_types_list, entity_confidences_list)):
            entity_types[i, :len(et)] = et
            entity_confidences[i, :len(ec)] = ec
        
        # Batch biological context
        biological_context = {
            'entity_embeddings': torch.stack([
                sample['biological_context']['entity_embeddings'] 
                for sample in batch
            ]),
            'pathway_embeddings': torch.stack([
                sample['biological_context']['pathway_embeddings']
                for sample in batch
            ]),
            'pathway_relevance_scores': torch.stack([
                sample['biological_context']['pathway_relevance_scores']
                for sample in batch
            ]),
            'entity_to_token_map': entity_mappings
        }
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'graph_data': batched_graph,
            'entity_mapping': entity_mappings,
            'entity_types': entity_types,
            'entity_confidences': entity_confidences,
            'biological_context': biological_context,
            'indices': torch.tensor([sample['idx'] for sample in batch])
        }