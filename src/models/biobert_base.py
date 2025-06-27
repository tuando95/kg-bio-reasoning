"""
BioBERT Base Model with Entity-Aware Encoding

This module implements the BioBERT base model enhanced with biological entity
awareness for cancer hallmark classification.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BioBERTEntityAware(nn.Module):
    """
    BioBERT model enhanced with biological entity awareness.
    
    This model:
    1. Encodes text using BioBERT
    2. Incorporates entity information into token representations
    3. Provides entity-pooled representations for downstream tasks
    """
    
    def __init__(self, config: Dict):
        """
        Initialize BioBERT with entity awareness.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Load pre-trained BioBERT
        model_name = config.get('base_model', 'dmis-lab/biobert-base-cased-v1.1')
        self.biobert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.biobert.config.hidden_size
        
        # Entity type embeddings
        self.entity_type_embeddings = nn.Embedding(
            num_embeddings=5,  # GENE, PROTEIN, CHEMICAL, DISEASE, NONE
            embedding_dim=self.hidden_size
        )
        
        # Entity confidence projection
        self.confidence_projection = nn.Linear(1, self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.get('dropout_rate', 0.1))
        
        # Entity pooling strategy
        self.entity_pooling = config.get('entity_pooling', 'mean')  # mean, max, first
        
        logger.info(f"Initialized BioBERT Entity-Aware model with {model_name}")
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                entity_mapping: Optional[Dict[int, List[int]]] = None,
                entity_types: Optional[torch.Tensor] = None,
                entity_confidences: Optional[torch.Tensor] = None,
                return_entity_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BioBERT with entity awareness.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            entity_mapping: Mapping from entity index to token indices
            entity_types: Entity type IDs [batch_size, num_entities]
            entity_confidences: Entity confidence scores [batch_size, num_entities]
            return_entity_embeddings: Whether to return entity-pooled embeddings
            
        Returns:
            Dictionary containing:
                - last_hidden_state: Token embeddings [batch_size, seq_len, hidden_size]
                - pooler_output: [CLS] token embedding [batch_size, hidden_size]
                - entity_embeddings: Entity-pooled embeddings (if requested)
        """
        # Get BioBERT outputs
        outputs = self.biobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]
        pooler_output = outputs.pooler_output  # [batch, hidden]
        
        # Enhance with entity information if provided
        if entity_mapping is not None and entity_types is not None:
            last_hidden_state = self._enhance_with_entities(
                last_hidden_state,
                entity_mapping,
                entity_types,
                entity_confidences
            )
        
        result = {
            'last_hidden_state': last_hidden_state,
            'pooler_output': pooler_output
        }
        
        # Compute entity-pooled embeddings if requested
        if return_entity_embeddings and entity_mapping is not None:
            entity_embeddings = self._pool_entity_embeddings(
                last_hidden_state,
                entity_mapping
            )
            result['entity_embeddings'] = entity_embeddings
        
        return result
    
    def _enhance_with_entities(self,
                              hidden_states: torch.Tensor,
                              entity_mapping: Dict[int, List[int]],
                              entity_types: torch.Tensor,
                              entity_confidences: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Enhance token representations with entity information.
        
        Args:
            hidden_states: Token embeddings [batch, seq_len, hidden]
            entity_mapping: Mapping from entity index to token indices
            entity_types: Entity type IDs [batch, num_entities]
            entity_confidences: Entity confidence scores [batch, num_entities]
            
        Returns:
            Enhanced token embeddings
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get entity type embeddings
        type_embeddings = self.entity_type_embeddings(entity_types)  # [batch, num_entities, hidden]
        
        # Add confidence information if available
        if entity_confidences is not None:
            confidence_emb = self.confidence_projection(
                entity_confidences.unsqueeze(-1)
            )  # [batch, num_entities, hidden]
            entity_emb = type_embeddings + confidence_emb
        else:
            entity_emb = type_embeddings
        
        # Apply dropout
        entity_emb = self.dropout(entity_emb)
        
        # Add entity embeddings to corresponding tokens
        enhanced_states = hidden_states.clone()
        
        for batch_idx in range(batch_size):
            for entity_idx, token_indices in entity_mapping.items():
                if entity_idx < entity_emb.shape[1]:  # Check bounds
                    for token_idx in token_indices:
                        if token_idx < seq_len:  # Check bounds
                            enhanced_states[batch_idx, token_idx] += entity_emb[batch_idx, entity_idx]
        
        return enhanced_states
    
    def _pool_entity_embeddings(self,
                               hidden_states: torch.Tensor,
                               entity_mapping: Dict[int, List[int]]) -> torch.Tensor:
        """
        Pool token embeddings for each entity.
        
        Args:
            hidden_states: Token embeddings [batch, seq_len, hidden]
            entity_mapping: Mapping from entity index to token indices
            
        Returns:
            Entity-pooled embeddings [batch, num_entities, hidden]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_entities = len(entity_mapping)
        
        # Initialize entity embeddings
        entity_embeddings = torch.zeros(
            batch_size, num_entities, hidden_size,
            device=hidden_states.device
        )
        
        # Pool embeddings for each entity
        for entity_idx, token_indices in entity_mapping.items():
            if not token_indices:
                continue
                
            # Get embeddings for entity tokens
            entity_token_embs = []
            for batch_idx in range(batch_size):
                token_embs = []
                for token_idx in token_indices:
                    if token_idx < seq_len:
                        token_embs.append(hidden_states[batch_idx, token_idx])
                
                if token_embs:
                    token_embs = torch.stack(token_embs)
                    
                    # Apply pooling strategy
                    if self.entity_pooling == 'mean':
                        pooled = token_embs.mean(dim=0)
                    elif self.entity_pooling == 'max':
                        pooled = token_embs.max(dim=0)[0]
                    elif self.entity_pooling == 'first':
                        pooled = token_embs[0]
                    else:
                        pooled = token_embs.mean(dim=0)  # Default to mean
                    
                    entity_embeddings[batch_idx, entity_idx] = pooled
        
        return entity_embeddings
    
    def get_tokenizer(self):
        """Get the tokenizer for this model"""
        model_name = self.config.get('base_model', 'dmis-lab/biobert-base-cased-v1.1')
        return AutoTokenizer.from_pretrained(model_name)


class BioBERTClassificationHead(nn.Module):
    """
    Classification head for BioBERT models.
    
    Supports multi-label classification for cancer hallmarks.
    """
    
    def __init__(self, hidden_size: int, num_labels: int, dropout_rate: float = 0.1):
        """
        Initialize classification head.
        
        Args:
            hidden_size: Input hidden size
            num_labels: Number of output labels
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            hidden_states: Input hidden states [batch_size, hidden_size]
            
        Returns:
            Logits [batch_size, num_labels]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        
        return logits