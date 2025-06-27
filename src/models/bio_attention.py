"""
Biological Pathway-Guided Attention Mechanism

This module implements the core innovation: attention weights informed by
biological pathway relevance for cancer hallmark classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class BiologicalPathwayGuidedAttention(nn.Module):
    """
    Attention mechanism that integrates biological pathway information.
    
    Replaces standard self-attention with biologically-informed attention
    that considers pathway relationships between entities.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize biological attention.
        
        Args:
            config: Model configuration with bio_attention settings
        """
        super().__init__()
        self.config = config
        
        # Dimensions
        self.hidden_size = config['model']['hidden_size']
        self.num_heads = config['model']['bio_attention']['num_heads']
        self.head_dim = self.hidden_size // self.num_heads
        self.pathway_relevance_dim = config['model']['bio_attention']['pathway_relevance_dim']
        
        # Fusion parameters
        self.fusion_strategy = config['model']['bio_attention']['fusion_strategy']
        if self.fusion_strategy == 'learned':
            # Learnable fusion weights per head
            self.fusion_weights = nn.Parameter(
                torch.ones(self.num_heads) * config['model']['bio_attention']['fusion_weight']
            )
        else:
            self.fusion_weight = config['model']['bio_attention']['fusion_weight']
        
        # Standard attention components
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Biological attention components
        self.bio_q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.bio_k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.bio_v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Pathway relevance matrix
        self.pathway_relevance_matrix = nn.Linear(
            self.pathway_relevance_dim * 2,
            1
        )
        
        # Entity-pathway association encoder
        self.entity_pathway_encoder = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.pathway_relevance_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout_rate'])
        )
        
        # Output projection
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config['model']['dropout_rate'])
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        logger.info(f"Initialized Biological Pathway-Guided Attention with {self.num_heads} heads")
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                biological_context: Optional[Dict[str, torch.Tensor]] = None,
                return_attention_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through biological attention.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            biological_context: Dictionary containing:
                - entity_embeddings: Entity representations [batch_size, num_entities, hidden_size]
                - pathway_embeddings: Pathway representations [batch_size, num_pathways, hidden_size]
                - entity_to_token_map: Mapping entities to token positions
                - pathway_relevance_scores: Pre-computed pathway relevance [batch_size, num_pathways]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of:
                - Output hidden states [batch_size, seq_len, hidden_size]
                - Attention weights (if requested) [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute standard attention
        text_attention_out, text_attention_weights = self._compute_text_attention(
            hidden_states, attention_mask
        )
        
        # Compute biological attention if context is provided
        if biological_context is not None:
            bio_attention_out, bio_attention_weights = self._compute_biological_attention(
                hidden_states, attention_mask, biological_context
            )
            
            # Fuse text and biological attention
            output = self._fuse_attention_outputs(
                text_attention_out, bio_attention_out, hidden_states
            )
            
            # Combine attention weights for visualization
            if return_attention_weights:
                attention_weights = self._fuse_attention_weights(
                    text_attention_weights, bio_attention_weights
                )
            else:
                attention_weights = None
        else:
            # Fall back to standard attention
            output = text_attention_out
            attention_weights = text_attention_weights if return_attention_weights else None
        
        # Apply layer normalization (residual connection)
        output = self.layer_norm(output + hidden_states)
        
        return output, attention_weights
    
    def _compute_text_attention(self,
                               hidden_states: torch.Tensor,
                               attention_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute standard multi-head self-attention.
        
        Returns:
            Tuple of output and attention weights
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Use smaller value for half precision compatibility
            scores = scores.masked_fill(mask == 0, -1e4)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        # Output projection
        output = self.out_proj(context)
        
        return output, attention_weights
    
    def _compute_biological_attention(self,
                                    hidden_states: torch.Tensor,
                                    attention_mask: Optional[torch.Tensor],
                                    biological_context: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute biologically-informed attention using pathway relevance.
        
        Returns:
            Tuple of output and attention weights
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Get biological embeddings
        entity_embeddings = biological_context.get('entity_embeddings')
        pathway_embeddings = biological_context.get('pathway_embeddings')
        entity_to_token_map = biological_context.get('entity_to_token_map')
        pathway_relevance_scores = biological_context.get('pathway_relevance_scores')
        
        # Enhance hidden states with biological information
        bio_enhanced_states = self._enhance_with_biological_context(
            hidden_states, entity_embeddings, entity_to_token_map
        )
        
        # Linear projections for biological attention
        Q_bio = self.bio_q_proj(bio_enhanced_states)
        K_bio = self.bio_k_proj(bio_enhanced_states)
        V_bio = self.bio_v_proj(bio_enhanced_states)
        
        # Reshape for multi-head attention
        Q_bio = Q_bio.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K_bio = K_bio.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V_bio = V_bio.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute biological attention scores
        scores_bio = torch.matmul(Q_bio, K_bio.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add pathway relevance bias
        if pathway_embeddings is not None and pathway_relevance_scores is not None:
            pathway_bias = self._compute_pathway_relevance_bias(
                hidden_states, pathway_embeddings, pathway_relevance_scores, entity_to_token_map
            )
            scores_bio = scores_bio + pathway_bias.unsqueeze(1)  # Add head dimension
        
        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores_bio = scores_bio.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights_bio = F.softmax(scores_bio, dim=-1)
        attention_weights_bio = self.dropout(attention_weights_bio)
        
        # Apply attention to values
        context_bio = torch.matmul(attention_weights_bio, V_bio)
        
        # Reshape back
        context_bio = context_bio.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        # Output projection
        output_bio = self.out_proj(context_bio)
        
        return output_bio, attention_weights_bio
    
    def _enhance_with_biological_context(self,
                                       hidden_states: torch.Tensor,
                                       entity_embeddings: Optional[torch.Tensor],
                                       entity_to_token_map: Optional[Dict]) -> torch.Tensor:
        """
        Enhance token representations with biological entity information.
        """
        if entity_embeddings is None or entity_to_token_map is None:
            return hidden_states
        
        enhanced_states = hidden_states.clone()
        batch_size = hidden_states.shape[0]
        
        # Add entity embeddings to corresponding tokens
        for batch_idx in range(batch_size):
            if batch_idx in entity_to_token_map:
                for entity_idx, token_indices in entity_to_token_map[batch_idx].items():
                    if entity_idx < entity_embeddings.shape[1]:
                        entity_emb = entity_embeddings[batch_idx, entity_idx]
                        for token_idx in token_indices:
                            if token_idx < enhanced_states.shape[1]:
                                enhanced_states[batch_idx, token_idx] = \
                                    enhanced_states[batch_idx, token_idx] + 0.1 * entity_emb
        
        return enhanced_states
    
    def _compute_pathway_relevance_bias(self,
                                      hidden_states: torch.Tensor,
                                      pathway_embeddings: torch.Tensor,
                                      pathway_relevance_scores: torch.Tensor,
                                      entity_to_token_map: Optional[Dict]) -> torch.Tensor:
        """
        Compute pathway-based attention bias matrix.
        
        Returns:
            Attention bias matrix [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Initialize bias matrix
        bias_matrix = torch.zeros(batch_size, seq_len, seq_len, device=hidden_states.device)
        
        # Compute token-pathway associations
        token_pathway_scores = self._compute_token_pathway_associations(
            hidden_states, pathway_embeddings, entity_to_token_map
        )  # [batch_size, seq_len, num_pathways]
        
        # Weight by pathway relevance
        weighted_scores = token_pathway_scores * pathway_relevance_scores.unsqueeze(1)
        
        # Vectorized computation of pairwise similarity (much faster than nested loops)
        # Normalize weighted scores for cosine similarity
        norm_scores = F.normalize(weighted_scores, p=2, dim=-1)  # [batch_size, seq_len, num_pathways]
        
        # Compute pairwise cosine similarity matrix
        # bmm: batch matrix multiplication
        bias_matrix = torch.bmm(norm_scores, norm_scores.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        
        # Scale down the bias
        bias_matrix = bias_matrix * 0.1
        
        return bias_matrix
    
    def _compute_token_pathway_associations(self,
                                          hidden_states: torch.Tensor,
                                          pathway_embeddings: torch.Tensor,
                                          entity_to_token_map: Optional[Dict]) -> torch.Tensor:
        """
        Compute association scores between tokens and pathways.
        
        Returns:
            Association scores [batch_size, seq_len, num_pathways]
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_pathways = pathway_embeddings.shape[1]
        
        # Compute similarity between tokens and pathways
        # Expand dimensions for broadcasting
        tokens_expanded = hidden_states.unsqueeze(2)  # [batch, seq_len, 1, hidden]
        pathways_expanded = pathway_embeddings.unsqueeze(1)  # [batch, 1, num_pathways, hidden]
        
        # Concatenate and encode
        combined = torch.cat([
            tokens_expanded.expand(-1, -1, num_pathways, -1),
            pathways_expanded.expand(-1, seq_len, -1, -1)
        ], dim=-1)  # [batch, seq_len, num_pathways, hidden*2]
        
        # Compute association scores
        associations = self.entity_pathway_encoder(combined)  # [batch, seq_len, num_pathways, pathway_dim]
        associations = associations.mean(dim=-1)  # Simple aggregation
        
        return F.softmax(associations, dim=-1)
    
    def _fuse_attention_outputs(self,
                              text_output: torch.Tensor,
                              bio_output: torch.Tensor,
                              hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Fuse text and biological attention outputs based on fusion strategy.
        """
        if self.fusion_strategy == 'learned':
            # Use learned fusion weights
            fusion_weights = torch.sigmoid(self.fusion_weights).mean()
            output = fusion_weights * text_output + (1 - fusion_weights) * bio_output
        
        elif self.fusion_strategy == 'adaptive':
            # Compute adaptive fusion based on input
            fusion_gate = torch.sigmoid(
                self.out_proj(torch.cat([text_output, bio_output, hidden_states], dim=-1))
            )
            output = fusion_gate * text_output + (1 - fusion_gate) * bio_output
        
        else:  # fixed
            output = self.fusion_weight * text_output + (1 - self.fusion_weight) * bio_output
        
        return output
    
    def _fuse_attention_weights(self,
                              text_weights: torch.Tensor,
                              bio_weights: torch.Tensor) -> torch.Tensor:
        """
        Fuse attention weight matrices for visualization.
        """
        if self.fusion_strategy == 'learned':
            fusion_weight = torch.sigmoid(self.fusion_weights).view(1, -1, 1, 1)
            return fusion_weight * text_weights + (1 - fusion_weight) * bio_weights
        else:
            return self.fusion_weight * text_weights + (1 - self.fusion_weight) * bio_weights


class BiologicalTransformerLayer(nn.Module):
    """
    Transformer layer with biological pathway-guided attention.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Biological attention
        self.attention = BiologicalPathwayGuidedAttention(config)
        
        # Feed-forward network
        self.intermediate = nn.Linear(config['model']['hidden_size'], 
                                    config['model']['hidden_size'] * 4)
        self.output = nn.Linear(config['model']['hidden_size'] * 4,
                              config['model']['hidden_size'])
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config['model']['dropout_rate'])
        self.layer_norm = nn.LayerNorm(config['model']['hidden_size'])
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                biological_context: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass through transformer layer.
        """
        # Attention sub-layer
        attention_output, _ = self.attention(
            hidden_states, attention_mask, biological_context
        )
        
        # Feed-forward sub-layer
        intermediate_output = self.activation(self.intermediate(attention_output))
        intermediate_output = self.dropout(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        
        # Residual connection and layer norm
        layer_output = self.layer_norm(layer_output + attention_output)
        
        return layer_output