"""
BioKG-BioBERT: Complete Model Architecture

This module implements the full BioKG-BioBERT model that integrates:
- BioBERT with entity awareness
- Biological knowledge graph embeddings via GNN
- Biological pathway-guided attention
- Multi-modal fusion for cancer hallmark classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

from .biobert_base import BioBERTEntityAware, BioBERTClassificationHead
from .gnn_module import BiologicalGraphEncoder
from .bio_attention import BiologicalTransformerLayer

logger = logging.getLogger(__name__)


class BioKGBioBERT(nn.Module):
    """
    Complete BioKG-BioBERT model for cancer hallmark classification.
    
    Architecture:
    1. Text encoding with BioBERT + entity awareness
    2. Knowledge graph encoding with GAT
    3. Biological pathway-guided attention
    4. Multi-modal fusion
    5. Multi-task classification heads
    """
    
    def __init__(self, config: Dict):
        """
        Initialize BioKG-BioBERT model.
        
        Args:
            config: Complete configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Model components based on configuration
        self.use_kg = config.get('use_knowledge_graph', True)
        self.use_bio_attention = config.get('use_bio_attention', True)
        
        # Text encoder with entity awareness
        self.text_encoder = BioBERTEntityAware(config)
        
        # Knowledge graph encoder (if enabled)
        if self.use_kg:
            gnn_config = config.get('gnn', {
                'type': 'GAT',
                'num_layers': 3,
                'hidden_dim': 512,
                'num_heads': 4,
                'dropout': 0.1,
                'residual': True
            })
            config['gnn'] = gnn_config
            self.graph_encoder = BiologicalGraphEncoder(config)
        
        # Biological attention layers (if enabled)
        if self.use_bio_attention:
            # Replace top BioBERT layers with biological attention
            num_bio_layers = config.get('num_bio_attention_layers', 2)
            self.bio_attention_layers = nn.ModuleList([
                BiologicalTransformerLayer(config) for _ in range(num_bio_layers)
            ])
        
        # Multi-modal fusion
        fusion_config = config.get('fusion', {
            'strategy': 'late',
            'text_dim': 768,
            'graph_dim': 512,
            'fusion_dim': 1024
        })
        self.fusion_strategy = fusion_config.get('strategy', 'late')
        config['fusion'] = fusion_config
        self.fusion_module = self._build_fusion_module(config)
        
        # Classification heads
        self.num_labels = config.get('num_labels', 11)
        dropout_rate = config.get('dropout_rate', 0.1)
        fusion_dim = config.get('fusion', {}).get('fusion_dim', 1024)
        
        # Primary task: Cancer hallmark classification
        self.hallmark_classifier = BioBERTClassificationHead(
            hidden_size=fusion_dim,
            num_labels=self.num_labels,
            dropout_rate=dropout_rate
        )
        
        # Auxiliary task 1: Pathway activation prediction (if enabled)
        loss_weights = config.get('loss_weights', {})
        if loss_weights.get('pathway_loss', 0) > 0:
            self.pathway_classifier = nn.Linear(
                fusion_dim,
                config.get('num_pathways', 50)  # Top pathways
            )
        else:
            self.pathway_classifier = None
        
        # Auxiliary task 2: Entity consistency validation (if enabled)
        if loss_weights.get('consistency_loss', 0) > 0:
            self.consistency_predictor = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 1)  # Binary consistency score
            )
        else:
            self.consistency_predictor = None
        
        logger.info(f"Initialized BioKG-BioBERT model")
        logger.info(f"  - KG enabled: {self.use_kg}")
        logger.info(f"  - Bio attention enabled: {self.use_bio_attention}")
        logger.info(f"  - Fusion strategy: {self.fusion_strategy}")
    
    def _build_fusion_module(self, config: Dict) -> nn.Module:
        """Build fusion module based on strategy."""
        fusion_config = config.get('fusion', {
            'strategy': 'late',
            'text_dim': 768,
            'graph_dim': 512,
            'fusion_dim': 1024
        })
        
        if fusion_config['strategy'] == 'early':
            return EarlyFusion(
                text_dim=fusion_config['text_dim'],
                graph_dim=fusion_config['graph_dim'],
                fusion_dim=fusion_config['fusion_dim'],
                dropout_rate=config.get('dropout_rate', 0.1)
            )
        elif fusion_config['strategy'] == 'late':
            return LateFusion(
                text_dim=fusion_config['text_dim'],
                graph_dim=fusion_config['graph_dim'],
                fusion_dim=fusion_config['fusion_dim'],
                dropout_rate=config.get('dropout_rate', 0.1)
            )
        elif fusion_config['strategy'] == 'cross_modal':
            return CrossModalFusion(
                text_dim=fusion_config['text_dim'],
                graph_dim=fusion_config['graph_dim'],
                fusion_dim=fusion_config['fusion_dim'],
                dropout_rate=config.get('dropout_rate', 0.1)
            )
        else:
            # No fusion - text only
            return nn.Linear(fusion_config['text_dim'], fusion_config['fusion_dim'])
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                graph_data: Optional[Dict[str, torch.Tensor]] = None,
                entity_mapping: Optional[Dict[int, List[int]]] = None,
                entity_types: Optional[torch.Tensor] = None,
                entity_confidences: Optional[torch.Tensor] = None,
                biological_context: Optional[Dict[str, torch.Tensor]] = None,
                labels: Optional[torch.Tensor] = None,
                pathway_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BioKG-BioBERT.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            graph_data: Knowledge graph data (if KG enabled)
            entity_mapping: Entity to token mapping
            entity_types: Entity type IDs
            entity_confidences: Entity confidence scores
            biological_context: Context for biological attention
            labels: Cancer hallmark labels [batch_size, num_labels]
            pathway_labels: Pathway activation labels [batch_size, num_pathways]
            
        Returns:
            Dictionary containing:
                - logits: Hallmark classification logits
                - loss: Total loss (if labels provided)
                - pathway_logits: Pathway predictions (if enabled)
                - text_embedding: Text representation
                - graph_embedding: Graph representation (if KG enabled)
        """
        # Step 1: Encode text with entity awareness
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_mapping=entity_mapping,
            entity_types=entity_types,
            entity_confidences=entity_confidences,
            return_entity_embeddings=True
        )
        
        hidden_states = text_output['last_hidden_state']
        pooler_output = text_output['pooler_output']
        
        # Step 2: Apply biological attention layers (if enabled)
        if self.use_bio_attention and biological_context is not None:
            for bio_layer in self.bio_attention_layers:
                hidden_states = bio_layer(
                    hidden_states,
                    attention_mask,
                    biological_context
                )
            
            # Re-pool after biological attention
            text_embedding = hidden_states.mean(dim=1)
        else:
            text_embedding = pooler_output
        
        # Step 3: Encode knowledge graph (if enabled)
        if self.use_kg and graph_data is not None:
            graph_output = self.graph_encoder(graph_data)
            graph_embedding = graph_output['graph_embedding']
        else:
            graph_embedding = None
        
        # Step 4: Multi-modal fusion
        if graph_embedding is not None:
            fused_representation = self.fusion_module(text_embedding, graph_embedding)
        else:
            # Text only
            fused_representation = self.fusion_module(text_embedding)
        
        # Step 5: Classification
        hallmark_logits = self.hallmark_classifier(fused_representation)
        
        # Prepare output
        output = {
            'logits': hallmark_logits,
            'text_embedding': text_embedding,
            'fused_representation': fused_representation
        }
        
        if graph_embedding is not None:
            output['graph_embedding'] = graph_embedding
        
        # Auxiliary predictions
        if hasattr(self, 'pathway_classifier'):
            pathway_logits = self.pathway_classifier(fused_representation)
            output['pathway_logits'] = pathway_logits
        
        if hasattr(self, 'consistency_predictor'):
            consistency_score = self.consistency_predictor(fused_representation)
            output['consistency_score'] = consistency_score
        
        # Compute loss if labels provided
        if labels is not None:
            loss = self._compute_loss(output, labels, pathway_labels)
            output['loss'] = loss
        
        return output
    
    def _compute_loss(self,
                     output: Dict[str, torch.Tensor],
                     labels: torch.Tensor,
                     pathway_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute multi-task loss.
        
        Args:
            output: Model outputs
            labels: Hallmark labels
            pathway_labels: Pathway activation labels
            
        Returns:
            Total loss
        """
        loss_weights = self.config['training']['loss_weights']
        
        # Hallmark classification loss (multi-label)
        hallmark_loss = F.binary_cross_entropy_with_logits(
            output['logits'], labels.float()
        )
        
        total_loss = loss_weights['hallmark_loss'] * hallmark_loss
        
        # Pathway loss (if enabled)
        if 'pathway_logits' in output and pathway_labels is not None:
            pathway_loss = F.binary_cross_entropy_with_logits(
                output['pathway_logits'], pathway_labels.float()
            )
            total_loss += loss_weights['pathway_loss'] * pathway_loss
        
        # Consistency loss (if enabled)
        if 'consistency_score' in output:
            # Compute biological consistency constraints
            consistency_loss = self._compute_consistency_loss(
                output['logits'], labels
            )
            total_loss += loss_weights['consistency_loss'] * consistency_loss
        
        return total_loss
    
    def _compute_consistency_loss(self,
                                logits: torch.Tensor,
                                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute biological consistency loss.
        
        Penalizes predictions that violate biological constraints.
        """
        # Define incompatible hallmark pairs (simplified)
        incompatible_pairs = [
            (0, 9),  # Evading growth suppressors vs Sustaining proliferative signaling
            (4, 8),  # Resisting cell death vs Inducing angiogenesis (context-dependent)
        ]
        
        predictions = torch.sigmoid(logits)
        consistency_loss = 0.0
        
        for i, j in incompatible_pairs:
            # Penalize when both incompatible hallmarks are predicted
            violation = torch.relu(predictions[:, i] + predictions[:, j] - 1.5)
            consistency_loss += violation.mean()
        
        return consistency_loss


class EarlyFusion(nn.Module):
    """Early fusion strategy - combine embeddings before processing."""
    
    def __init__(self, text_dim: int, graph_dim: int, fusion_dim: int, dropout_rate: float):
        super().__init__()
        self.projection = nn.Linear(text_dim + graph_dim, fusion_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(fusion_dim)
    
    def forward(self, text_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        concatenated = torch.cat([text_emb, graph_emb], dim=-1)
        fused = self.projection(concatenated)
        fused = self.activation(fused)
        fused = self.dropout(fused)
        fused = self.layer_norm(fused)
        return fused


class LateFusion(nn.Module):
    """Late fusion strategy - process modalities separately then combine."""
    
    def __init__(self, text_dim: int, graph_dim: int, fusion_dim: int, dropout_rate: float):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.graph_projection = nn.Linear(graph_dim, fusion_dim)
        self.fusion_layer = nn.Linear(fusion_dim * 2, fusion_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(fusion_dim)
    
    def forward(self, text_emb: torch.Tensor, graph_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        text_proj = self.text_projection(text_emb)
        text_proj = self.activation(text_proj)
        
        if graph_emb is not None:
            graph_proj = self.graph_projection(graph_emb)
            graph_proj = self.activation(graph_proj)
            
            combined = torch.cat([text_proj, graph_proj], dim=-1)
            fused = self.fusion_layer(combined)
        else:
            fused = text_proj
        
        fused = self.dropout(fused)
        fused = self.layer_norm(fused)
        return fused


class CrossModalFusion(nn.Module):
    """Cross-modal attention fusion - attend between modalities."""
    
    def __init__(self, text_dim: int, graph_dim: int, fusion_dim: int, dropout_rate: float):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.graph_projection = nn.Linear(graph_dim, fusion_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=dropout_rate
        )
        
        self.fusion_layer = nn.Linear(fusion_dim * 2, fusion_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(fusion_dim)
    
    def forward(self, text_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        # Project to same dimension
        text_proj = self.text_projection(text_emb)
        graph_proj = self.graph_projection(graph_emb)
        
        # Add sequence dimension for attention
        text_seq = text_proj.unsqueeze(0)  # [1, batch, fusion_dim]
        graph_seq = graph_proj.unsqueeze(0)  # [1, batch, fusion_dim]
        
        # Cross-attention: text attends to graph
        text_attended, _ = self.cross_attention(
            query=text_seq,
            key=graph_seq,
            value=graph_seq
        )
        
        # Cross-attention: graph attends to text
        graph_attended, _ = self.cross_attention(
            query=graph_seq,
            key=text_seq,
            value=text_seq
        )
        
        # Remove sequence dimension
        text_attended = text_attended.squeeze(0)
        graph_attended = graph_attended.squeeze(0)
        
        # Combine attended representations
        combined = torch.cat([text_attended, graph_attended], dim=-1)
        fused = self.fusion_layer(combined)
        fused = self.activation(fused)
        fused = self.dropout(fused)
        fused = self.layer_norm(fused)
        
        return fused