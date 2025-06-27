"""
Graph Neural Network Module for Biological Embeddings

This module implements Graph Attention Networks (GAT) to compute biologically-informed
node embeddings from the knowledge graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BiologicalGAT(nn.Module):
    """
    Graph Attention Network for biological knowledge graphs.
    
    Computes node embeddings that capture biological relationships and
    pathway information.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the GAT model.
        
        Args:
            config: GNN configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Model parameters
        self.num_layers = config.get('num_layers', 3)
        self.hidden_dim = config.get('hidden_dim', 512)
        self.num_heads = config.get('num_heads', 4)
        self.dropout = config.get('dropout', 0.1)
        self.residual = config.get('residual', True)
        
        # Input dimension (will be set dynamically)
        self.input_dim = config.get('input_dim', 768)  # BioBERT dimension
        
        # Input projection layer if dimensions don't match
        if self.input_dim != self.hidden_dim:
            self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        else:
            self.input_projection = None
        
        # Node type embeddings
        self.node_type_embedding = nn.Embedding(
            num_embeddings=6,  # gene, protein, pathway, go_term, hallmark, unknown
            embedding_dim=self.hidden_dim
        )
        
        # Edge type embeddings for multi-relational graphs
        self.edge_type_embedding = nn.Embedding(
            num_embeddings=5,  # interacts, regulates, pathway_member, associated_with, unknown
            embedding_dim=self.hidden_dim
        )
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(
                in_channels=self.input_dim,
                out_channels=self.hidden_dim // self.num_heads,
                heads=self.num_heads,
                dropout=self.dropout,
                concat=True
            )
        )
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.gat_layers.append(
                GATConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim // self.num_heads,
                    heads=self.num_heads,
                    dropout=self.dropout,
                    concat=True
                )
            )
        
        # Last layer
        self.gat_layers.append(
            GATConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                heads=1,
                dropout=self.dropout,
                concat=False
            )
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Pathway-specific attention
        self.pathway_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            dropout=self.dropout
        )
        
        logger.info(f"Initialized Biological GAT with {self.num_layers} layers")
    
    def forward(self, 
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                node_types: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the GAT.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge type indices [num_edges]
            node_types: Node type indices [num_nodes]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            Dictionary containing:
                - node_embeddings: Final node embeddings [num_nodes, hidden_dim]
                - graph_embedding: Graph-level embedding [batch_size, hidden_dim]
                - attention_weights: Attention weights from GAT layers
        """
        # Project input features to hidden dimension if needed
        if self.input_projection is not None:
            x = self.input_projection(x)
        
        # Add node type embeddings
        if node_types is not None:
            type_emb = self.node_type_embedding(node_types)
            x = x + type_emb
        
        # Store attention weights
        attention_weights = []
        
        # Forward through GAT layers
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            # Store input for residual connection
            residual = x if i > 0 else None
            
            # GAT forward pass
            if i < len(self.gat_layers) - 1:
                x, (edge_index_out, alpha) = gat_layer(x, edge_index, return_attention_weights=True)
                attention_weights.append(alpha)
            else:
                x = gat_layer(x, edge_index)
            
            # Apply activation and dropout
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
            
            # Layer normalization
            x = layer_norm(x)
            
            # Residual connection
            if self.residual and residual is not None and x.shape == residual.shape:
                x = x + residual
        
        # Node embeddings
        node_embeddings = x
        
        # Compute graph-level embedding
        if batch is None:
            # Single graph
            graph_embedding = global_mean_pool(node_embeddings, 
                                              torch.zeros(node_embeddings.size(0), 
                                                        dtype=torch.long, 
                                                        device=node_embeddings.device))
        else:
            # Batched graphs
            graph_embedding = global_mean_pool(node_embeddings, batch)
        
        return {
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding,
            'attention_weights': attention_weights
        }
    
    def compute_pathway_attention(self, 
                                node_embeddings: torch.Tensor,
                                pathway_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute attention over pathway nodes.
        
        Args:
            node_embeddings: Node embeddings [num_nodes, hidden_dim]
            pathway_mask: Boolean mask for pathway nodes [num_nodes]
            
        Returns:
            Pathway-attended embeddings [num_pathways, hidden_dim]
        """
        # Extract pathway embeddings
        pathway_embeddings = node_embeddings[pathway_mask]
        
        if pathway_embeddings.size(0) == 0:
            return torch.zeros(1, self.hidden_dim, device=node_embeddings.device)
        
        # Self-attention over pathways
        pathway_embeddings = pathway_embeddings.unsqueeze(0)  # [1, num_pathways, hidden_dim]
        attended, _ = self.pathway_attention(
            pathway_embeddings,
            pathway_embeddings,
            pathway_embeddings
        )
        
        return attended.squeeze(0)


class GraphReadout(nn.Module):
    """
    Graph readout module for obtaining graph-level representations.
    
    Supports multiple readout strategies.
    """
    
    def __init__(self, hidden_dim: int, readout_type: str = 'mean_max'):
        """
        Initialize graph readout.
        
        Args:
            hidden_dim: Dimension of node embeddings
            readout_type: Type of readout ('mean', 'max', 'mean_max', 'attention')
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.readout_type = readout_type
        
        if readout_type == 'attention':
            # Attention-based readout
            self.attention_weights = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softmax(dim=0)
            )
        elif readout_type == 'mean_max':
            # Combine mean and max pooling
            self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, 
                node_embeddings: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute graph-level representation.
        
        Args:
            node_embeddings: Node embeddings [num_nodes, hidden_dim]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            Graph embedding [batch_size, output_dim]
        """
        if self.readout_type == 'mean':
            return global_mean_pool(node_embeddings, batch)
        
        elif self.readout_type == 'max':
            return global_max_pool(node_embeddings, batch)
        
        elif self.readout_type == 'mean_max':
            mean_pool = global_mean_pool(node_embeddings, batch)
            max_pool = global_max_pool(node_embeddings, batch)
            combined = torch.cat([mean_pool, max_pool], dim=-1)
            return self.projection(combined)
        
        elif self.readout_type == 'attention':
            if batch is None:
                # Single graph
                weights = self.attention_weights(node_embeddings)
                return torch.sum(weights * node_embeddings, dim=0, keepdim=True)
            else:
                # Batched graphs
                graph_embeddings = []
                for b in torch.unique(batch):
                    mask = batch == b
                    nodes = node_embeddings[mask]
                    weights = self.attention_weights(nodes)
                    graph_emb = torch.sum(weights * nodes, dim=0)
                    graph_embeddings.append(graph_emb)
                return torch.stack(graph_embeddings)
        
        else:
            raise ValueError(f"Unknown readout type: {self.readout_type}")


class BiologicalGraphEncoder(nn.Module):
    """
    Complete biological graph encoder combining GAT and readout.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the biological graph encoder.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.gat = BiologicalGAT(config['gnn'])
        self.readout = GraphReadout(
            hidden_dim=config['gnn']['hidden_dim'],
            readout_type=config['gnn'].get('readout_type', 'mean_max')
        )
        
        # Projection to match text embedding dimension
        self.output_projection = nn.Linear(
            config['gnn']['hidden_dim'],
            config['model']['hidden_size']
        )
        
    def forward(self, graph_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode biological graph.
        
        Args:
            graph_data: Dictionary containing graph tensors
            
        Returns:
            Dictionary with node and graph embeddings
        """
        # Extract graph components using PyTorch Geometric conventions
        x = graph_data.x if hasattr(graph_data, 'x') else graph_data.get('node_features')
        edge_index = graph_data.edge_index if hasattr(graph_data, 'edge_index') else graph_data.get('edge_index')
        edge_attr = graph_data.edge_attr if hasattr(graph_data, 'edge_attr') else graph_data.get('edge_types')
        node_types = graph_data.node_types if hasattr(graph_data, 'node_types') else graph_data.get('node_types')
        batch = graph_data.batch if hasattr(graph_data, 'batch') else graph_data.get('batch')
        
        # Forward through GAT
        gat_output = self.gat(x, edge_index, edge_attr, node_types, batch)
        
        # Get graph-level representation
        graph_embedding = self.readout(gat_output['node_embeddings'], batch)
        
        # Project to text dimension
        graph_embedding = self.output_projection(graph_embedding)
        
        return {
            'node_embeddings': gat_output['node_embeddings'],
            'graph_embedding': graph_embedding,
            'attention_weights': gat_output['attention_weights']
        }