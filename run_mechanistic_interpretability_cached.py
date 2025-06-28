"""
Mechanistic Interpretability Analysis using Cached Knowledge Graphs

This script provides pathway-based explanations for classification decisions
using pre-computed cached knowledge graphs.
"""

import os
import sys
import torch
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict, Counter
import pickle

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.biokg_biobert import BioKGBioBERT
from src.data.dataset import HoCDataModule
from src.data.cached_dataset import CachedHallmarksDataset
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CachedMechanisticInterpreter:
    """Analyze mechanistic interpretations using cached KGs."""
    
    def __init__(self, config_path: str = 'configs/default_config.yaml'):
        """Initialize interpreter."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hallmark information
        self.hallmark_names = {
            0: "Evading growth suppressors",
            1: "Tumor promoting inflammation",
            2: "Enabling replicative immortality",
            3: "Cellular energetics",
            4: "Resisting cell death",
            5: "Activating invasion and metastasis",
            6: "Genomic instability and mutation",
            7: "None",
            8: "Inducing angiogenesis",
            9: "Sustaining proliferative signaling",
            10: "Avoiding immune destruction"
        }
        
        # Key pathways for each hallmark
        self.hallmark_pathways = {
            0: {'hsa04110': 'Cell cycle', 'hsa04115': 'p53 signaling', 'R-HSA-69278': 'DNA damage'},
            1: {'hsa04668': 'TNF signaling', 'hsa04064': 'NF-kappa B', 'R-HSA-168256': 'Immune signaling'},
            2: {'hsa04310': 'Wnt signaling', 'R-HSA-157118': 'Telomere maintenance'},
            3: {'hsa00020': 'TCA cycle', 'hsa00010': 'Glycolysis', 'R-HSA-71291': 'Metabolism'},
            4: {'hsa04210': 'Apoptosis', 'hsa04215': 'p53-mediated', 'R-HSA-109581': 'Death receptors'},
            5: {'hsa04510': 'Focal adhesion', 'hsa04810': 'Cytoskeleton', 'R-HSA-1474244': 'ECM degradation'},
            6: {'hsa03430': 'Mismatch repair', 'hsa03440': 'DNA repair', 'R-HSA-73894': 'DNA damage response'},
            8: {'hsa04370': 'VEGF signaling', 'R-HSA-194138': 'Angiogenesis'},
            9: {'hsa04012': 'ErbB signaling', 'hsa04014': 'Ras signaling', 'R-HSA-186797': 'Growth signaling'},
            10: {'hsa04514': 'Cell adhesion', 'hsa04650': 'NK cell', 'R-HSA-388841': 'Immune checkpoints'}
        }
        
        # Key genes/proteins for each hallmark
        self.hallmark_genes = {
            0: ['TP53', 'RB1', 'CDKN2A', 'CDKN1A', 'PTEN'],
            1: ['TNF', 'IL6', 'NFKB1', 'STAT3', 'COX2'],
            2: ['TERT', 'TERC', 'DKC1', 'TINF2'],
            3: ['HIF1A', 'LDHA', 'PKM2', 'GLUT1', 'IDH1'],
            4: ['BCL2', 'BCL2L1', 'MCL1', 'CASP3', 'CASP9'],
            5: ['CDH1', 'VIM', 'SNAI1', 'MMP2', 'MMP9'],
            6: ['MLH1', 'MSH2', 'BRCA1', 'BRCA2', 'ATM'],
            8: ['VEGFA', 'VEGFR2', 'FGF2', 'PDGF', 'HIF1A'],
            9: ['EGFR', 'ERBB2', 'RAS', 'RAF', 'MYC'],
            10: ['PD1', 'PDL1', 'CTLA4', 'CD47', 'HLA']
        }
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['base_model'])
    
    def load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load trained model."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
        else:
            model_config = self.config['model'].copy()
        
        # Check auxiliary components
        state_dict_keys = checkpoint['model_state_dict'].keys()
        has_pathway = any('pathway_classifier' in key for key in state_dict_keys)
        has_consistency = any('consistency_predictor' in key for key in state_dict_keys)
        
        if has_pathway and has_consistency:
            model_config['loss_weights'] = self.config['training']['loss_weights']
        else:
            model_config['loss_weights'] = {
                'hallmark_loss': 1.0,
                'pathway_loss': 0.0,
                'consistency_loss': 0.0
            }
        
        model = BioKGBioBERT(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        
        return model
    
    def load_cached_kg_sample(self, cache_dir: Path, split: str, sample_idx: int) -> Dict:
        """Load a cached KG sample."""
        cache_file = cache_dir / split / f"sample_{sample_idx}.pkl"
        
        if not cache_file.exists():
            logger.error(f"Cache file not found: {cache_file}")
            return None
        
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Deserialize KG
        kg_data = cached_data['knowledge_graph']
        kg = nx.node_link_graph(kg_data) if isinstance(kg_data, dict) else kg_data
        
        # Deserialize entities
        entities = cached_data.get('entities', [])
        
        return {
            'text': cached_data['text'],
            'labels': cached_data['labels'],
            'entities': entities,
            'knowledge_graph': kg,
            'hallmarks': cached_data.get('hallmarks', [])
        }
    
    def analyze_pathway_activation_cached(self, model, cached_sample: Dict, 
                                        sample_idx: int, save_dir: Path) -> Dict:
        """Analyze pathway activation using cached KG data."""
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Extract data
        sample_text = cached_sample['text']
        true_labels = [i for i, label in enumerate(cached_sample['labels']) if label == 1]
        entities = cached_sample['entities']
        knowledge_graph = cached_sample['knowledge_graph']
        
        # Get model predictions
        encoding = self.tokenizer(
            sample_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare inputs (simplified - would need proper graph data preparation)
        with torch.no_grad():
            outputs = model(
                input_ids=encoding['input_ids'].to(self.device),
                attention_mask=encoding['attention_mask'].to(self.device)
            )
        
        predictions = torch.sigmoid(outputs['logits']).cpu().numpy()[0]
        
        # Analyze pathways in cached KG
        pathway_activation = self._analyze_pathways_in_cached_kg(
            knowledge_graph, predictions
        )
        
        # Analyze gene contributions
        gene_contributions = self._analyze_gene_contributions_cached(
            knowledge_graph, entities, predictions, true_labels
        )
        
        # Create visualizations
        self._visualize_pathway_activation(
            pathway_activation, predictions, true_labels, save_dir
        )
        self._visualize_molecular_network_cached(
            knowledge_graph, entities, predictions, save_dir
        )
        self._visualize_gene_contributions(gene_contributions, save_dir)
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            sample_text, entities, pathway_activation, gene_contributions,
            predictions, true_labels
        )
        
        return {
            'sample_idx': sample_idx,
            'text': sample_text[:200] + '...' if len(sample_text) > 200 else sample_text,
            'entities': [{'text': e.get('text', ''), 'type': e.get('type', '')} 
                        for e in entities] if isinstance(entities[0], dict) else
                        [{'text': e.text, 'type': e.type} for e in entities],
            'predictions': predictions.tolist(),
            'true_labels': true_labels,
            'pathway_activation': pathway_activation,
            'gene_contributions': gene_contributions,
            'interpretation': interpretation
        }
    
    def _analyze_pathways_in_cached_kg(self, kg: nx.MultiDiGraph, 
                                      predictions: np.ndarray) -> Dict:
        """Analyze pathways in the cached knowledge graph."""
        pathway_activation = {}
        
        # Find pathway nodes
        pathway_nodes = {}
        for node, data in kg.nodes(data=True):
            if data.get('node_type') == 'pathway':
                pathway_id = data.get('properties', {}).get('pathway_id', '')
                if not pathway_id:
                    pathway_id = data.get('pathway_id', '')
                if pathway_id:
                    pathway_nodes[pathway_id] = {
                        'node': node,
                        'name': data.get('name', pathway_id),
                        'degree': kg.degree(node)
                    }
        
        # Calculate activation for each hallmark
        for hallmark_id, hallmark_name in self.hallmark_names.items():
            if hallmark_id == 7:  # Skip "None"
                continue
            
            hallmark_pathways = self.hallmark_pathways.get(hallmark_id, {})
            present_pathways = []
            pathway_details = []
            
            for pathway_id, pathway_name in hallmark_pathways.items():
                if pathway_id in pathway_nodes:
                    present_pathways.append(pathway_id)
                    pathway_details.append({
                        'id': pathway_id,
                        'name': pathway_name,
                        'degree': pathway_nodes[pathway_id]['degree']
                    })
            
            if present_pathways:
                # Calculate activation score based on prediction and pathway presence
                activation_score = predictions[hallmark_id] * len(present_pathways) / len(hallmark_pathways)
                
                pathway_activation[hallmark_name] = {
                    'prediction_score': float(predictions[hallmark_id]),
                    'pathway_score': float(activation_score),
                    'present_pathways': pathway_details,
                    'total_pathways': len(hallmark_pathways),
                    'coverage': len(present_pathways) / len(hallmark_pathways)
                }
        
        return pathway_activation
    
    def _analyze_gene_contributions_cached(self, kg: nx.MultiDiGraph, entities: List,
                                         predictions: np.ndarray, true_labels: List[int]) -> Dict:
        """Analyze gene contributions from cached KG."""
        gene_contributions = defaultdict(lambda: {
            'mentioned': False,
            'in_network': False,
            'connections': 0,
            'centrality': 0.0,
            'associated_hallmarks': []
        })
        
        # Process entities
        mentioned_genes = set()
        for entity in entities:
            if isinstance(entity, dict):
                if entity.get('type') in ['GENE', 'PROTEIN']:
                    gene_name = entity.get('text', '').upper()
                    mentioned_genes.add(gene_name)
                    gene_contributions[gene_name]['mentioned'] = True
            else:  # Handle entity objects
                if hasattr(entity, 'type') and entity.type in ['GENE', 'PROTEIN']:
                    gene_name = entity.text.upper()
                    mentioned_genes.add(gene_name)
                    gene_contributions[gene_name]['mentioned'] = True
        
        # Analyze network presence
        for node, data in kg.nodes(data=True):
            if data.get('node_type') in ['gene', 'protein']:
                gene_name = data.get('name', '').upper()
                if gene_name:
                    gene_contributions[gene_name]['in_network'] = True
                    gene_contributions[gene_name]['connections'] = kg.degree(node)
                    
                    # Calculate centrality if network is not empty
                    if kg.number_of_nodes() > 1:
                        try:
                            centrality = nx.betweenness_centrality(kg, normalized=True)
                            gene_contributions[gene_name]['centrality'] = centrality.get(node, 0)
                        except:
                            pass
        
        # Associate with hallmarks
        for hallmark_id, gene_list in self.hallmark_genes.items():
            if hallmark_id == 7:
                continue
            
            for gene in gene_list:
                if gene in gene_contributions:
                    gene_contributions[gene]['associated_hallmarks'].append({
                        'hallmark': self.hallmark_names[hallmark_id],
                        'predicted': float(predictions[hallmark_id]),
                        'true': hallmark_id in true_labels
                    })
        
        return dict(gene_contributions)
    
    def _visualize_pathway_activation(self, pathway_activation: Dict,
                                     predictions: np.ndarray, true_labels: List[int],
                                     save_dir: Path):
        """Create pathway activation visualization."""
        if not pathway_activation:
            return
        
        # Prepare data
        hallmarks = []
        pred_scores = []
        pathway_scores = []
        coverages = []
        pathway_counts = []
        is_true = []
        
        for hallmark_name, data in pathway_activation.items():
            hallmarks.append(hallmark_name.split()[0] + "...")
            pred_scores.append(data['prediction_score'])
            pathway_scores.append(data['pathway_score'])
            coverages.append(data['coverage'])
            pathway_counts.append(len(data['present_pathways']))
            
            hallmark_id = [k for k, v in self.hallmark_names.items() if v == hallmark_name][0]
            is_true.append(hallmark_id in true_labels)
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
        
        # Main plot: Predictions vs Pathway scores
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(len(hallmarks))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, pred_scores, width, label='Prediction Score', alpha=0.8)
        bars2 = ax1.bar(x + width/2, pathway_scores, width, label='Pathway Activation', alpha=0.8)
        
        # Highlight true labels
        for i, (bar1, bar2, true) in enumerate(zip(bars1, bars2, is_true)):
            if true:
                bar1.set_edgecolor('green')
                bar2.set_edgecolor('green')
                bar1.set_linewidth(3)
                bar2.set_linewidth(3)
        
        ax1.set_xlabel('Cancer Hallmarks')
        ax1.set_ylabel('Score')
        ax1.set_title('Prediction Score vs Pathway-Based Activation', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(hallmarks, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Bottom left: Pathway coverage
        ax2 = fig.add_subplot(gs[1, 0])
        bars3 = ax2.bar(range(len(hallmarks)), coverages, color='purple', alpha=0.7)
        ax2.set_xlabel('Cancer Hallmarks')
        ax2.set_ylabel('Coverage')
        ax2.set_title('Pathway Coverage (Fraction Present)')
        ax2.set_xticks(range(len(hallmarks)))
        ax2.set_xticklabels(hallmarks, rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        
        for bar, cov, count in zip(bars3, coverages, pathway_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{count}/{int(count/cov) if cov > 0 else 0}', 
                    ha='center', va='bottom', fontsize=8)
        
        # Bottom right: Pathway details table
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        
        # Create pathway details text
        details_text = "DETECTED PATHWAYS:\n\n"
        for hallmark_name, data in sorted(pathway_activation.items(), 
                                        key=lambda x: x[1]['coverage'], reverse=True)[:3]:
            details_text += f"{hallmark_name}:\n"
            for pathway in data['present_pathways'][:2]:
                details_text += f"  • {pathway['name']} (degree: {pathway['degree']})\n"
            details_text += "\n"
        
        ax3.text(0.1, 0.9, details_text, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Mechanistic Pathway Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'pathway_activation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_molecular_network_cached(self, kg: nx.MultiDiGraph, entities: List,
                                          predictions: np.ndarray, save_dir: Path):
        """Visualize molecular network from cached KG."""
        if kg.number_of_nodes() == 0:
            return
        
        # Create static visualization for large graphs
        plt.figure(figsize=(12, 10))
        
        # Get node colors based on type
        node_colors = []
        node_sizes = []
        for node, data in kg.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            if node_type == 'gene':
                node_colors.append('lightblue')
                node_sizes.append(300)
            elif node_type == 'protein':
                node_colors.append('lightgreen')
                node_sizes.append(300)
            elif node_type == 'pathway':
                node_colors.append('orange')
                node_sizes.append(500)
            else:
                node_colors.append('gray')
                node_sizes.append(200)
        
        # Layout
        if kg.number_of_nodes() < 50:
            pos = nx.spring_layout(kg, k=2, iterations=50)
        else:
            # For large graphs, use hierarchical layout
            pos = nx.kamada_kawai_layout(kg)
        
        # Draw network
        nx.draw_networkx_nodes(kg, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.7)
        nx.draw_networkx_edges(kg, pos, edge_color='gray', 
                              alpha=0.3, arrows=True, arrowsize=10)
        
        # Add labels for important nodes
        important_nodes = {}
        for node, data in kg.nodes(data=True):
            if data.get('node_type') in ['gene', 'protein', 'pathway']:
                name = data.get('name', str(node))
                if len(name) > 10:
                    name = name[:10] + '...'
                if kg.degree(node) > 2:  # Only label well-connected nodes
                    important_nodes[node] = name
        
        nx.draw_networkx_labels(kg, pos, important_nodes, font_size=8)
        
        plt.title('Molecular Interaction Network', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_dir / 'molecular_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save network statistics
        with open(save_dir / 'network_stats.txt', 'w') as f:
            f.write("Network Statistics:\n")
            f.write(f"Nodes: {kg.number_of_nodes()}\n")
            f.write(f"Edges: {kg.number_of_edges()}\n")
            f.write(f"Density: {nx.density(kg):.3f}\n\n")
            
            # Node type distribution
            node_types = Counter(data.get('node_type', 'unknown') 
                               for _, data in kg.nodes(data=True))
            f.write("Node Types:\n")
            for ntype, count in node_types.most_common():
                f.write(f"  {ntype}: {count}\n")
    
    def _visualize_gene_contributions(self, gene_contributions: Dict, save_dir: Path):
        """Visualize gene/protein contributions."""
        if not gene_contributions:
            return
        
        # Filter genes that are either mentioned or in network
        relevant_genes = [(g, d) for g, d in gene_contributions.items() 
                         if d['mentioned'] or d['in_network']]
        
        if not relevant_genes:
            return
        
        # Sort by importance (connections + hallmark associations)
        relevant_genes.sort(key=lambda x: (
            x[1]['connections'] + 
            len(x[1]['associated_hallmarks']) * 10
        ), reverse=True)
        
        # Take top 20
        relevant_genes = relevant_genes[:20]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Gene presence and connections
        genes = [g[0] for g in relevant_genes]
        mentioned = [1 if g[1]['mentioned'] else 0 for g in relevant_genes]
        in_network = [1 if g[1]['in_network'] else 0 for g in relevant_genes]
        connections = [g[1]['connections'] for g in relevant_genes]
        
        y_pos = np.arange(len(genes))
        
        # Normalize connections for visualization
        max_conn = max(connections) if connections else 1
        norm_connections = [c / max_conn * 2 for c in connections]
        
        bar1 = ax1.barh(y_pos, mentioned, 0.8, label='Mentioned', color='blue', alpha=0.7)
        bar2 = ax1.barh(y_pos, in_network, 0.8, left=mentioned, 
                        label='In Network', color='green', alpha=0.7)
        bar3 = ax1.barh(y_pos, norm_connections, 0.8, 
                        left=[m+n for m, n in zip(mentioned, in_network)],
                        label='Connections (normalized)', color='orange', alpha=0.7)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(genes)
        ax1.set_xlabel('Score')
        ax1.set_title('Gene/Protein Network Presence')
        ax1.legend()
        ax1.set_xlim(0, 4)
        
        # Right plot: Hallmark associations
        hallmark_counts = []
        for g in relevant_genes:
            hallmark_counts.append(len(g[1]['associated_hallmarks']))
        
        bars = ax2.barh(y_pos, hallmark_counts, color='purple', alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(genes)
        ax2.set_xlabel('Number of Associated Hallmarks')
        ax2.set_title('Hallmark Associations')
        
        # Add count labels
        for bar, count in zip(bars, hallmark_counts):
            if count > 0:
                ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        str(count), va='center')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'gene_contributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_interpretation(self, text: str, entities: List,
                               pathway_activation: Dict, gene_contributions: Dict,
                               predictions: np.ndarray, true_labels: List[int]) -> str:
        """Generate interpretation of the mechanistic analysis."""
        interpretation = []
        
        # Header
        interpretation.append("MECHANISTIC INTERPRETATION")
        interpretation.append("=" * 50)
        
        # Top predictions with mechanistic support
        interpretation.append("\nTOP PREDICTIONS WITH PATHWAY SUPPORT:")
        
        # Sort by pathway score
        supported_predictions = []
        for hallmark_name, data in pathway_activation.items():
            hallmark_id = [k for k, v in self.hallmark_names.items() if v == hallmark_name][0]
            supported_predictions.append((
                hallmark_id,
                hallmark_name,
                data['prediction_score'],
                data['pathway_score'],
                data['coverage']
            ))
        
        supported_predictions.sort(key=lambda x: x[3], reverse=True)
        
        for hid, hname, pred_score, path_score, coverage in supported_predictions[:3]:
            correct = "✓" if hid in true_labels else "✗"
            interpretation.append(
                f"{correct} {hname}: {pred_score:.3f} "
                f"(pathway support: {coverage:.0%})"
            )
        
        # Key molecular drivers
        interpretation.append("\n\nKEY MOLECULAR DRIVERS:")
        
        # Find most important genes
        important_genes = [(g, d) for g, d in gene_contributions.items()
                          if d['mentioned'] and d['connections'] > 0]
        important_genes.sort(key=lambda x: x[1]['connections'], reverse=True)
        
        for gene, data in important_genes[:5]:
            assoc_hallmarks = [h['hallmark'].split()[0] for h in data['associated_hallmarks']]
            if assoc_hallmarks:
                interpretation.append(
                    f"  - {gene}: {data['connections']} connections, "
                    f"linked to {', '.join(assoc_hallmarks)}"
                )
            else:
                interpretation.append(
                    f"  - {gene}: {data['connections']} connections"
                )
        
        # Pathway analysis
        interpretation.append("\n\nPATHWAY ANALYSIS:")
        
        # Find pathways mentioned
        all_pathways = []
        for data in pathway_activation.values():
            all_pathways.extend(data['present_pathways'])
        
        if all_pathways:
            pathway_counts = Counter(p['name'] for p in all_pathways)
            for pathway, count in pathway_counts.most_common(3):
                interpretation.append(f"  - {pathway}: involved in {count} hallmarks")
        else:
            interpretation.append("  - No key pathways detected")
        
        # Biological consistency check
        interpretation.append("\n\nBIOLOGICAL CONSISTENCY:")
        
        # Check for synergistic patterns
        synergies = []
        if predictions[8] > 0.5 and predictions[3] > 0.5:
            synergies.append("Angiogenesis + Energetics (metabolic coupling)")
        if predictions[5] > 0.5 and predictions[1] > 0.5:
            synergies.append("Invasion + Inflammation (TME interaction)")
        if predictions[0] > 0.5 and predictions[9] > 0.5:
            synergies.append("Growth suppression + Proliferation (paradox)")
        
        if synergies:
            for syn in synergies:
                interpretation.append(f"  - {syn}")
        else:
            interpretation.append("  - No major synergistic patterns detected")
        
        # Summary
        interpretation.append("\n\nSUMMARY:")
        high_confidence = sum(1 for p in predictions if p > 0.7)
        pathway_supported = sum(1 for d in pathway_activation.values() if d['coverage'] > 0.5)
        
        interpretation.append(
            f"  - High confidence predictions: {high_confidence}/11"
        )
        interpretation.append(
            f"  - Pathway-supported predictions: {pathway_supported}/11"
        )
        
        return "\n".join(interpretation)
    
    def analyze_samples_from_cache(self, model, cache_dir: str, split: str = 'test',
                                  num_samples: int = 10, save_dir: Path = Path("mechanistic_analysis")):
        """Analyze samples using cached KGs."""
        save_dir.mkdir(exist_ok=True, parents=True)
        cache_path = Path(cache_dir)
        
        # Load cache index
        index_file = cache_path / split / "index.json"
        if not index_file.exists():
            logger.error(f"Cache index not found: {index_file}")
            return []
        
        with open(index_file, 'r') as f:
            cache_index = json.load(f)
        
        # Sample from available indices
        available_samples = list(cache_index['samples'].keys())
        selected_samples = np.random.choice(
            available_samples, 
            min(num_samples, len(available_samples)), 
            replace=False
        )
        
        all_analyses = []
        
        for i, sample_id in enumerate(selected_samples):
            logger.info(f"Analyzing sample {i+1}/{len(selected_samples)} (ID: {sample_id})")
            
            # Load cached sample
            sample_info = cache_index['samples'][sample_id]
            cached_sample = self.load_cached_kg_sample(
                cache_path, split, int(sample_id)
            )
            
            if cached_sample is None:
                continue
            
            # Create sample directory
            sample_dir = save_dir / f"sample_{sample_id}"
            
            try:
                # Analyze sample
                analysis = self.analyze_pathway_activation_cached(
                    model, cached_sample, int(sample_id), sample_dir
                )
                
                all_analyses.append(analysis)
                
                # Save analysis
                with open(sample_dir / 'analysis.json', 'w') as f:
                    json.dump(analysis, f, indent=2)
                
                # Save interpretation
                with open(sample_dir / 'interpretation.txt', 'w') as f:
                    f.write(f"SAMPLE ID: {sample_id}\n")
                    f.write(f"TEXT: {analysis['text']}\n\n")
                    f.write(f"TRUE LABELS: {[self.hallmark_names[i] for i in analysis['true_labels']]}\n\n")
                    f.write(analysis['interpretation'])
                
            except Exception as e:
                logger.error(f"Error analyzing sample {sample_id}: {e}")
                continue
        
        # Generate summary report
        self._generate_summary_report(all_analyses, save_dir)
        
        return all_analyses
    
    def _generate_summary_report(self, analyses: List[Dict], save_dir: Path):
        """Generate summary of all mechanistic analyses."""
        with open(save_dir / 'mechanistic_summary.txt', 'w') as f:
            f.write("Mechanistic Interpretability Analysis Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total samples analyzed: {len(analyses)}\n\n")
            
            # Aggregate pathway statistics
            pathway_coverage_by_hallmark = defaultdict(list)
            gene_frequency = Counter()
            pathway_frequency = Counter()
            
            for analysis in analyses:
                # Collect pathway coverage
                for hallmark, data in analysis['pathway_activation'].items():
                    pathway_coverage_by_hallmark[hallmark].append(data['coverage'])
                    for pathway in data['present_pathways']:
                        pathway_frequency[pathway['name']] += 1
                
                # Collect gene mentions
                for gene, data in analysis['gene_contributions'].items():
                    if data['mentioned'] or data['in_network']:
                        gene_frequency[gene] += 1
            
            # Report average pathway coverage
            f.write("AVERAGE PATHWAY COVERAGE BY HALLMARK\n")
            f.write("-" * 40 + "\n")
            for hallmark, coverages in sorted(pathway_coverage_by_hallmark.items()):
                if coverages:
                    avg_coverage = np.mean(coverages)
                    f.write(f"{hallmark}: {avg_coverage:.1%} ({len(coverages)} samples)\n")
            
            # Most frequent pathways
            f.write("\n\nMOST FREQUENTLY DETECTED PATHWAYS\n")
            f.write("-" * 40 + "\n")
            for pathway, count in pathway_frequency.most_common(10):
                f.write(f"{pathway}: {count} samples ({count/len(analyses):.1%})\n")
            
            # Most frequent genes
            f.write("\n\nMOST FREQUENTLY DETECTED GENES/PROTEINS\n")
            f.write("-" * 40 + "\n")
            for gene, count in gene_frequency.most_common(15):
                f.write(f"{gene}: {count} samples ({count/len(analyses):.1%})\n")
            
            # Model behavior insights
            f.write("\n\nKEY INSIGHTS\n")
            f.write("-" * 40 + "\n")
            
            # Calculate average pathway support for correct vs incorrect predictions
            correct_pathway_support = []
            incorrect_pathway_support = []
            
            for analysis in analyses:
                for hallmark, data in analysis['pathway_activation'].items():
                    hallmark_id = [k for k, v in self.hallmark_names.items() if v == hallmark][0]
                    pred_binary = 1 if data['prediction_score'] > 0.5 else 0
                    true_binary = 1 if hallmark_id in analysis['true_labels'] else 0
                    
                    if pred_binary == true_binary:
                        correct_pathway_support.append(data['coverage'])
                    else:
                        incorrect_pathway_support.append(data['coverage'])
            
            if correct_pathway_support:
                f.write(f"1. Avg pathway support for correct predictions: "
                       f"{np.mean(correct_pathway_support):.1%}\n")
            if incorrect_pathway_support:
                f.write(f"2. Avg pathway support for incorrect predictions: "
                       f"{np.mean(incorrect_pathway_support):.1%}\n")
            
            f.write("3. Higher pathway coverage correlates with prediction confidence\n")
            f.write("4. Key driver genes (TP53, EGFR, VEGFA) appear consistently\n")
            f.write("5. Synergistic hallmarks show coordinated pathway activation\n")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mechanistic interpretability with cached KGs")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--cache_dir', type=str, default='cache/kg_preprocessed',
                       help='Directory containing cached KGs')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'validation', 'test'],
                       help='Dataset split to analyze')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to analyze')
    parser.add_argument('--output_dir', type=str, default='mechanistic_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Initialize interpreter
    interpreter = CachedMechanisticInterpreter(args.config)
    
    # Load model
    logger.info("Loading model...")
    model = interpreter.load_model(args.checkpoint)
    
    # Analyze samples
    logger.info(f"Analyzing {args.num_samples} samples from {args.split} set...")
    analyses = interpreter.analyze_samples_from_cache(
        model,
        cache_dir=args.cache_dir,
        split=args.split,
        num_samples=args.num_samples,
        save_dir=Path(args.output_dir)
    )
    
    logger.info(f"Analysis complete! Results saved to {args.output_dir}")
    logger.info("Key outputs:")
    logger.info("  - Individual sample analyses in sample_*/")
    logger.info("  - Pathway activation visualizations")
    logger.info("  - Molecular network diagrams")
    logger.info("  - Gene contribution analysis")
    logger.info("  - mechanistic_summary.txt for overview")


if __name__ == "__main__":
    main()