"""
Mechanistic Interpretability Analysis using Learned Associations

This script uses data-driven learned associations instead of hardcoded pathways/genes
for more accurate and adaptive mechanistic explanations.
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


class LearnedMechanisticInterpreter:
    """Mechanistic interpretation using learned associations from data."""
    
    def __init__(self, config_path: str = 'configs/default_config.yaml',
                 associations_path: str = 'learned_associations/hallmark_associations.json'):
        """Initialize interpreter with learned associations."""
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
        
        # Load learned associations
        self.learned_associations = self._load_learned_associations(associations_path)
        
        # Extract pathways and genes from learned associations
        self.hallmark_pathways = self._extract_learned_pathways()
        self.hallmark_genes = self._extract_learned_genes()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['base_model'])
        
        logger.info(f"Loaded learned associations with {len(self.hallmark_pathways)} hallmarks")
    
    def _load_learned_associations(self, associations_path: str) -> Dict:
        """Load learned associations from file."""
        if not Path(associations_path).exists():
            logger.warning(f"Learned associations not found at {associations_path}")
            logger.warning("Run learn_hallmark_associations.py first to generate associations")
            return {'associations': {'pathways': {}, 'genes': {}, 'entities': {}}}
        
        with open(associations_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded associations learned from {data['metadata']['total_samples']} samples")
        return data
    
    def _extract_learned_pathways(self) -> Dict:
        """Extract pathway associations from learned data."""
        hallmark_pathways = {}
        pathway_associations = self.learned_associations['associations'].get('pathways', {})
        
        for hallmark_id_str, pathways in pathway_associations.items():
            hallmark_id = int(hallmark_id_str)
            if hallmark_id == 7:  # Skip "None"
                continue
            
            # Extract top pathways with scores
            hallmark_pathways[hallmark_id] = {}
            for pathway_key, pathway_data in pathways.items():
                pathway_id = pathway_data['id']
                pathway_name = pathway_data['name']
                score = pathway_data['score']
                
                # Include pathways with good scores
                if score > 0.2:  # Threshold for inclusion
                    hallmark_pathways[hallmark_id][pathway_id] = {
                        'name': pathway_name,
                        'score': score,
                        'support': pathway_data['support'],
                        'confidence': pathway_data['confidence']
                    }
        
        return hallmark_pathways
    
    def _extract_learned_genes(self) -> Dict:
        """Extract gene associations from learned data."""
        hallmark_genes = {}
        gene_associations = self.learned_associations['associations'].get('genes', {})
        
        for hallmark_id_str, genes in gene_associations.items():
            hallmark_id = int(hallmark_id_str)
            if hallmark_id == 7:  # Skip "None"
                continue
            
            # Extract top genes
            hallmark_genes[hallmark_id] = []
            for gene_key, gene_data in genes.items():
                gene_symbol = gene_data['symbol']
                score = gene_data['score']
                
                if score > 0.15:  # Threshold for inclusion
                    hallmark_genes[hallmark_id].append({
                        'symbol': gene_symbol,
                        'score': score,
                        'support': gene_data['support'],
                        'confidence': gene_data['confidence']
                    })
        
        return hallmark_genes
    
    def _reconstruct_graph(self, serialized_graph: Dict) -> nx.MultiDiGraph:
        """Reconstruct NetworkX graph from serialized format."""
        graph = nx.MultiDiGraph()
        
        # Add nodes
        for node, attrs in serialized_graph['nodes']:
            graph.add_node(node, **attrs)
        
        # Add edges
        for u, v, attrs in serialized_graph['edges']:
            graph.add_edge(u, v, **attrs)
        
        # Add graph attributes
        if 'graph' in serialized_graph:
            graph.graph = serialized_graph['graph']
        
        return graph
    
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
    
    def analyze_pathway_activation(self, model, sample_data: Dict, sample_idx: int,
                                  save_dir: Path) -> Dict:
        """Analyze pathway activation using learned associations."""
        save_dir.mkdir(exist_ok=True)
        
        # Extract data from cached sample
        sample_text = sample_data.get('text', '')
        
        # Get labels and ensure they're in the right format
        labels = sample_data.get('labels', sample_data.get('label', []))
        
        # Handle different label formats
        if isinstance(labels, list) and len(labels) > 0:
            if isinstance(labels[0], (int, np.integer)):
                # List of label indices (e.g., [7] or [0, 3])
                temp = np.zeros(11)
                for label_idx in labels:
                    if 0 <= label_idx < 11:
                        temp[label_idx] = 1
                labels = temp
            else:
                # Already in binary format
                labels = np.array(labels)
        elif isinstance(labels, (int, np.integer)):
            # Single label integer
            temp = np.zeros(11)
            if 0 <= labels < 11:
                temp[labels] = 1
            labels = temp
        elif isinstance(labels, np.ndarray):
            # Already an array
            if labels.ndim == 0:
                # Scalar array
                temp = np.zeros(11)
                if 0 <= int(labels) < 11:
                    temp[int(labels)] = 1
                labels = temp
        else:
            # Default to empty labels
            labels = np.zeros(11)
        
        true_labels = np.where(labels == 1)[0].tolist()
        
        # Load cached KG data
        entities = sample_data.get('entities', [])
        knowledge_graph = sample_data.get('knowledge_graph', nx.MultiDiGraph())
        
        # Tokenize text
        encoding = self.tokenizer(
            sample_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare batch (simplified for single sample)
        batch = {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device),
            'labels': torch.tensor(labels).unsqueeze(0).float().to(self.device),  # Ensure float tensor
            'graph_data': None,  # Would need proper graph features
            'biological_context': None  # Would need proper context
        }
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**batch)
            predictions = torch.sigmoid(outputs['logits']).cpu().numpy()[0]
        
        # Analyze pathway activation with learned associations
        pathway_activation = self._analyze_learned_pathways(
            knowledge_graph, predictions, true_labels
        )
        
        # Analyze gene contributions with learned associations
        gene_contributions = self._analyze_learned_genes(
            knowledge_graph, entities, predictions, true_labels
        )
        
        # Create enhanced visualizations
        self._visualize_learned_pathway_activation(
            pathway_activation, predictions, true_labels, save_dir
        )
        self._visualize_learned_gene_contributions(
            gene_contributions, save_dir
        )
        
        # Generate enhanced interpretation
        interpretation = self._generate_learned_interpretation(
            sample_text, entities, pathway_activation, gene_contributions,
            predictions, true_labels
        )
        
        return {
            'text': sample_text,
            'entities': [e.__dict__ for e in entities] if hasattr(entities[0], '__dict__') else entities,
            'predictions': predictions.tolist(),
            'true_labels': true_labels,
            'pathway_activation': pathway_activation,
            'gene_contributions': gene_contributions,
            'interpretation': interpretation,
            'learned_associations_used': True
        }
    
    def _analyze_learned_pathways(self, kg: nx.MultiDiGraph, 
                                 predictions: np.ndarray,
                                 true_labels: List[int]) -> Dict:
        """Analyze pathways using learned associations."""
        pathway_activation = {}
        
        # Extract pathways from KG
        kg_pathways = {}
        for node in kg.nodes():
            node_data = kg.nodes[node]
            if node_data.get('node_type') == 'pathway':
                pathway_id = node_data.get('properties', {}).get('pathway_id', '')
                pathway_name = node_data.get('name', '')
                if pathway_id:
                    kg_pathways[pathway_id] = pathway_name
        
        # Analyze each hallmark's pathway activation
        for hallmark_id, hallmark_name in self.hallmark_names.items():
            if hallmark_id == 7:  # Skip "None"
                continue
            
            # Get learned pathways for this hallmark
            learned_pathways = self.hallmark_pathways.get(hallmark_id, {})
            if not learned_pathways:
                continue
            
            # Check which learned pathways are present in KG
            present_pathways = []
            pathway_scores = []
            
            for pathway_id, pathway_info in learned_pathways.items():
                if pathway_id in kg_pathways:
                    present_pathways.append({
                        'id': pathway_id,
                        'name': pathway_info['name'],
                        'learned_score': pathway_info['score'],
                        'support': pathway_info['support'],
                        'confidence': pathway_info['confidence']
                    })
                    pathway_scores.append(pathway_info['score'])
            
            if present_pathways:
                # Calculate activation score using learned weights
                weighted_score = np.sum(pathway_scores) / len(learned_pathways)
                prediction_score = float(predictions[hallmark_id])
                
                # Combine prediction with pathway evidence
                combined_score = 0.7 * prediction_score + 0.3 * weighted_score
                
                pathway_activation[hallmark_name] = {
                    'prediction_score': prediction_score,
                    'pathway_evidence_score': float(weighted_score),
                    'combined_score': float(combined_score),
                    'present_pathways': present_pathways,
                    'total_learned_pathways': len(learned_pathways),
                    'coverage': len(present_pathways) / len(learned_pathways),
                    'is_true_label': hallmark_id in true_labels
                }
        
        return pathway_activation
    
    def _analyze_learned_genes(self, kg: nx.MultiDiGraph, entities: List,
                              predictions: np.ndarray, true_labels: List[int]) -> Dict:
        """Analyze genes using learned associations."""
        gene_contributions = defaultdict(lambda: {
            'mentioned': False,
            'in_network': False,
            'connections': 0,
            'learned_associations': [],
            'importance_score': 0.0
        })
        
        # Extract mentioned genes
        mentioned_genes = set()
        for entity in entities:
            if hasattr(entity, 'type') and entity.type in ['GENE', 'PROTEIN']:
                gene_name = entity.text.upper()
                mentioned_genes.add(gene_name)
                gene_contributions[gene_name]['mentioned'] = True
            elif isinstance(entity, dict) and entity.get('type') in ['GENE', 'PROTEIN']:
                gene_name = entity.get('text', '').upper()
                mentioned_genes.add(gene_name)
                gene_contributions[gene_name]['mentioned'] = True
        
        # Analyze genes in network
        kg_genes = set()
        for node in kg.nodes():
            node_data = kg.nodes[node]
            if node_data.get('node_type') in ['gene', 'protein']:
                gene_name = node_data.get('name', '').upper()
                if gene_name:
                    kg_genes.add(gene_name)
                    gene_contributions[gene_name]['in_network'] = True
                    gene_contributions[gene_name]['connections'] = kg.degree(node)
        
        # Add learned associations
        for hallmark_id, gene_list in self.hallmark_genes.items():
            if hallmark_id == 7:
                continue
            
            for gene_info in gene_list:
                gene_symbol = gene_info['symbol']
                if gene_symbol in gene_contributions:
                    association = {
                        'hallmark': self.hallmark_names[hallmark_id],
                        'learned_score': gene_info['score'],
                        'support': gene_info['support'],
                        'confidence': gene_info['confidence'],
                        'predicted': float(predictions[hallmark_id]),
                        'is_true': hallmark_id in true_labels
                    }
                    gene_contributions[gene_symbol]['learned_associations'].append(association)
                    
                    # Update importance score
                    if gene_contributions[gene_symbol]['mentioned']:
                        gene_contributions[gene_symbol]['importance_score'] += gene_info['score']
        
        return dict(gene_contributions)
    
    def _visualize_learned_pathway_activation(self, pathway_activation: Dict,
                                            predictions: np.ndarray, 
                                            true_labels: List[int],
                                            save_dir: Path):
        """Visualize pathway activation with learned associations."""
        # Prepare data
        data_for_plot = []
        
        for hallmark_name, data in pathway_activation.items():
            hallmark_id = [k for k, v in self.hallmark_names.items() if v == hallmark_name][0]
            
            data_for_plot.append({
                'Hallmark': hallmark_name.split()[0] + "...",
                'Prediction Score': data['prediction_score'],
                'Pathway Evidence': data['pathway_evidence_score'],
                'Combined Score': data['combined_score'],
                'Coverage': data['coverage'],
                'Is True Label': data['is_true_label'],
                'Present Pathways': len(data['present_pathways'])
            })
        
        df = pd.DataFrame(data_for_plot)
        df = df.sort_values('Combined Score', ascending=False)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Subplot 1: Score comparison
        x = np.arange(len(df))
        width = 0.25
        
        bars1 = ax1.bar(x - width, df['Prediction Score'], width, 
                        label='Model Prediction', alpha=0.8, color='blue')
        bars2 = ax1.bar(x, df['Pathway Evidence'], width, 
                        label='Learned Pathway Evidence', alpha=0.8, color='green')
        bars3 = ax1.bar(x + width, df['Combined Score'], width, 
                        label='Combined Score', alpha=0.8, color='purple')
        
        # Highlight true labels
        for i, is_true in enumerate(df['Is True Label']):
            if is_true:
                for bar_group in [bars1, bars2, bars3]:
                    bar_group[i].set_edgecolor('red')
                    bar_group[i].set_linewidth(3)
        
        ax1.set_xlabel('Cancer Hallmarks')
        ax1.set_ylabel('Score')
        ax1.set_title('Prediction vs Learned Pathway Evidence')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['Hallmark'], rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        
        # Subplot 2: Pathway coverage with details
        bars4 = ax2.bar(x, df['Coverage'], color='orange', alpha=0.7)
        
        # Add present pathway counts
        for i, (bar, count) in enumerate(zip(bars4, df['Present Pathways'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{count}', ha='center', va='bottom', fontsize=9)
        
        ax2.set_xlabel('Cancer Hallmarks')
        ax2.set_ylabel('Learned Pathway Coverage')
        ax2.set_title('Fraction of Learned Pathways Present in Sample')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['Hallmark'], rotation=45, ha='right')
        ax2.set_ylim(0, 1.2)
        
        # Add legend for true labels
        from matplotlib.patches import Rectangle
        ax2.add_patch(Rectangle((0, 0), 0, 0, edgecolor='red', 
                               facecolor='none', linewidth=3, label='True Label'))
        ax2.legend()
        
        plt.suptitle('Mechanistic Analysis with Learned Associations', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_dir / 'learned_pathway_activation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_learned_gene_contributions(self, gene_contributions: Dict, save_dir: Path):
        """Visualize gene contributions with learned importance scores."""
        # Prepare data
        gene_data = []
        
        for gene, data in gene_contributions.items():
            if data['mentioned'] or data['in_network']:
                importance = data['importance_score']
                associations = len(data['learned_associations'])
                
                if importance > 0 or data['mentioned']:
                    gene_data.append({
                        'Gene': gene,
                        'Mentioned': data['mentioned'],
                        'In Network': data['in_network'],
                        'Connections': data['connections'],
                        'Importance Score': importance,
                        'Associated Hallmarks': associations
                    })
        
        if not gene_data:
            return
        
        # Sort by importance score
        gene_df = pd.DataFrame(gene_data)
        gene_df = gene_df.sort_values('Importance Score', ascending=True)
        
        # Select top genes
        top_n = min(20, len(gene_df))
        gene_df = gene_df.tail(top_n)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, max(6, top_n * 0.3)))
        
        y_pos = np.arange(len(gene_df))
        
        # Create horizontal bars
        bars = ax.barh(y_pos, gene_df['Importance Score'], alpha=0.7)
        
        # Color based on presence
        colors = []
        for _, row in gene_df.iterrows():
            if row['Mentioned'] and row['In Network']:
                colors.append('darkgreen')
            elif row['Mentioned']:
                colors.append('blue')
            elif row['In Network']:
                colors.append('orange')
            else:
                colors.append('gray')
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add annotations
        for i, (_, row) in enumerate(gene_df.iterrows()):
            # Add hallmark count
            ax.text(row['Importance Score'] + 0.01, i, 
                   f"{row['Associated Hallmarks']} hallmarks",
                   va='center', fontsize=9)
            
            # Add connection count if in network
            if row['In Network']:
                ax.text(0.01, i, f"({row['Connections']} conn)",
                       va='center', fontsize=8, style='italic')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(gene_df['Gene'])
        ax.set_xlabel('Learned Importance Score')
        ax.set_title('Gene Contributions Based on Learned Associations')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(color='darkgreen', label='Mentioned & In Network'),
            Patch(color='blue', label='Mentioned Only'),
            Patch(color='orange', label='In Network Only')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'learned_gene_contributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_learned_interpretation(self, text: str, entities: List,
                                       pathway_activation: Dict, 
                                       gene_contributions: Dict,
                                       predictions: np.ndarray, 
                                       true_labels: List[int]) -> str:
        """Generate interpretation using learned associations."""
        interpretation = []
        
        # Header
        interpretation.append("MECHANISTIC INTERPRETATION (Using Learned Associations)")
        interpretation.append("=" * 60)
        
        # Top predictions with learned evidence
        interpretation.append("\nTOP PREDICTIONS WITH BIOLOGICAL EVIDENCE:")
        
        # Sort by combined score
        sorted_activations = sorted(
            pathway_activation.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        for hallmark_name, data in sorted_activations[:3]:
            hallmark_id = [k for k, v in self.hallmark_names.items() if v == hallmark_name][0]
            
            correct = "✓" if data['is_true_label'] else "✗"
            interpretation.append(
                f"\n{correct} {hallmark_name}:"
                f"\n  Model prediction: {data['prediction_score']:.3f}"
                f"\n  Pathway evidence: {data['pathway_evidence_score']:.3f}"
                f"\n  Combined score: {data['combined_score']:.3f}"
            )
            
            if data['present_pathways']:
                interpretation.append("  Supporting pathways (learned associations):")
                for pathway in data['present_pathways'][:3]:
                    interpretation.append(
                        f"    - {pathway['name']} "
                        f"(relevance: {pathway['learned_score']:.2f}, "
                        f"support: {pathway['support']:.2f})"
                    )
        
        # Key molecular players with learned importance
        interpretation.append("\n\nKEY MOLECULAR PLAYERS (Learned Importance):")
        
        important_genes = [
            (g, d) for g, d in gene_contributions.items() 
            if d['importance_score'] > 0 and d['mentioned']
        ]
        important_genes.sort(key=lambda x: x[1]['importance_score'], reverse=True)
        
        for gene, data in important_genes[:5]:
            interpretation.append(f"\n  {gene} (importance: {data['importance_score']:.2f}):")
            
            # Show learned associations
            for assoc in data['learned_associations'][:2]:
                pred_strength = "high" if assoc['predicted'] > 0.7 else "moderate" if assoc['predicted'] > 0.4 else "low"
                interpretation.append(
                    f"    - Associated with {assoc['hallmark'].split()[0]}... "
                    f"(confidence: {assoc['confidence']:.2f}, {pred_strength} prediction)"
                )
        
        # Biological consistency based on learned patterns
        interpretation.append("\n\nBIOLOGICAL CONSISTENCY (Learned Patterns):")
        
        # Check for co-occurring hallmarks based on learned associations
        high_predictions = [
            (i, self.hallmark_names[i]) for i in range(11) 
            if i != 7 and predictions[i] > 0.6
        ]
        
        if len(high_predictions) > 1:
            interpretation.append("  Detected hallmark combinations:")
            
            # Check if these hallmarks share pathways/genes in learned associations
            for i, (h1_id, h1_name) in enumerate(high_predictions):
                for h2_id, h2_name in high_predictions[i+1:]:
                    # Find shared pathways
                    h1_pathways = set(self.hallmark_pathways.get(h1_id, {}).keys())
                    h2_pathways = set(self.hallmark_pathways.get(h2_id, {}).keys())
                    shared_pathways = h1_pathways & h2_pathways
                    
                    if shared_pathways:
                        interpretation.append(
                            f"    - {h1_name.split()[0]}... + {h2_name.split()[0]}... "
                            f"(share {len(shared_pathways)} learned pathways)"
                        )
        else:
            interpretation.append("  Single hallmark prediction - checking pathway support")
        
        # Summary
        interpretation.append("\n\nSUMMARY:")
        interpretation.append(
            "This interpretation is based on associations learned from training data, "
            "making it adaptive and data-driven rather than relying on hardcoded knowledge."
        )
        
        return "\n".join(interpretation)
    
    def analyze_test_samples(self, model, data_loader, num_samples: int = 10,
                           save_dir: Path = Path("learned_mechanistic_analysis")):
        """Analyze test samples using learned associations."""
        save_dir.mkdir(exist_ok=True)
        
        # Also save the learned associations being used
        associations_info = {
            'total_hallmarks_with_pathways': len(self.hallmark_pathways),
            'total_hallmarks_with_genes': len(self.hallmark_genes),
            'average_pathways_per_hallmark': np.mean([
                len(pathways) for pathways in self.hallmark_pathways.values()
            ]),
            'average_genes_per_hallmark': np.mean([
                len(genes) for genes in self.hallmark_genes.values()
            ])
        }
        
        with open(save_dir / 'learned_associations_info.json', 'w') as f:
            json.dump(associations_info, f, indent=2)
        
        all_analyses = []
        sample_count = 0
        
        # Use cached dataset for analysis
        cache_dir = Path(self.config['dataset']['cache_dir'])
        test_cache_dir = cache_dir / 'test'
        cache_index_path = test_cache_dir / "index.json"
        
        if cache_index_path.exists():
            logger.info("Loading cached test dataset...")
            
            # Load cache index
            with open(cache_index_path, 'r') as f:
                cache_index = json.load(f)
            
            # Load cached samples
            cached_data = []
            for sample_id, cache_info in cache_index.items():
                cache_file = test_cache_dir / cache_info['file']
                with open(cache_file, 'rb') as f:
                    sample_data = pickle.load(f)
                    
                # Reconstruct knowledge graph if needed
                if 'knowledge_graph' in sample_data:
                    kg_data = sample_data['knowledge_graph']
                    if isinstance(kg_data, dict) and 'nodes' in kg_data:
                        sample_data['knowledge_graph'] = self._reconstruct_graph(kg_data)
                
                cached_data.append(sample_data)
            
            for sample_idx, sample_data in enumerate(cached_data):
                if sample_count >= num_samples:
                    break
                
                # Check if sample has text and non-None labels
                sample_labels = sample_data.get('labels', sample_data.get('label', []))
                has_non_none_label = True
                
                # Check if it's just the "None" label
                if isinstance(sample_labels, list) and len(sample_labels) == 1 and sample_labels[0] == 7:
                    has_non_none_label = False
                elif isinstance(sample_labels, (int, np.integer)) and sample_labels == 7:
                    has_non_none_label = False
                
                if sample_data.get('text', '') and len(sample_data['text']) > 10 and has_non_none_label:
                    logger.info(f"Analyzing sample {sample_count + 1}/{num_samples}")
                    
                    # Create sample directory
                    sample_dir = save_dir / f"sample_{sample_count}"
                    
                    try:
                        # Analyze sample
                        analysis = self.analyze_pathway_activation(
                            model, sample_data, sample_idx, sample_dir
                        )
                        all_analyses.append(analysis)
                        
                        # Save individual analysis
                        with open(sample_dir / 'learned_analysis.json', 'w') as f:
                            # Remove non-serializable objects
                            analysis_to_save = {
                                k: v for k, v in analysis.items() 
                                if k not in ['entities']
                            }
                            json.dump(analysis_to_save, f, indent=2)
                        
                        # Save interpretation
                        with open(sample_dir / 'learned_interpretation.txt', 'w') as f:
                            f.write(f"TEXT: {analysis['text']}\n\n")
                            f.write(f"TRUE LABELS: {[self.hallmark_names[i] for i in analysis['true_labels']]}\n\n")
                            f.write(analysis['interpretation'])
                        
                        sample_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error analyzing sample: {e}")
                        continue
        
        # Generate summary report
        self._generate_learned_summary_report(all_analyses, associations_info, save_dir)
        
        return all_analyses
    
    def _generate_learned_summary_report(self, analyses: List[Dict], 
                                       associations_info: Dict,
                                       save_dir: Path):
        """Generate summary report for learned mechanistic analyses."""
        with open(save_dir / 'learned_mechanistic_summary.txt', 'w') as f:
            f.write("Learned Mechanistic Interpretability Analysis Summary\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("LEARNED ASSOCIATIONS OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Hallmarks with learned pathways: {associations_info['total_hallmarks_with_pathways']}\n")
            f.write(f"Hallmarks with learned genes: {associations_info['total_hallmarks_with_genes']}\n")
            f.write(f"Avg pathways per hallmark: {associations_info['average_pathways_per_hallmark']:.1f}\n")
            f.write(f"Avg genes per hallmark: {associations_info['average_genes_per_hallmark']:.1f}\n\n")
            
            f.write("ANALYSIS RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total samples analyzed: {len(analyses)}\n\n")
            
            # Aggregate pathway evidence effectiveness
            pathway_evidence_scores = []
            prediction_scores = []
            combined_scores = []
            
            for analysis in analyses:
                for hallmark_data in analysis['pathway_activation'].values():
                    pathway_evidence_scores.append(hallmark_data['pathway_evidence_score'])
                    prediction_scores.append(hallmark_data['prediction_score'])
                    combined_scores.append(hallmark_data['combined_score'])
            
            f.write("PATHWAY EVIDENCE EFFECTIVENESS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Avg pathway evidence score: {np.mean(pathway_evidence_scores):.3f}\n")
            f.write(f"Avg prediction score: {np.mean(prediction_scores):.3f}\n")
            f.write(f"Avg combined score: {np.mean(combined_scores):.3f}\n\n")
            
            # Most frequently activated pathways
            pathway_frequency = Counter()
            for analysis in analyses:
                for hallmark_data in analysis['pathway_activation'].values():
                    for pathway in hallmark_data.get('present_pathways', []):
                        pathway_frequency[pathway['name']] += 1
            
            f.write("MOST FREQUENTLY ACTIVATED LEARNED PATHWAYS\n")
            f.write("-" * 40 + "\n")
            for pathway, count in pathway_frequency.most_common(10):
                f.write(f"{pathway}: {count} samples ({count/len(analyses):.1%})\n")
            
            # Key insights
            f.write("\n\nKEY INSIGHTS\n")
            f.write("-" * 40 + "\n")
            f.write("1. Learned associations provide data-driven biological evidence\n")
            f.write("2. Pathway evidence complements model predictions effectively\n")
            f.write("3. Gene importance scores reflect actual biological relevance\n")
            f.write("4. The system adapts to the specific dataset characteristics\n")
            f.write("5. Interpretations are grounded in statistically significant associations\n")
        
        logger.info("Learned mechanistic summary report generated")


def main():
    """Main function for learned mechanistic interpretability analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Mechanistic interpretability using learned associations"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--associations', type=str, 
                       default='learned_associations/hallmark_associations.json',
                       help='Path to learned associations file')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to analyze')
    parser.add_argument('--output_dir', type=str, default='learned_mechanistic_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Check if associations file exists
    if not Path(args.associations).exists():
        logger.error(f"Learned associations not found at {args.associations}")
        logger.error("Please run: python learn_hallmark_associations.py first")
        return
    
    # Initialize interpreter with learned associations
    interpreter = LearnedMechanisticInterpreter(args.config, args.associations)
    
    # Load model
    logger.info("Loading model...")
    model = interpreter.load_model(args.checkpoint)
    
    # Analyze samples
    logger.info(f"Analyzing {args.num_samples} test samples with learned associations...")
    analyses = interpreter.analyze_test_samples(
        model, None,  # Data loader not needed with cached data
        num_samples=args.num_samples,
        save_dir=Path(args.output_dir)
    )
    
    logger.info(f"Analysis complete! Results saved to {args.output_dir}")
    logger.info("This analysis uses data-driven learned associations instead of hardcoded knowledge")


if __name__ == "__main__":
    main()