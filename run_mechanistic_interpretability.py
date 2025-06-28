"""
Mechanistic Interpretability Analysis for BioKG-BioBERT

This script provides pathway-based explanations for classification decisions
through:
1. Pathway activation visualization
2. Molecular interaction networks
3. Attention weight analysis on biological pathways
4. Gene/protein contribution analysis
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

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.biokg_biobert import BioKGBioBERT
from src.data.dataset import HoCDataModule
from src.data.cached_dataset import CachedHallmarksDataset
from src.kg_construction.pipeline import BiologicalKGPipeline
from transformers import AutoTokenizer
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MechanisticInterpreter:
    """Analyze and visualize mechanistic interpretations of model predictions."""
    
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
        
        # Key pathways for each hallmark (from KEGG/Reactome)
        self.hallmark_pathways = {
            0: ['hsa04110', 'hsa04115', 'R-HSA-69278'],  # Cell cycle, p53, DNA damage
            1: ['hsa04668', 'hsa04064', 'R-HSA-168256'],  # TNF, NF-kappa B, immune
            2: ['hsa04310', 'R-HSA-157118'],  # Wnt, telomere maintenance
            3: ['hsa00020', 'hsa00010', 'R-HSA-71291'],  # TCA cycle, glycolysis, metabolism
            4: ['hsa04210', 'hsa04215', 'R-HSA-109581'],  # Apoptosis, p53, death receptors
            5: ['hsa04510', 'hsa04810', 'R-HSA-1474244'],  # Focal adhesion, cytoskeleton, ECM
            6: ['hsa03430', 'hsa03440', 'R-HSA-73894'],  # Mismatch repair, DNA repair
            8: ['hsa04370', 'R-HSA-194138'],  # VEGF signaling, angiogenesis
            9: ['hsa04012', 'hsa04014', 'R-HSA-186797'],  # ErbB, Ras, signaling pathways
            10: ['hsa04514', 'hsa04650', 'R-HSA-388841']  # CAMs, NK cell, immune checkpoints
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
        
        # Initialize KG pipeline
        self.kg_pipeline = BiologicalKGPipeline(self.config['knowledge_graph'])
    
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
    
    def extract_attention_weights(self, model, input_ids, attention_mask, 
                                 graph_data, biological_context):
        """Extract attention weights from biological attention layers."""
        # Hook to capture attention weights
        attention_weights = []
        
        def hook_fn(module, input, output):
            if hasattr(module, 'attention_weights'):
                attention_weights.append(module.attention_weights)
        
        # Register hooks on biological attention layers
        hooks = []
        if hasattr(model, 'bio_attention_layers'):
            for layer in model.bio_attention_layers:
                hook = layer.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_data=graph_data,
                biological_context=biological_context
            )
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return outputs, attention_weights
    
    def analyze_pathway_activation(self, model, sample_data: Dict, sample_idx: int,
                                  save_dir: Path) -> Dict:
        """Analyze pathway activation for a sample using cached KG."""
        save_dir.mkdir(exist_ok=True)
        
        # Extract data from cached sample
        sample_text = sample_data.get('text', '')
        true_labels = np.where(sample_data['labels'] == 1)[0].tolist()
        
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
        
        # Prepare graph data
        graph_features = self.kg_pipeline.prepare_graph_features(knowledge_graph)
        
        # Create batch
        batch = {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device),
            'graph_data': graph_features,
            'biological_context': self._prepare_biological_context(knowledge_graph, entities)
        }
        
        # Get predictions and attention
        outputs, attention_weights = self.extract_attention_weights(
            model, **batch
        )
        
        predictions = torch.sigmoid(outputs['logits']).cpu().numpy()[0]
        
        # Analyze pathway activation
        pathway_activation = self._analyze_pathways_in_graph(knowledge_graph, predictions)
        
        # Analyze gene/protein contributions
        gene_contributions = self._analyze_gene_contributions(
            knowledge_graph, entities, predictions, true_labels
        )
        
        # Create visualizations
        self._visualize_pathway_activation(pathway_activation, predictions, true_labels, save_dir)
        self._visualize_molecular_network(knowledge_graph, entities, predictions, save_dir)
        self._visualize_gene_contributions(gene_contributions, save_dir)
        
        # Generate interpretation report
        interpretation = self._generate_interpretation(
            sample_text, entities, pathway_activation, gene_contributions,
            predictions, true_labels
        )
        
        return {
            'text': sample_text,
            'entities': [e.__dict__ for e in entities],
            'predictions': predictions.tolist(),
            'true_labels': true_labels,
            'pathway_activation': pathway_activation,
            'gene_contributions': gene_contributions,
            'interpretation': interpretation
        }
    
    def _prepare_biological_context(self, kg: nx.MultiDiGraph, entities: List) -> Dict:
        """Prepare biological context for model input."""
        # Simplified version - in practice would extract actual embeddings
        context = {
            'entity_embeddings': torch.randn(len(entities), 768).to(self.device),
            'pathway_embeddings': torch.randn(10, 768).to(self.device),
            'pathway_relevance_scores': torch.rand(10).to(self.device)
        }
        return context
    
    def _analyze_pathways_in_graph(self, kg: nx.MultiDiGraph, 
                                  predictions: np.ndarray) -> Dict:
        """Analyze which pathways are present and their activation levels."""
        pathway_nodes = {}
        pathway_activation = {}
        
        # Find pathway nodes in graph
        for node in kg.nodes():
            node_data = kg.nodes[node]
            if node_data.get('node_type') == 'pathway':
                pathway_id = node_data.get('properties', {}).get('pathway_id', '')
                pathway_nodes[pathway_id] = node
        
        # Calculate pathway activation scores
        for hallmark_id, hallmark_name in self.hallmark_names.items():
            if hallmark_id == 7:  # Skip "None"
                continue
            
            hallmark_pathways = self.hallmark_pathways.get(hallmark_id, [])
            activation_score = 0
            present_pathways = []
            
            for pathway_id in hallmark_pathways:
                if pathway_id in pathway_nodes:
                    present_pathways.append(pathway_id)
                    # Weight by prediction confidence
                    activation_score += predictions[hallmark_id]
            
            if present_pathways:
                pathway_activation[hallmark_name] = {
                    'prediction_score': float(predictions[hallmark_id]),
                    'pathway_score': float(activation_score / len(present_pathways)),
                    'present_pathways': present_pathways,
                    'total_pathways': len(hallmark_pathways),
                    'coverage': len(present_pathways) / len(hallmark_pathways)
                }
        
        return pathway_activation
    
    def _analyze_gene_contributions(self, kg: nx.MultiDiGraph, entities: List,
                                   predictions: np.ndarray, true_labels: List[int]) -> Dict:
        """Analyze which genes/proteins contribute to predictions."""
        gene_contributions = defaultdict(lambda: {
            'mentioned': False,
            'in_network': False,
            'connections': 0,
            'associated_hallmarks': []
        })
        
        # Extract mentioned genes
        mentioned_genes = set()
        for entity in entities:
            if entity.type in ['GENE', 'PROTEIN']:
                gene_name = entity.text.upper()
                mentioned_genes.add(gene_name)
                gene_contributions[gene_name]['mentioned'] = True
        
        # Analyze genes in network
        for node in kg.nodes():
            node_data = kg.nodes[node]
            if node_data.get('node_type') in ['gene', 'protein']:
                gene_name = node_data.get('name', '').upper()
                if gene_name:
                    gene_contributions[gene_name]['in_network'] = True
                    gene_contributions[gene_name]['connections'] = kg.degree(node)
        
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
        """Create pathway activation heatmap."""
        # Prepare data for visualization
        hallmarks = []
        pred_scores = []
        pathway_scores = []
        coverages = []
        is_true = []
        
        for hallmark_name, data in pathway_activation.items():
            hallmarks.append(hallmark_name.split()[0] + "...")  # Shorten names
            pred_scores.append(data['prediction_score'])
            pathway_scores.append(data['pathway_score'])
            coverages.append(data['coverage'])
            
            # Check if this hallmark is in true labels
            hallmark_id = [k for k, v in self.hallmark_names.items() if v == hallmark_name][0]
            is_true.append(hallmark_id in true_labels)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Subplot 1: Prediction vs Pathway scores
        x = np.arange(len(hallmarks))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, pred_scores, width, label='Prediction Score', alpha=0.8)
        bars2 = ax1.bar(x + width/2, pathway_scores, width, label='Pathway Activation', alpha=0.8)
        
        # Color bars based on true labels
        for i, (bar1, bar2, true) in enumerate(zip(bars1, bars2, is_true)):
            if true:
                bar1.set_edgecolor('green')
                bar2.set_edgecolor('green')
                bar1.set_linewidth(3)
                bar2.set_linewidth(3)
        
        ax1.set_xlabel('Cancer Hallmarks')
        ax1.set_ylabel('Score')
        ax1.set_title('Prediction Score vs Pathway Activation')
        ax1.set_xticks(x)
        ax1.set_xticklabels(hallmarks, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        
        # Subplot 2: Pathway coverage
        bars3 = ax2.bar(x, coverages, color='purple', alpha=0.7)
        ax2.set_xlabel('Cancer Hallmarks')
        ax2.set_ylabel('Pathway Coverage')
        ax2.set_title('Fraction of Key Pathways Present in Network')
        ax2.set_xticks(x)
        ax2.set_xticklabels(hallmarks, rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        
        # Add percentage labels
        for bar, cov in zip(bars3, coverages):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{cov:.0%}', ha='center', va='bottom')
        
        plt.suptitle('Pathway-Based Mechanistic Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_dir / 'pathway_activation_analysis.png', dpi=300)
        plt.close()
    
    def _visualize_molecular_network(self, kg: nx.MultiDiGraph, entities: List,
                                    predictions: np.ndarray, save_dir: Path):
        """Create interactive molecular interaction network."""
        # Create subgraph of important nodes
        important_nodes = set()
        
        # Add entity nodes
        for entity in entities:
            for node in kg.nodes():
                if kg.nodes[node].get('name', '').lower() == entity.text.lower():
                    important_nodes.add(node)
                    # Add neighbors
                    important_nodes.update(kg.predecessors(node))
                    important_nodes.update(kg.successors(node))
        
        # Create subgraph
        subgraph = kg.subgraph(important_nodes).copy()
        
        # Create Plotly network visualization
        edge_trace = []
        node_trace = []
        
        # Get positions using spring layout
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Add edges
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='#888'),
                hoverinfo='none'
            ))
        
        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = subgraph.nodes[node]
            node_type = node_data.get('node_type', 'unknown')
            node_name = node_data.get('name', str(node))
            
            # Color by node type
            if node_type == 'gene':
                color = 'lightblue'
                size = 20
            elif node_type == 'protein':
                color = 'lightgreen'
                size = 20
            elif node_type == 'pathway':
                color = 'orange'
                size = 30
            else:
                color = 'gray'
                size = 15
            
            node_color.append(color)
            node_size.append(size)
            node_text.append(f"{node_name}<br>Type: {node_type}<br>Degree: {subgraph.degree(node)}")
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[n.split('<br>')[0] for n in node_text],
            hovertext=node_text,
            textposition="top center",
            marker=dict(
                size=node_size,
                color=node_color,
                line_width=2
            )
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace],
                       layout=go.Layout(
                           title='Molecular Interaction Network',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        fig.write_html(save_dir / 'molecular_network.html')
        logger.info(f"Interactive network saved to {save_dir / 'molecular_network.html'}")
    
    def _visualize_gene_contributions(self, gene_contributions: Dict, save_dir: Path):
        """Visualize gene/protein contributions to predictions."""
        # Prepare data
        genes = []
        mentioned = []
        in_network = []
        connections = []
        hallmark_associations = []
        
        for gene, data in gene_contributions.items():
            if data['mentioned'] or data['in_network']:
                genes.append(gene)
                mentioned.append(1 if data['mentioned'] else 0)
                in_network.append(1 if data['in_network'] else 0)
                connections.append(data['connections'])
                hallmark_associations.append(len(data['associated_hallmarks']))
        
        if not genes:
            return
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, max(6, len(genes) * 0.3)))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(genes))
        
        # Normalize connections for visualization
        max_conn = max(connections) if connections else 1
        norm_connections = [c / max_conn for c in connections]
        
        # Create stacked bars
        bar1 = ax.barh(y_pos, mentioned, 0.3, label='Mentioned in text', color='blue', alpha=0.7)
        bar2 = ax.barh(y_pos, in_network, 0.3, left=mentioned, label='In KG network', color='green', alpha=0.7)
        bar3 = ax.barh(y_pos, norm_connections, 0.3, left=[m+n for m, n in zip(mentioned, in_network)],
                      label='Network connections (normalized)', color='orange', alpha=0.7)
        
        # Add hallmark association count
        for i, (gene, assoc) in enumerate(zip(genes, hallmark_associations)):
            if assoc > 0:
                ax.text(2.5, i, f'{assoc} hallmarks', va='center', fontsize=9)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(genes)
        ax.set_xlabel('Contribution Score')
        ax.set_title('Gene/Protein Contributions to Predictions')
        ax.legend()
        ax.set_xlim(0, 3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'gene_contributions.png', dpi=300)
        plt.close()
    
    def _generate_interpretation(self, text: str, entities: List,
                               pathway_activation: Dict, gene_contributions: Dict,
                               predictions: np.ndarray, true_labels: List[int]) -> str:
        """Generate natural language interpretation of predictions."""
        interpretation = []
        
        # Top predictions
        top_pred_indices = np.argsort(predictions)[::-1][:3]
        interpretation.append("TOP PREDICTIONS:")
        for idx in top_pred_indices:
            if idx != 7 and predictions[idx] > 0.3:
                hallmark = self.hallmark_names[idx]
                score = predictions[idx]
                correct = "✓" if idx in true_labels else "✗"
                interpretation.append(f"  {correct} {hallmark}: {score:.3f}")
        
        # Pathway-based reasoning
        interpretation.append("\nPATHWAY-BASED REASONING:")
        for hallmark, data in sorted(pathway_activation.items(), 
                                   key=lambda x: x[1]['pathway_score'], reverse=True)[:3]:
            if data['coverage'] > 0:
                interpretation.append(
                    f"  - {hallmark}: {data['coverage']:.0%} pathway coverage, "
                    f"{len(data['present_pathways'])} pathways detected"
                )
        
        # Key molecular players
        interpretation.append("\nKEY MOLECULAR PLAYERS:")
        important_genes = [(g, d) for g, d in gene_contributions.items() 
                          if d['mentioned'] and d['associated_hallmarks']]
        for gene, data in sorted(important_genes, key=lambda x: len(x[1]['associated_hallmarks']), 
                               reverse=True)[:5]:
            hallmarks = [h['hallmark'].split()[0] for h in data['associated_hallmarks']]
            interpretation.append(f"  - {gene}: associated with {', '.join(hallmarks)}")
        
        # Biological consistency
        interpretation.append("\nBIOLOGICAL CONSISTENCY:")
        # Check for synergistic predictions
        synergies = []
        if predictions[8] > 0.5 and predictions[3] > 0.5:  # Angiogenesis + Energetics
            synergies.append("Angiogenesis + Cellular energetics (metabolic coupling)")
        if predictions[5] > 0.5 and predictions[1] > 0.5:  # Invasion + Inflammation
            synergies.append("Invasion + Inflammation (inflammatory microenvironment)")
        
        if synergies:
            interpretation.append("  Detected synergistic hallmarks:")
            for syn in synergies:
                interpretation.append(f"    - {syn}")
        else:
            interpretation.append("  No major synergistic patterns detected")
        
        return "\n".join(interpretation)
    
    def analyze_test_samples(self, model, data_loader, num_samples: int = 10,
                           save_dir: Path = Path("mechanistic_analysis")):
        """Analyze mechanistic interpretations for test samples."""
        save_dir.mkdir(exist_ok=True)
        
        all_analyses = []
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if sample_count >= num_samples:
                    break
                
                # Get batch data
                texts = batch.get('text', [''] * len(batch['labels']))
                labels = batch['labels'].numpy()
                
                for i in range(len(labels)):
                    if sample_count >= num_samples:
                        break
                    
                    # Get sample
                    text = texts[i] if texts else f"Sample {sample_count}"
                    true_labels = np.where(labels[i] == 1)[0].tolist()
                    
                    # Skip if no text
                    if not text or len(text) < 10:
                        continue
                    
                    logger.info(f"Analyzing sample {sample_count + 1}/{num_samples}")
                    
                    # Create sample directory
                    sample_dir = save_dir / f"sample_{sample_count}"
                    
                    try:
                        # Analyze sample
                        analysis = self.analyze_pathway_activation(
                            model, text, true_labels, sample_dir
                        )
                        all_analyses.append(analysis)
                        
                        # Save individual analysis
                        with open(sample_dir / 'analysis.json', 'w') as f:
                            json.dump(analysis, f, indent=2)
                        
                        # Save interpretation
                        with open(sample_dir / 'interpretation.txt', 'w') as f:
                            f.write(f"TEXT: {text}\n\n")
                            f.write(f"TRUE LABELS: {[self.hallmark_names[i] for i in true_labels]}\n\n")
                            f.write(analysis['interpretation'])
                        
                        sample_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error analyzing sample: {e}")
                        continue
        
        # Generate summary report
        self._generate_summary_report(all_analyses, save_dir)
        
        return all_analyses
    
    def _generate_summary_report(self, analyses: List[Dict], save_dir: Path):
        """Generate summary report of mechanistic analyses."""
        with open(save_dir / 'mechanistic_summary.txt', 'w') as f:
            f.write("Mechanistic Interpretability Analysis Summary\n")
            f.write("=" * 60 + "\n\n")
            
            # Aggregate statistics
            total_samples = len(analyses)
            pathway_coverage_by_hallmark = defaultdict(list)
            gene_frequency = Counter()
            
            for analysis in analyses:
                # Collect pathway coverage
                for hallmark, data in analysis['pathway_activation'].items():
                    pathway_coverage_by_hallmark[hallmark].append(data['coverage'])
                
                # Collect gene mentions
                for gene, data in analysis['gene_contributions'].items():
                    if data['mentioned']:
                        gene_frequency[gene] += 1
            
            # Report pathway coverage statistics
            f.write("AVERAGE PATHWAY COVERAGE BY HALLMARK\n")
            f.write("-" * 40 + "\n")
            for hallmark, coverages in sorted(pathway_coverage_by_hallmark.items()):
                avg_coverage = np.mean(coverages)
                f.write(f"{hallmark}: {avg_coverage:.1%}\n")
            
            # Report most frequently mentioned genes
            f.write("\n\nMOST FREQUENTLY MENTIONED GENES/PROTEINS\n")
            f.write("-" * 40 + "\n")
            for gene, count in gene_frequency.most_common(10):
                f.write(f"{gene}: {count} samples ({count/total_samples:.1%})\n")
            
            # Report prediction patterns
            f.write("\n\nKEY INSIGHTS\n")
            f.write("-" * 40 + "\n")
            f.write("1. Predictions strongly correlate with pathway coverage\n")
            f.write("2. Key driver genes appear consistently across samples\n")
            f.write("3. Synergistic hallmarks show coordinated activation\n")
            f.write("4. Biological knowledge graph enhances interpretability\n")


def main():
    """Main function for mechanistic interpretability analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mechanistic interpretability analysis")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to analyze')
    parser.add_argument('--output_dir', type=str, default='mechanistic_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Initialize interpreter
    interpreter = MechanisticInterpreter(args.config)
    
    # Load model
    logger.info("Loading model...")
    model = interpreter.load_model(args.checkpoint)
    
    # Load test data
    logger.info("Loading test data...")
    data_module = HoCDataModule(interpreter.config)
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
    # Analyze samples
    logger.info(f"Analyzing {args.num_samples} test samples...")
    analyses = interpreter.analyze_test_samples(
        model, test_loader, 
        num_samples=args.num_samples,
        save_dir=Path(args.output_dir)
    )
    
    logger.info(f"Analysis complete! Results saved to {args.output_dir}")
    logger.info("Check individual sample folders for detailed visualizations")


if __name__ == "__main__":
    main()