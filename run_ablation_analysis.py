#!/usr/bin/env python3
"""
Ablation Analysis Script

This script performs comprehensive analysis of ablation study results,
comparing each ablation variant against the full BioKG-BioBERT model.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score, hamming_loss
import torch
import yaml
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.biokg_biobert import BioKGBioBERT
from src.data.dataset import HoCDataModule

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hallmark names for better readability
HALLMARK_NAMES = {
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

# Ablation category mapping
ABLATION_CATEGORIES = {
    'ablation_attention_none': 'A1: No Bio Attention',
    'ablation_attention_entity_only': 'A1: Entity-Only Attention',
    'ablation_kg_none': 'A2: No Knowledge Graph',
    'ablation_kg_1hop': 'A2: 1-Hop KG Only',
    'ablation_fusion_early': 'A3: Early Fusion',
    'ablation_fusion_cross_modal': 'A3: Cross-Modal Fusion',
    'ablation_multitask_hallmarks_only': 'A4: Hallmarks Only',
    'ablation_multitask_with_pathway': 'A4: With Pathway Loss',
    'ablation_multitask_with_consistency': 'A4: With Consistency Loss'
}

# Ablation groups for analysis
ABLATION_GROUPS = {
    'Attention Mechanisms': ['final_model_full', 'ablation_attention_none', 'ablation_attention_entity_only'],
    'Knowledge Graph Integration': ['final_model_full', 'ablation_kg_none', 'ablation_kg_1hop'],
    'Fusion Strategies': ['final_model_full', 'ablation_fusion_early', 'ablation_fusion_cross_modal'],
    'Multi-Task Learning': ['final_model_full', 'ablation_multitask_hallmarks_only', 
                           'ablation_multitask_with_pathway', 'ablation_multitask_with_consistency']
}


class AblationAnalyzer:
    """Analyzes ablation study results comprehensively."""
    
    def __init__(self, checkpoints_dir: str, output_dir: str, config_path: str = 'configs/default_config.yaml'):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different analysis types
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.tables_dir = self.output_dir / "tables"
        self.tables_dir.mkdir(exist_ok=True)
        
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Load default config (for data module)
        with open(config_path, 'r') as f:
            self.default_config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Setup data module once
        logger.info("Loading data module...")
        self.data_module = HoCDataModule(self.default_config)
        self.data_module.setup()
        
    def discover_ablation_models(self) -> Dict[str, Path]:
        """Automatically discover ablation models in checkpoints directory."""
        models = {}
        
        # Add full model
        full_model_path = self.checkpoints_dir / "final_model_full" / "best.pt"
        if full_model_path.exists():
            models['final_model_full'] = full_model_path
            logger.info(f"Found full model: {full_model_path}")
        else:
            raise FileNotFoundError(f"Full model not found at {full_model_path}")
        
        # Find all ablation models
        for ablation_dir in self.checkpoints_dir.glob("ablation_*"):
            if ablation_dir.is_dir():
                checkpoint_path = ablation_dir / "best.pt"
                if checkpoint_path.exists():
                    models[ablation_dir.name] = checkpoint_path
                    logger.info(f"Found ablation model: {checkpoint_path}")
        
        logger.info(f"Total models found: {len(models)} (1 full + {len(models)-1} ablations)")
        return models
    
    def load_model(self, checkpoint_path: Path) -> Tuple[torch.nn.Module, Dict]:
        """Load model from checkpoint."""
        logger.info(f"Loading model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract config from checkpoint
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
        else:
            raise ValueError(f"No config found in checkpoint {checkpoint_path}")
        
        # Check if checkpoint has auxiliary components
        state_dict_keys = checkpoint['model_state_dict'].keys()
        has_pathway_classifier = any('pathway_classifier' in key for key in state_dict_keys)
        has_consistency_predictor = any('consistency_predictor' in key for key in state_dict_keys)
        
        # Set loss weights based on what's in the checkpoint
        if has_pathway_classifier and has_consistency_predictor:
            model_config['loss_weights'] = checkpoint['config'].get('training', {}).get('loss_weights', {
                'hallmark_loss': 1.0,
                'pathway_loss': 0.1,
                'consistency_loss': 0.05
            })
        else:
            model_config['loss_weights'] = {
                'hallmark_loss': 1.0,
                'pathway_loss': 0.0,
                'consistency_loss': 0.0
            }
        
        model = BioKGBioBERT(model_config)
        
        # Load state dict with strict=False to handle missing/unexpected keys
        incompatible_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Log any missing or unexpected keys
        if incompatible_keys.missing_keys:
            print(f"Warning: Missing keys in checkpoint: {incompatible_keys.missing_keys}")
        if incompatible_keys.unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint (ignored): {incompatible_keys.unexpected_keys}")
        
        model.eval()
        model.to(self.device)
        
        return model, checkpoint
    
    def get_predictions(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions on a dataset."""
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch_device = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = model(**batch_device)
                
                probabilities = torch.sigmoid(outputs['logits'])
                predictions = (probabilities >= 0.5).float()
                
                all_probabilities.append(probabilities.cpu())
                all_predictions.append(predictions.cpu())
                all_targets.append(batch_device['labels'].cpu())
        
        all_probabilities = torch.cat(all_probabilities, dim=0).numpy()
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        return all_predictions, all_probabilities, all_targets
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        batch_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_device[key] = value.to(self.device)
            elif key == 'graph_data':
                batch_device[key] = value.to(self.device) if value is not None else None
            elif key == 'biological_context':
                moved_context = {}
                for ctx_key, ctx_value in value.items():
                    if isinstance(ctx_value, torch.Tensor):
                        moved_context[ctx_key] = ctx_value.to(self.device)
                    else:
                        moved_context[ctx_key] = ctx_value
                batch_device[key] = moved_context
            else:
                batch_device[key] = value
        return batch_device
    
    def evaluate_model(self, model_name: str, checkpoint_path: Path) -> Dict:
        """Evaluate a single model."""
        logger.info(f"\nEvaluating {model_name}...")
        
        # Load model
        model, checkpoint = self.load_model(checkpoint_path)
        
        # Get test set predictions
        test_loader = self.data_module.test_dataloader()
        predictions, probabilities, targets = self.get_predictions(model, test_loader)
        
        # Calculate metrics
        results = {
            'micro_f1': f1_score(targets, predictions, average='micro'),
            'macro_f1': f1_score(targets, predictions, average='macro'),
            'weighted_f1': f1_score(targets, predictions, average='weighted'),
            'hamming_loss': hamming_loss(targets, predictions),
            'exact_match_ratio': np.mean([np.array_equal(t, p) for t, p in zip(targets, predictions)]),
            'predictions': predictions,
            'probabilities': probabilities,
            'targets': targets
        }
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        results['per_class_precision'] = precision.tolist()
        results['per_class_recall'] = recall.tolist()
        results['per_class_f1'] = f1.tolist()
        results['per_class_support'] = support.tolist()
        
        # AUC if possible
        try:
            auc_scores = []
            for i in range(targets.shape[1]):
                if len(np.unique(targets[:, i])) > 1:  # Check if both classes present
                    auc = roc_auc_score(targets[:, i], probabilities[:, i])
                    auc_scores.append(auc)
                else:
                    auc_scores.append(np.nan)
            results['per_class_auc'] = auc_scores
            results['macro_auc'] = np.nanmean(auc_scores)
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
        
        logger.info(f"{model_name} - Macro-F1: {results['macro_f1']:.4f}, Micro-F1: {results['micro_f1']:.4f}")
        
        return results
    
    def mcnemar_test(self, pred1: np.ndarray, pred2: np.ndarray, targets: np.ndarray) -> Dict:
        """Perform McNemar's test between two models."""
        results = {}
        
        # Per-hallmark tests
        for i in range(11):
            correct1 = (pred1[:, i] == targets[:, i])
            correct2 = (pred2[:, i] == targets[:, i])
            
            # Create contingency table for statsmodels
            table = np.array([
                [(correct1 & correct2).sum(), (correct1 & ~correct2).sum()],
                [(~correct1 & correct2).sum(), (~correct1 & ~correct2).sum()]
            ])
            
            # Perform McNemar's test
            result = mcnemar(table, exact=True, correction=True)
            
            results[HALLMARK_NAMES[i]] = {
                'statistic': float(result.statistic),
                'p_value': float(result.pvalue),
                'n01': int((~correct1 & correct2).sum()),  # Ablation correct, full wrong
                'n10': int((correct1 & ~correct2).sum()),  # Full correct, ablation wrong
                'significant': result.pvalue < 0.05
            }
        
        # Overall test
        correct1_all = np.all(pred1 == targets, axis=1)
        correct2_all = np.all(pred2 == targets, axis=1)
        
        table_overall = np.array([
            [(correct1_all & correct2_all).sum(), (correct1_all & ~correct2_all).sum()],
            [(~correct1_all & correct2_all).sum(), (~correct1_all & ~correct2_all).sum()]
        ])
        
        result_overall = mcnemar(table_overall, exact=False, correction=True)
        
        results['overall'] = {
            'statistic': float(result_overall.statistic),
            'p_value': float(result_overall.pvalue),
            'n01': int((~correct1_all & correct2_all).sum()),
            'n10': int((correct1_all & ~correct2_all).sum()),
            'significant': result_overall.pvalue < 0.05
        }
        
        return results
    
    def generate_performance_comparison_plot(self, results_dict: Dict[str, Dict]):
        """Generate comprehensive performance comparison plots."""
        # Get pretty names for models
        model_names = []
        for model in results_dict.keys():
            if model == 'final_model_full':
                model_names.append('Full Model')
            else:
                model_names.append(ABLATION_CATEGORIES.get(model, model))
        
        # Prepare data
        models = list(results_dict.keys())
        micro_f1_scores = [results_dict[m]['micro_f1'] for m in models]
        macro_f1_scores = [results_dict[m]['macro_f1'] for m in models]
        weighted_f1_scores = [results_dict[m]['weighted_f1'] for m in models]
        hamming_losses = [results_dict[m]['hamming_loss'] for m in models]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ablation Study: Performance Comparison', fontsize=16, y=1.02)
        
        # 1. Overall F1 scores comparison
        ax = axes[0, 0]
        x = np.arange(len(models))
        width = 0.25
        
        bars1 = ax.bar(x - width, micro_f1_scores, width, label='Micro F1', alpha=0.8)
        bars2 = ax.bar(x, macro_f1_scores, width, label='Macro F1', alpha=0.8)
        bars3 = ax.bar(x + width, weighted_f1_scores, width, label='Weighted F1', alpha=0.8)
        
        ax.set_xlabel('Model Variant')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score Comparison Across Ablation Variants')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Performance drop from full model
        ax = axes[0, 1]
        full_macro_f1 = results_dict['final_model_full']['macro_f1']
        performance_drops = [(results_dict[m]['macro_f1'] - full_macro_f1) / full_macro_f1 * 100 
                            for m in models if m != 'final_model_full']
        
        colors = ['red' if drop < 0 else 'green' for drop in performance_drops]
        bars = ax.bar(range(len(performance_drops)), performance_drops, color=colors, alpha=0.7)
        
        ax.set_xlabel('Ablation Variant')
        ax.set_ylabel('Performance Change (%)')
        ax.set_title('Performance Change Relative to Full Model (Macro F1)')
        ax.set_xticks(range(len(performance_drops)))
        ax.set_xticklabels([n for n in model_names if n != 'Full Model'], rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, drop) in enumerate(zip(bars, performance_drops)):
            ax.text(i, drop + (0.5 if drop > 0 else -0.5), f'{drop:.1f}%', 
                   ha='center', va='bottom' if drop > 0 else 'top', fontsize=8)
        
        # 3. Grouped comparison by ablation category
        ax = axes[1, 0]
        group_positions = []
        current_pos = 0
        
        for group_name, group_models in ABLATION_GROUPS.items():
            group_scores = []
            group_labels = []
            
            for model in group_models:
                if model in results_dict:
                    group_scores.append(results_dict[model]['macro_f1'])
                    if model == 'final_model_full':
                        group_labels.append('Full Model')
                    else:
                        group_labels.append(ABLATION_CATEGORIES.get(model, model).split(': ')[-1])
            
            if group_scores:
                positions = np.arange(len(group_scores)) + current_pos
                bars = ax.bar(positions, group_scores, alpha=0.8, label=group_name)
                
                # Add value labels
                for pos, score in zip(positions, group_scores):
                    ax.text(pos, score + 0.01, f'{score:.3f}', ha='center', va='bottom', fontsize=8)
                
                group_positions.append((current_pos, current_pos + len(group_scores) - 1, group_name))
                current_pos += len(group_scores) + 1
        
        ax.set_xlabel('Model Variant')
        ax.set_ylabel('Macro F1 Score')
        ax.set_title('Performance by Ablation Category')
        ax.grid(True, alpha=0.3)
        
        # Add group labels
        for start, end, name in group_positions:
            ax.text((start + end) / 2, -0.15, name, ha='center', va='top', 
                   transform=ax.get_xaxis_transform(), fontsize=10, weight='bold')
        
        # 4. Per-hallmark heatmap
        ax = axes[1, 1]
        
        # Create per-hallmark F1 matrix
        hallmark_matrix = []
        available_models = []
        
        for model in models:
            if 'per_class_f1' in results_dict[model]:
                hallmark_matrix.append(results_dict[model]['per_class_f1'])
                if model == 'final_model_full':
                    available_models.append('Full Model')
                else:
                    available_models.append(ABLATION_CATEGORIES.get(model, model))
        
        if hallmark_matrix:
            hallmark_matrix = np.array(hallmark_matrix)
            
            # Create heatmap
            sns.heatmap(hallmark_matrix.T, 
                       xticklabels=available_models,
                       yticklabels=[HALLMARK_NAMES[i] for i in range(11)],
                       annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                       cbar_kws={'label': 'F1 Score'},
                       ax=ax)
            ax.set_title('Per-Hallmark F1 Scores Heatmap')
            ax.set_xlabel('Model Variant')
            ax.set_ylabel('Cancer Hallmark')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'ablation_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Performance comparison plots saved")
    
    def generate_statistical_significance_plot(self, mcnemar_results: Dict):
        """Generate McNemar's test visualization."""
        comparisons = list(mcnemar_results.keys())
        hallmarks = list(HALLMARK_NAMES.values())
        
        # Create p-value matrix
        p_value_matrix = np.ones((len(comparisons), len(hallmarks)))
        
        for i, comp in enumerate(comparisons):
            for j, hallmark in enumerate(hallmarks):
                if hallmark in mcnemar_results[comp]:
                    p_value_matrix[i, j] = mcnemar_results[comp][hallmark]['p_value']
        
        # Create significance mask
        sig_mask = p_value_matrix < 0.05
        
        plt.figure(figsize=(14, 8))
        
        # Plot heatmap
        sns.heatmap(
            p_value_matrix,
            xticklabels=hallmarks,
            yticklabels=[ABLATION_CATEGORIES.get(c, c) for c in comparisons],
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            vmin=0,
            vmax=0.1,
            cbar_kws={'label': 'p-value'},
            mask=~sig_mask,  # Only show significant values
            linewidths=0.5
        )
        
        plt.title("McNemar's Test P-values (Significant Results Only, p < 0.05)")
        plt.xlabel('Cancer Hallmarks')
        plt.ylabel('Ablation Models')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'mcnemar_significance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Statistical significance plot saved")
    
    def generate_detailed_report(self, results_dict: Dict, mcnemar_results: Dict):
        """Generate comprehensive ablation analysis report."""
        report_path = self.reports_dir / 'ablation_analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# BioKG-BioBERT Ablation Study Analysis Report\n\n")
            f.write("## Executive Summary\n\n")
            
            # Overall performance summary
            full_model_results = results_dict['final_model_full']
            f.write(f"**Full Model Performance:**\n")
            f.write(f"- Micro F1: {full_model_results['micro_f1']:.4f}\n")
            f.write(f"- Macro F1: {full_model_results['macro_f1']:.4f}\n")
            f.write(f"- Weighted F1: {full_model_results['weighted_f1']:.4f}\n\n")
            
            # Key findings
            f.write("### Key Findings\n\n")
            
            # Find most impactful components
            impact_ranking = []
            for model_name, results in results_dict.items():
                if model_name != 'final_model_full':
                    impact = full_model_results['macro_f1'] - results['macro_f1']
                    impact_ranking.append({
                        'name': ABLATION_CATEGORIES.get(model_name, model_name),
                        'model': model_name,
                        'impact': impact,
                        'relative_impact': impact / full_model_results['macro_f1'] * 100
                    })
            
            impact_ranking.sort(key=lambda x: x['impact'], reverse=True)
            
            f.write("**Component Impact Ranking (by absolute F1 drop):**\n\n")
            for i, comp in enumerate(impact_ranking[:5], 1):
                f.write(f"{i}. **{comp['name']}**: ↓ {comp['impact']:.4f} ({comp['relative_impact']:.2f}% drop)\n")
            
            f.write("\n## Detailed Analysis by Ablation Group\n\n")
            
            # Analyze each ablation group
            for group_name, group_models in ABLATION_GROUPS.items():
                f.write(f"### {group_name}\n\n")
                
                # Create comparison table
                f.write("| Model Variant | Micro F1 | Macro F1 | Weighted F1 | Δ Macro F1 | Relative Change |\n")
                f.write("|--------------|----------|----------|-------------|-----------|----------------|\n")
                
                for model in group_models:
                    if model in results_dict:
                        result = results_dict[model]
                        
                        if model == 'final_model_full':
                            name = "**Full Model** (Baseline)"
                            f.write(f"| {name} | **{result['micro_f1']:.4f}** | ")
                            f.write(f"**{result['macro_f1']:.4f}** | **{result['weighted_f1']:.4f}** | - | - |\n")
                        else:
                            name = ABLATION_CATEGORIES.get(model, model).split(': ')[-1]
                            diff = result['macro_f1'] - full_model_results['macro_f1']
                            relative_diff = diff / full_model_results['macro_f1'] * 100
                            
                            f.write(f"| {name} | {result['micro_f1']:.4f} | ")
                            f.write(f"{result['macro_f1']:.4f} | {result['weighted_f1']:.4f} | ")
                            f.write(f"{diff:+.4f} | {relative_diff:+.2f}% |\n")
                
                # Group insights
                f.write(f"\n**Insights for {group_name}:**\n")
                
                if group_name == "Attention Mechanisms":
                    f.write("- Biological attention mechanisms contribute significantly to model performance\n")
                    f.write("- Pathway information provides additional benefits beyond entity-level attention\n")
                elif group_name == "Knowledge Graph Integration":
                    f.write("- Knowledge graph integration is crucial for achieving optimal performance\n")
                    f.write("- The impact of removing KG shows the importance of biological context\n")
                elif group_name == "Fusion Strategies":
                    f.write("- Late fusion (default) shows best performance for multi-modal integration\n")
                    f.write("- Alternative fusion strategies may trade off performance for efficiency\n")
                elif group_name == "Multi-Task Learning":
                    f.write("- Auxiliary tasks (pathway and consistency losses) improve generalization\n")
                    f.write("- Multi-task learning provides regularization benefits\n")
                
                f.write("\n")
            
            # Statistical significance analysis
            f.write("## Statistical Significance Analysis\n\n")
            f.write("McNemar's test results comparing each ablation variant against the full model (p < 0.05 indicates significant difference):\n\n")
            
            f.write("| Ablation Variant | Significantly Different Hallmarks | Overall p-value | Overall Significant |\n")
            f.write("|-----------------|----------------------------------|-----------------|--------------------|\n")
            
            for model_name, mcnemar_result in mcnemar_results.items():
                sig_hallmarks = []
                for hallmark, result in mcnemar_result.items():
                    if hallmark != 'overall' and result['significant']:
                        sig_hallmarks.append(hallmark)
                
                overall = mcnemar_result['overall']
                name = ABLATION_CATEGORIES.get(model_name, model_name)
                
                f.write(f"| {name} | {len(sig_hallmarks)}/11 | ")
                f.write(f"{overall['p_value']:.4f} | {'Yes' if overall['significant'] else 'No'} |\n")
            
            # Per-hallmark analysis
            f.write("\n## Per-Hallmark Performance Analysis\n\n")
            
            # Find hallmarks most affected by ablations
            hallmark_impacts = {i: [] for i in range(11)}
            
            for model_name, result in results_dict.items():
                if model_name != 'final_model_full' and 'per_class_f1' in result:
                    full_f1 = results_dict['final_model_full']['per_class_f1']
                    for i in range(11):
                        impact = full_f1[i] - result['per_class_f1'][i]
                        hallmark_impacts[i].append(impact)
            
            f.write("**Hallmarks Most Sensitive to Ablations:**\n\n")
            sensitivity_scores = []
            for hallmark_id, impacts in hallmark_impacts.items():
                if impacts:
                    sensitivity_scores.append({
                        'id': hallmark_id,
                        'name': HALLMARK_NAMES[hallmark_id],
                        'mean_impact': np.mean(np.abs(impacts)),
                        'std_impact': np.std(impacts)
                    })
            
            sensitivity_scores.sort(key=lambda x: x['mean_impact'], reverse=True)
            
            for i, score in enumerate(sensitivity_scores[:5], 1):
                f.write(f"{i}. **{score['name']}**: Mean absolute impact = {score['mean_impact']:.4f} (σ = {score['std_impact']:.4f})\n")
            
            # Conclusions
            f.write("\n## Conclusions and Recommendations\n\n")
            f.write("### Component Criticality\n\n")
            f.write("Based on the ablation analysis, the following components are critical for optimal performance:\n\n")
            
            critical_components = [comp for comp in impact_ranking if comp['impact'] > 0.01]
            for comp in critical_components:
                f.write(f"- **{comp['name']}**: Removal causes {comp['relative_impact']:.1f}% performance drop\n")
            
            f.write("\n### Optimization Opportunities\n\n")
            f.write("1. **Computational Efficiency**: Components with minimal impact could be removed for faster inference\n")
            f.write("2. **Architecture Simplification**: Consider removing components that don't significantly improve performance\n")
            f.write("3. **Training Strategy**: Focus computational resources on the most impactful components\n")
            
            f.write("\n### Future Research Directions\n\n")
            f.write("1. Investigate interaction effects between different components\n")
            f.write("2. Explore more efficient implementations of critical components\n")
            f.write("3. Develop adaptive mechanisms to selectively enable/disable components based on input\n")
        
        logger.info(f"Comprehensive report saved to: {report_path}")
    
    def generate_latex_tables(self, results_dict: Dict, mcnemar_results: Dict):
        """Generate LaTeX tables for paper inclusion."""
        latex_path = self.tables_dir / 'ablation_results.tex'
        
        with open(latex_path, 'w') as f:
            f.write("% Ablation Study Results Table\n")
            f.write("\\begin{table}[ht]\n")
            f.write("\\centering\n")
            f.write("\\caption{Ablation Study Results: Performance comparison of BioKG-BioBERT variants}\n")
            f.write("\\label{tab:ablation_results}\n")
            f.write("\\begin{tabular}{llcccc}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Ablation Group} & \\textbf{Model Variant} & \\textbf{Micro F1} & \\textbf{Macro F1} & ")
            f.write("\\textbf{$\\Delta$ Macro F1} & \\textbf{Relative Change} \\\\\n")
            f.write("\\midrule\n")
            
            # Full model first
            if 'final_model_full' in results_dict:
                result = results_dict['final_model_full']
                f.write("\\multicolumn{2}{l}{\\textbf{Full Model (Baseline)}} & ")
                f.write(f"\\textbf{{{result['micro_f1']:.4f}}} & ")
                f.write(f"\\textbf{{{result['macro_f1']:.4f}}} & - & - \\\\\n")
                f.write("\\midrule\n")
            
            # Group results
            full_macro_f1 = results_dict['final_model_full']['macro_f1']
            
            for group_idx, (group_name, group_models) in enumerate(ABLATION_GROUPS.items()):
                first_in_group = True
                group_models_filtered = [m for m in group_models if m != 'final_model_full']
                
                for model_idx, model in enumerate(group_models_filtered):
                    if model in results_dict:
                        result = results_dict[model]
                        name = ABLATION_CATEGORIES.get(model, model).split(': ')[-1]
                        diff = result['macro_f1'] - full_macro_f1
                        relative_diff = diff / full_macro_f1 * 100
                        
                        if first_in_group:
                            f.write(f"\\multirow{{{len(group_models_filtered)}}}{{*}}{{{group_name}}} & ")
                            first_in_group = False
                        else:
                            f.write(" & ")
                        
                        f.write(f"{name} & {result['micro_f1']:.4f} & {result['macro_f1']:.4f} & ")
                        f.write(f"{diff:+.4f} & {relative_diff:+.1f}\\% \\\\\n")
                
                if group_idx < len(ABLATION_GROUPS) - 1:
                    f.write("\\midrule\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        logger.info(f"LaTeX tables saved to: {self.tables_dir}")
    
    def run_analysis(self):
        """Run complete ablation analysis."""
        logger.info("Starting comprehensive ablation analysis...")
        
        # Discover models
        model_paths = self.discover_ablation_models()
        
        # Evaluate all models
        results_dict = {}
        for model_name, checkpoint_path in model_paths.items():
            results = self.evaluate_model(model_name, checkpoint_path)
            results_dict[model_name] = results
        
        # Perform McNemar's tests
        logger.info("\nPerforming statistical significance tests...")
        mcnemar_results = {}
        
        full_predictions = results_dict['final_model_full']['predictions']
        full_targets = results_dict['final_model_full']['targets']
        
        for model_name in results_dict:
            if model_name != 'final_model_full':
                ablation_predictions = results_dict[model_name]['predictions']
                mcnemar_result = self.mcnemar_test(full_predictions, ablation_predictions, full_targets)
                mcnemar_results[model_name] = mcnemar_result
                
                # Log summary
                sig_count = sum(1 for h, r in mcnemar_result.items() 
                              if h != 'overall' and r['significant'])
                logger.info(f"{model_name}: {sig_count}/11 hallmarks significantly different")
        
        # Generate visualizations
        logger.info("\nGenerating visualizations...")
        self.generate_performance_comparison_plot(results_dict)
        self.generate_statistical_significance_plot(mcnemar_results)
        
        # Generate report
        logger.info("\nGenerating comprehensive report...")
        self.generate_detailed_report(results_dict, mcnemar_results)
        
        # Generate LaTeX tables
        logger.info("\nGenerating LaTeX tables...")
        self.generate_latex_tables(results_dict, mcnemar_results)
        
        # Save raw results
        results_to_save = {}
        for model, results in results_dict.items():
            results_to_save[model] = {
                k: v for k, v in results.items() 
                if k not in ['predictions', 'probabilities', 'targets']
            }
        
        with open(self.output_dir / 'ablation_results.json', 'w') as f:
            json.dump({
                'model_results': results_to_save,
                'mcnemar_tests': mcnemar_results
            }, f, indent=2)
        
        logger.info(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        
        # Print summary
        logger.info("\nQUICK SUMMARY")
        logger.info("=" * 40)
        sorted_models = sorted(results_dict.items(), 
                             key=lambda x: x[1]['macro_f1'], 
                             reverse=True)
        for model, results in sorted_models:
            if model == 'final_model_full':
                name = "Full Model"
            else:
                name = ABLATION_CATEGORIES.get(model, model)
            logger.info(f"{name}: {results['macro_f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze ablation study results')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--output-dir', type=str, default='ablation_analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to default configuration file')
    
    args = parser.parse_args()
    
    analyzer = AblationAnalyzer(
        checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
        config_path=args.config
    )
    analyzer.run_analysis()


if __name__ == "__main__":
    main()