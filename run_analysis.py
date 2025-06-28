"""
Comprehensive Analysis Script for BioKG-BioBERT vs Baselines

This script performs detailed comparative analysis including:
- Performance comparison between baselines and BioKG-BioBERT
- McNemar's statistical test for model comparison
- Attention visualizations
- Error analysis
- Performance breakdown by cancer hallmark
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
from scipy.stats import mcnemar
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, 
    precision_recall_curve, roc_curve, auc
)
import argparse
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.biokg_biobert import BioKGBioBERT
from src.data.dataset import HoCDataModule
from src.evaluation import Evaluator
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare multiple models and perform statistical tests."""
    
    def __init__(self, config_path: str = 'configs/default_config.yaml'):
        """Initialize comparator with config."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    def load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load model from checkpoint."""
        logger.info(f"Loading model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract config from checkpoint if available
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
        else:
            model_config = self.config['model'].copy()
        
        # Check if checkpoint has auxiliary components
        state_dict_keys = checkpoint['model_state_dict'].keys()
        has_pathway_classifier = any('pathway_classifier' in key for key in state_dict_keys)
        has_consistency_predictor = any('consistency_predictor' in key for key in state_dict_keys)
        
        # Set loss weights based on what's in the checkpoint
        if has_pathway_classifier and has_consistency_predictor:
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
        
        return model, checkpoint
    
    def get_predictions(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions on a dataset."""
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch_device = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = model(**batch_device)
                
                predictions = torch.sigmoid(outputs['logits'])
                all_predictions.append(predictions.cpu())
                all_targets.append(batch_device['labels'].cpu())
        
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        return all_predictions, all_targets
    
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
    
    def mcnemar_test(self, pred1: np.ndarray, pred2: np.ndarray, targets: np.ndarray) -> Dict:
        """
        Perform McNemar's test between two models.
        
        Args:
            pred1: Predictions from model 1 (binary)
            pred2: Predictions from model 2 (binary)
            targets: True labels
            
        Returns:
            Dictionary with test results for each hallmark
        """
        results = {}
        
        for i in range(11):
            # Get binary predictions for this hallmark
            p1 = pred1[:, i]
            p2 = pred2[:, i]
            t = targets[:, i]
            
            # Create contingency table
            # a: both correct
            # b: model1 correct, model2 wrong
            # c: model1 wrong, model2 correct
            # d: both wrong
            a = np.sum((p1 == t) & (p2 == t))
            b = np.sum((p1 == t) & (p2 != t))
            c = np.sum((p1 != t) & (p2 == t))
            d = np.sum((p1 != t) & (p2 != t))
            
            # Perform McNemar's test
            # Use continuity correction for small samples
            n = b + c
            if n > 0:
                if n < 25:  # Use exact binomial test for small samples
                    from scipy.stats import binom_test
                    p_value = binom_test(b, n, 0.5)
                else:
                    # Use chi-squared approximation with continuity correction
                    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
                    from scipy.stats import chi2 as chi2_dist
                    p_value = 1 - chi2_dist.cdf(chi2, df=1)
            else:
                p_value = 1.0  # No disagreement
            
            results[self.hallmark_names[i]] = {
                'contingency_table': {'a': int(a), 'b': int(b), 'c': int(c), 'd': int(d)},
                'p_value': float(p_value),
                'model1_better': int(b),
                'model2_better': int(c),
                'significant': p_value < 0.05
            }
        
        # Overall McNemar's test (all predictions)
        p1_flat = pred1.flatten()
        p2_flat = pred2.flatten()
        t_flat = targets.flatten()
        
        b_overall = np.sum((p1_flat == t_flat) & (p2_flat != t_flat))
        c_overall = np.sum((p1_flat != t_flat) & (p2_flat == t_flat))
        
        if b_overall + c_overall > 25:
            chi2 = (abs(b_overall - c_overall) - 1) ** 2 / (b_overall + c_overall)
            from scipy.stats import chi2 as chi2_dist
            p_value_overall = 1 - chi2_dist.cdf(chi2, df=1)
        else:
            from scipy.stats import binom_test
            p_value_overall = binom_test(b_overall, b_overall + c_overall, 0.5)
        
        results['overall'] = {
            'model1_better': int(b_overall),
            'model2_better': int(c_overall),
            'p_value': float(p_value_overall),
            'significant': p_value_overall < 0.05
        }
        
        return results
    
    def plot_performance_comparison(self, results_dict: Dict, save_path: str):
        """Plot performance comparison across models."""
        # Prepare data for plotting
        models = list(results_dict.keys())
        metrics = ['f1_micro', 'f1_macro', 'hamming_loss', 'exact_match_ratio']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            values = [results_dict[model].get(metric, 0) for model in models]
            
            ax = axes[idx]
            bars = ax.bar(range(len(models)), values)
            
            # Color the best performing model
            best_idx = np.argmax(values) if metric != 'hamming_loss' else np.argmin(values)
            bars[best_idx].set_color('green')
            
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            
            # Add value labels on bars
            for i, v in enumerate(values):
                ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance comparison plot saved to {save_path}")
    
    def plot_hallmark_heatmap(self, results_dict: Dict, save_path: str):
        """Plot heatmap of per-hallmark F1 scores."""
        models = list(results_dict.keys())
        
        # Extract per-hallmark F1 scores
        f1_matrix = []
        for model in models:
            if 'per_hallmark_f1' in results_dict[model]:
                f1_scores = [results_dict[model]['per_hallmark_f1'][i] for i in range(11)]
            else:
                f1_scores = [0] * 11
            f1_matrix.append(f1_scores)
        
        f1_matrix = np.array(f1_matrix)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            f1_matrix,
            xticklabels=[self.hallmark_names[i] for i in range(11)],
            yticklabels=models,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'F1 Score'}
        )
        
        plt.title('Per-Hallmark F1 Score Comparison')
        plt.xlabel('Cancer Hallmarks')
        plt.ylabel('Models')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Hallmark heatmap saved to {save_path}")
    
    def plot_attention_visualization(self, model: torch.nn.Module, sample_text: str, save_path: str):
        """Visualize biological attention weights for a sample."""
        if not hasattr(model, 'bio_attention_layers') or not model.use_bio_attention:
            logger.warning("Model does not have biological attention layers")
            return
        
        # This is a placeholder - actual implementation would require
        # extracting attention weights during forward pass
        logger.info("Attention visualization would be saved to {save_path}")
    
    def generate_comparison_report(self, results_dict: Dict, mcnemar_results: Dict, save_path: str):
        """Generate comprehensive comparison report."""
        with open(save_path, 'w') as f:
            f.write("BioKG-BioBERT Comparative Analysis Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall performance comparison
            f.write("OVERALL PERFORMANCE COMPARISON\n")
            f.write("-" * 40 + "\n")
            
            # Create comparison table
            metrics = ['f1_micro', 'f1_macro', 'hamming_loss', 'exact_match_ratio']
            f.write(f"{'Model':<30} {'Micro-F1':<10} {'Macro-F1':<10} {'Hamming':<10} {'EMR':<10}\n")
            f.write("-" * 70 + "\n")
            
            for model, results in results_dict.items():
                f.write(f"{model:<30} ")
                for metric in metrics:
                    value = results.get(metric, 0)
                    f.write(f"{value:<10.4f} ")
                f.write("\n")
            
            # McNemar's test results
            f.write("\n\nSTATISTICAL SIGNIFICANCE (McNemar's Test)\n")
            f.write("-" * 40 + "\n")
            
            for comparison, mcnemar_result in mcnemar_results.items():
                f.write(f"\n{comparison}:\n")
                overall = mcnemar_result['overall']
                f.write(f"  Overall p-value: {overall['p_value']:.4f} ")
                f.write(f"({'Significant' if overall['significant'] else 'Not significant'})\n")
                f.write(f"  Model 1 better: {overall['model1_better']} cases\n")
                f.write(f"  Model 2 better: {overall['model2_better']} cases\n")
                
                # Per-hallmark significant differences
                f.write("\n  Significant differences by hallmark:\n")
                for hallmark, result in mcnemar_result.items():
                    if hallmark != 'overall' and result['significant']:
                        f.write(f"    - {hallmark}: p={result['p_value']:.4f}\n")
            
            # Best model summary
            f.write("\n\nBEST PERFORMING MODEL\n")
            f.write("-" * 40 + "\n")
            
            # Find best model by macro-F1
            best_model = max(results_dict.items(), key=lambda x: x[1].get('f1_macro', 0))
            f.write(f"Model: {best_model[0]}\n")
            f.write(f"Macro-F1: {best_model[1].get('f1_macro', 0):.4f}\n")
            f.write(f"Micro-F1: {best_model[1].get('f1_micro', 0):.4f}\n")
            
            # Improvement over baseline
            if 'baseline_biobert' in results_dict:
                baseline_f1 = results_dict['baseline_biobert'].get('f1_macro', 0)
                improvement = best_model[1].get('f1_macro', 0) - baseline_f1
                f.write(f"\nImprovement over BioBERT baseline:\n")
                f.write(f"  Absolute: +{improvement:.4f}\n")
                f.write(f"  Relative: +{(improvement/baseline_f1)*100:.2f}%\n")
        
        logger.info(f"Comparison report saved to {save_path}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Comparative analysis of BioKG-BioBERT models")
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                       help='List of checkpoint paths to compare')
    parser.add_argument('--names', type=str, nargs='+', required=True,
                       help='Names for each model (same order as checkpoints)')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='analysis_comparison',
                       help='Directory to save analysis results')
    parser.add_argument('--test_split', type=str, default='test',
                       choices=['validation', 'test'],
                       help='Which split to evaluate on')
    
    args = parser.parse_args()
    
    if len(args.checkpoints) != len(args.names):
        raise ValueError("Number of checkpoints must match number of names")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize comparator
    comparator = ModelComparator(args.config)
    
    # Load data
    logger.info("Loading data...")
    data_module = HoCDataModule(comparator.config)
    data_module.setup()
    
    if args.test_split == 'test':
        dataloader = data_module.test_dataloader()
    else:
        dataloader = data_module.val_dataloader()
    
    # Evaluate each model
    results_dict = {}
    predictions_dict = {}
    
    for checkpoint_path, model_name in zip(args.checkpoints, args.names):
        logger.info(f"\nEvaluating {model_name}...")
        
        # Load model
        model, checkpoint = comparator.load_model(checkpoint_path)
        
        # Get predictions
        predictions, targets = comparator.get_predictions(model, dataloader)
        predictions_dict[model_name] = predictions
        
        # Calculate metrics
        pred_binary = (predictions >= 0.5).astype(int)
        
        results = {
            'f1_micro': f1_score(targets, pred_binary, average='micro'),
            'f1_macro': f1_score(targets, pred_binary, average='macro'),
            'f1_weighted': f1_score(targets, pred_binary, average='weighted'),
            'hamming_loss': np.mean(targets != pred_binary),
            'exact_match_ratio': np.mean(np.all(targets == pred_binary, axis=1)),
            'per_hallmark_f1': [f1_score(targets[:, i], pred_binary[:, i]) for i in range(11)]
        }
        
        results_dict[model_name] = results
        
        logger.info(f"{model_name} - Macro-F1: {results['f1_macro']:.4f}, Micro-F1: {results['f1_micro']:.4f}")
    
    # Perform McNemar's tests
    mcnemar_results = {}
    
    # Compare each model against baseline
    if 'baseline_biobert' in args.names:
        baseline_idx = args.names.index('baseline_biobert')
        baseline_pred = (predictions_dict[args.names[baseline_idx]] >= 0.5).astype(int)
        
        for i, model_name in enumerate(args.names):
            if i != baseline_idx:
                model_pred = (predictions_dict[model_name] >= 0.5).astype(int)
                mcnemar_result = comparator.mcnemar_test(baseline_pred, model_pred, targets)
                mcnemar_results[f'baseline_biobert vs {model_name}'] = mcnemar_result
    
    # Compare best model against second best
    sorted_models = sorted(results_dict.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
    if len(sorted_models) >= 2:
        best_name = sorted_models[0][0]
        second_name = sorted_models[1][0]
        best_pred = (predictions_dict[best_name] >= 0.5).astype(int)
        second_pred = (predictions_dict[second_name] >= 0.5).astype(int)
        mcnemar_result = comparator.mcnemar_test(second_pred, best_pred, targets)
        mcnemar_results[f'{second_name} vs {best_name}'] = mcnemar_result
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'model_results': results_dict,
            'mcnemar_tests': mcnemar_results
        }, f, indent=2)
    
    # Generate visualizations
    comparator.plot_performance_comparison(results_dict, output_dir / 'performance_comparison.png')
    comparator.plot_hallmark_heatmap(results_dict, output_dir / 'hallmark_heatmap.png')
    
    # Generate report
    comparator.generate_comparison_report(
        results_dict, 
        mcnemar_results,
        output_dir / 'comparison_report.txt'
    )
    
    logger.info(f"\nAnalysis complete! Results saved to {output_dir}")
    
    # Print summary
    logger.info("\nSUMMARY")
    logger.info("=" * 40)
    for model_name, results in sorted(results_dict.items(), key=lambda x: x[1]['f1_macro'], reverse=True):
        logger.info(f"{model_name}: Macro-F1={results['f1_macro']:.4f}, Micro-F1={results['f1_micro']:.4f}")
    
    if mcnemar_results:
        logger.info("\nStatistical Significance:")
        for comparison, result in mcnemar_results.items():
            if result['overall']['significant']:
                logger.info(f"  {comparison}: p={result['overall']['p_value']:.4f} (Significant)")


if __name__ == "__main__":
    main()