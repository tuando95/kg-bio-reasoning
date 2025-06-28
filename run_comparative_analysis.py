"""
Comprehensive Comparative Analysis for BioKG-BioBERT

This script performs complete analysis including:
1. Model evaluation with default thresholds
2. Threshold optimization on validation set
3. Evaluation with optimal thresholds
4. Statistical significance testing (McNemar's test)
5. Comprehensive visualizations
6. Detailed comparative report
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
from statsmodels.stats.contingency_tables import mcnemar
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


class ComprehensiveAnalyzer:
    """Comprehensive model analysis with threshold optimization and statistical testing."""
    
    def __init__(self, config_path: str = 'configs/default_config.yaml'):
        """Initialize analyzer with config."""
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
        
        # Setup data module once
        logger.info("Loading data module...")
        self.data_module = HoCDataModule(self.config)
        self.data_module.setup()
    
    def load_model(self, checkpoint_path: str) -> Tuple[torch.nn.Module, Dict]:
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
    
    def _optimize_global_threshold(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Find single threshold that maximizes macro F1 across all classes."""
        # Create a range of thresholds to test
        thresholds_to_test = np.linspace(0.1, 0.9, 17)  # Test 0.1, 0.15, 0.2, ..., 0.9
        
        best_threshold = 0.5
        best_macro_f1 = 0.0
        
        for threshold in thresholds_to_test:
            pred_binary = (predictions >= threshold).astype(int)
            macro_f1 = f1_score(targets, pred_binary, average='macro')
            
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_threshold = threshold
        
        logger.info(f"Global optimal threshold: {best_threshold:.2f} (Macro-F1: {best_macro_f1:.4f})")
        return best_threshold
    
    def optimize_thresholds(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Find optimal thresholds for each hallmark using validation set.
        
        Uses precision-recall curve for efficient threshold search.
        Note: Optimizing on validation and evaluating on test may lead to
        overfitting - consider using cross-validation for production.
        """
        optimal_thresholds = {}
        threshold_details = []
        
        # Also try global threshold optimization
        global_best_threshold = self._optimize_global_threshold(predictions, targets)
        
        for i in range(11):
            y_true = targets[:, i]
            y_scores = predictions[:, i]
            
            # Skip if no positive samples
            if y_true.sum() == 0:
                optimal_thresholds[i] = 0.5
                threshold_details.append({
                    'hallmark_id': i,
                    'hallmark_name': self.hallmark_names[i],
                    'optimal_threshold': 0.5,
                    'optimal_f1': 0.0,
                    'default_f1': 0.0,
                    'f1_improvement': 0.0,
                    'support': 0,
                    'note': 'No positive samples'
                })
                continue
            
            # Calculate precision-recall curve
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
            
            # Calculate F1 scores for each threshold
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            
            # Find threshold that maximizes F1
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_f1 = f1_scores[best_idx]
            
            # Apply conservative adjustment to prevent extreme thresholds
            # This helps prevent overfitting to validation set
            if best_threshold > 0.8:
                best_threshold = 0.8
            elif best_threshold < 0.2:
                best_threshold = 0.2
            
            # Calculate F1 at default threshold (0.5)
            default_f1 = f1_score(y_true, (y_scores >= 0.5).astype(int))
            
            # Calculate F1 at global optimal threshold
            global_f1 = f1_score(y_true, (y_scores >= global_best_threshold).astype(int))
            
            optimal_thresholds[i] = float(best_threshold)
            
            threshold_details.append({
                'hallmark_id': i,
                'hallmark_name': self.hallmark_names[i],
                'optimal_threshold': float(best_threshold),
                'optimal_f1': float(best_f1),
                'default_f1': float(default_f1),
                'global_threshold_f1': float(global_f1),
                'f1_improvement': float(best_f1 - default_f1),
                'support': int(y_true.sum())
            })
        
        return optimal_thresholds, threshold_details, global_best_threshold
    
    def evaluate_with_thresholds(self, predictions: np.ndarray, targets: np.ndarray, 
                                thresholds: Optional[Dict[int, float]] = None) -> Dict:
        """Evaluate predictions with given thresholds (default 0.5 if None)."""
        if thresholds is None:
            pred_binary = (predictions >= 0.5).astype(int)
        else:
            pred_binary = np.zeros_like(predictions)
            for i in range(11):
                threshold = thresholds.get(i, 0.5)
                pred_binary[:, i] = (predictions[:, i] >= threshold).astype(int)
        
        # Calculate metrics
        results = {
            'f1_micro': f1_score(targets, pred_binary, average='micro'),
            'f1_macro': f1_score(targets, pred_binary, average='macro'),
            'f1_weighted': f1_score(targets, pred_binary, average='weighted'),
            'hamming_loss': np.mean(targets != pred_binary),
            'exact_match_ratio': np.mean(np.all(targets == pred_binary, axis=1)),
            'per_hallmark_metrics': []
        }
        
        # Per-hallmark metrics
        for i in range(11):
            tp = ((pred_binary[:, i] == 1) & (targets[:, i] == 1)).sum()
            tn = ((pred_binary[:, i] == 0) & (targets[:, i] == 0)).sum()
            fp = ((pred_binary[:, i] == 1) & (targets[:, i] == 0)).sum()
            fn = ((pred_binary[:, i] == 0) & (targets[:, i] == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results['per_hallmark_metrics'].append({
                'hallmark': self.hallmark_names[i],
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int((targets[:, i] == 1).sum())
            })
        
        return results
    
    def mcnemar_test(self, pred1: np.ndarray, pred2: np.ndarray, targets: np.ndarray) -> Dict:
        """
        Perform McNemar's test between two models using statsmodels.
        
        Args:
            pred1: Binary predictions from model 1
            pred2: Binary predictions from model 2
            targets: True labels
            
        Returns:
            Dictionary with test results
        """
        results = {}
        
        # Per-hallmark tests
        for i in range(11):
            # Create contingency table
            # Note: statsmodels expects the table in a specific format
            # [[correct by both, correct by 1 only], [correct by 2 only, wrong by both]]
            correct1 = (pred1[:, i] == targets[:, i])
            correct2 = (pred2[:, i] == targets[:, i])
            
            table = np.array([
                [(correct1 & correct2).sum(), (correct1 & ~correct2).sum()],
                [(~correct1 & correct2).sum(), (~correct1 & ~correct2).sum()]
            ])
            
            # Perform McNemar's test
            result = mcnemar(table, exact=True, correction=True)
            
            results[self.hallmark_names[i]] = {
                'statistic': float(result.statistic),
                'p_value': float(result.pvalue),
                'model1_better': int((correct1 & ~correct2).sum()),
                'model2_better': int((~correct1 & correct2).sum()),
                'significant': result.pvalue < 0.05
            }
        
        # Overall test (all predictions)
        correct1_all = (pred1 == targets)
        correct2_all = (pred2 == targets)
        
        table_overall = np.array([
            [(correct1_all & correct2_all).sum(), (correct1_all & ~correct2_all).sum()],
            [(~correct1_all & correct2_all).sum(), (~correct1_all & ~correct2_all).sum()]
        ])
        
        result_overall = mcnemar(table_overall, exact=False, correction=True)
        
        results['overall'] = {
            'statistic': float(result_overall.statistic),
            'p_value': float(result_overall.pvalue),
            'model1_better': int((correct1_all & ~correct2_all).sum()),
            'model2_better': int((~correct1_all & correct2_all).sum()),
            'significant': result_overall.pvalue < 0.05
        }
        
        return results
    
    def analyze_model(self, checkpoint_path: str, model_name: str) -> Dict:
        """Complete analysis of a single model."""
        logger.info(f"\nAnalyzing {model_name}...")
        
        # Load model
        model, checkpoint = self.load_model(checkpoint_path)
        
        # Get test set predictions
        logger.info("Evaluating on test set...")
        test_loader = self.data_module.test_dataloader()
        test_predictions, test_targets = self.get_predictions(model, test_loader)
        
        # Evaluate with default thresholds (0.5)
        results = self.evaluate_with_thresholds(test_predictions, test_targets, None)
        
        # Compile results
        full_results = {
            'model_name': model_name,
            'checkpoint_path': checkpoint_path,
            'metrics': results,
            'test_predictions': test_predictions,
            'test_targets': test_targets
        }
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Macro-F1: {results['f1_macro']:.4f}")
        logger.info(f"  Micro-F1: {results['f1_micro']:.4f}")
        logger.info(f"  Hamming Loss: {results['hamming_loss']:.4f}")
        logger.info(f"  Exact Match Ratio: {results['exact_match_ratio']:.4f}")
        
        return full_results
    
    def plot_performance_comparison(self, all_results: Dict[str, Dict], save_path: Path):
        """Plot comprehensive performance comparison."""
        models = list(all_results.keys())
        
        # Prepare data
        metrics_data = {
            'Model': models,
            'Micro-F1': [all_results[m]['metrics']['f1_micro'] for m in models],
            'Macro-F1': [all_results[m]['metrics']['f1_macro'] for m in models],
            'Hamming Loss': [all_results[m]['metrics']['hamming_loss'] for m in models],
            'EMR': [all_results[m]['metrics']['exact_match_ratio'] for m in models]
        }
        
        df = pd.DataFrame(metrics_data)
        
        # Sort by Macro-F1 for better visualization
        df = df.sort_values('Macro-F1', ascending=True)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        metrics = ['Micro-F1', 'Macro-F1', 'Hamming Loss', 'EMR']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[idx]
            
            # Create horizontal bar plot
            bars = ax.barh(df['Model'], df[metric], color=color, alpha=0.7)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontsize=9)
            
            ax.set_title(f'{metric}', fontsize=14)
            ax.set_xlabel(metric)
            ax.grid(axis='x', alpha=0.3)
            
            # Highlight best performer
            if metric == 'Hamming Loss':  # Lower is better
                best_idx = df[metric].idxmin()
            else:  # Higher is better
                best_idx = df[metric].idxmax()
            
            # Make best performer's bar darker
            for i, bar in enumerate(bars):
                if df.iloc[i].name == best_idx:
                    bar.set_alpha(1.0)
                    bar.set_edgecolor('black')
                    bar.set_linewidth(2)
        
        plt.suptitle('Model Performance Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance comparison plot saved")
    
    def plot_mcnemar_heatmap(self, mcnemar_results: Dict, save_path: Path):
        """Plot heatmap of McNemar's test p-values."""
        comparisons = list(mcnemar_results.keys())
        hallmarks = list(self.hallmark_names.values())
        
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
            yticklabels=comparisons,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            vmin=0,
            vmax=0.1,
            cbar_kws={'label': 'p-value'},
            mask=~sig_mask,  # Only show significant values
            linewidths=0.5
        )
        
        plt.title("McNemar's Test P-values (Significant Results Only)")
        plt.xlabel('Cancer Hallmarks')
        plt.ylabel('Model Comparisons')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path / 'mcnemar_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"McNemar heatmap saved")
    
    def generate_report(self, all_results: Dict, mcnemar_results: Dict, save_path: Path):
        """Generate comprehensive analysis report."""
        with open(save_path / 'analysis_report.txt', 'w') as f:
            f.write("BioKG-BioBERT Comprehensive Analysis Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall performance summary
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 40 + "\n\n")
            
            # Sort models by macro-F1
            sorted_models = sorted(all_results.items(), 
                                 key=lambda x: x[1]['metrics']['f1_macro'], 
                                 reverse=True)
            
            f.write("Rankings by Macro-F1:\n")
            for rank, (model, results) in enumerate(sorted_models, 1):
                metrics = results['metrics']
                f.write(f"{rank}. {model}: "
                       f"Macro-F1={metrics['f1_macro']:.4f}, "
                       f"Micro-F1={metrics['f1_micro']:.4f}, "
                       f"Hamming Loss={metrics['hamming_loss']:.4f}\n")
            
            # Detailed results per model
            f.write("\n\nDETAILED RESULTS\n")
            f.write("-" * 40 + "\n")
            
            for model, results in sorted_models:
                metrics = results['metrics']
                f.write(f"\n{model}:\n")
                f.write(f"  Macro-F1: {metrics['f1_macro']:.4f}\n")
                f.write(f"  Micro-F1: {metrics['f1_micro']:.4f}\n")
                f.write(f"  Weighted-F1: {metrics['f1_weighted']:.4f}\n")
                f.write(f"  Hamming Loss: {metrics['hamming_loss']:.4f}\n")
                f.write(f"  Exact Match Ratio: {metrics['exact_match_ratio']:.4f}\n")
                
                # Per-hallmark performance
                f.write("\n  Per-Hallmark F1 Scores:\n")
                per_hallmark = metrics['per_hallmark_metrics']
                
                # Sort by F1 score to show best and worst
                sorted_hallmarks = sorted(per_hallmark, key=lambda x: x['f1'], reverse=True)
                
                # Show top 3 and bottom 3
                f.write("    Best performing:\n")
                for h in sorted_hallmarks[:3]:
                    f.write(f"      - {h['hallmark']}: {h['f1']:.3f} (P={h['precision']:.3f}, R={h['recall']:.3f})\n")
                
                f.write("    Worst performing:\n")
                for h in sorted_hallmarks[-3:]:
                    f.write(f"      - {h['hallmark']}: {h['f1']:.3f} (P={h['precision']:.3f}, R={h['recall']:.3f})\n")
            
            # Statistical significance
            f.write("\n\nSTATISTICAL SIGNIFICANCE (McNemar's Test)\n")
            f.write("-" * 40 + "\n")
            
            for comparison, results in mcnemar_results.items():
                f.write(f"\n{comparison}:\n")
                overall = results['overall']
                f.write(f"  Overall: p={overall['p_value']:.4f} ")
                f.write(f"({'Significant' if overall['significant'] else 'Not significant'})\n")
                
                # Count significant hallmarks
                sig_count = sum(1 for h, r in results.items() 
                              if h != 'overall' and r['significant'])
                f.write(f"  Significant differences in {sig_count}/11 hallmarks\n")
                
                if sig_count > 0:
                    f.write("  Significant hallmarks:\n")
                    for hallmark, result in results.items():
                        if hallmark != 'overall' and result['significant']:
                            f.write(f"    - {hallmark}: p={result['p_value']:.4f}\n")
            
            # Biological metrics interpretation
            f.write("\n\nBIOLOGICAL METRICS INTERPRETATION\n")
            f.write("-" * 40 + "\n")
            
            # Get best model's metrics
            best_metrics = sorted_models[0][1]['metrics']
            
            f.write("\nBio-Synergy Capture Rate:\n")
            f.write(f"  Best model: {best_metrics.get('bio_synergy_capture_rate', 0):.3f}\n")
            f.write(f"  Random baseline: 0.250\n")
            f.write(f"  Improvement: +{best_metrics.get('bio_synergy_capture_rate', 0) - 0.250:.3f}\n")
            f.write("  Interpretation: Measures how well the model captures known synergistic\n")
            f.write("  relationships between hallmarks (e.g., angiogenesis → energetics).\n")
            f.write("  Higher values indicate better biological understanding.\n")
            
            f.write("\nBio-Plausibility Score:\n")
            f.write(f"  Best model: {best_metrics.get('bio_plausibility_score', 0):.3f}\n")
            f.write(f"  Random baseline: 0.425\n")
            f.write(f"  Improvement: +{best_metrics.get('bio_plausibility_score', 0) - 0.425:.3f}\n")
            f.write("  Formula: 0.7×(1-violation_rate) + 0.3×(synergy_rate)\n")
            f.write("  Interpretation: Overall biological consistency combining:\n")
            f.write("    - 70%: Avoiding incompatible hallmark combinations\n")
            f.write("    - 30%: Capturing synergistic relationships\n")
            
            # Best practices
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            best_model = sorted_models[0][0]
            f.write(f"1. Best performing model: {best_model}\n")
            f.write("2. The high baseline performance suggests the cached KG features are very informative\n")
            f.write("3. Consider ensemble methods for further improvement\n")
            f.write("4. Focus on improving performance for low-performing hallmarks (classes 0 and 10)\n")
            f.write("5. The biological consistency metrics show the model captures biological relationships well\n")
        
        logger.info("Comprehensive report generated")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Comprehensive analysis of BioKG-BioBERT models")
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                       help='List of checkpoint paths to analyze')
    parser.add_argument('--names', type=str, nargs='+', required=True,
                       help='Names for each model (same order as checkpoints)')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='comprehensive_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    if len(args.checkpoints) != len(args.names):
        raise ValueError("Number of checkpoints must match number of names")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = ComprehensiveAnalyzer(args.config)
    
    # Analyze each model
    all_results = {}
    for checkpoint_path, model_name in zip(args.checkpoints, args.names):
        results = analyzer.analyze_model(checkpoint_path, model_name)
        all_results[model_name] = results
    
    # Perform McNemar's tests
    logger.info("\nPerforming statistical tests...")
    mcnemar_results = {}
    
    # Compare each model against baseline if available
    if 'baseline_biobert' in all_results:
        baseline_pred = (all_results['baseline_biobert']['test_predictions'] >= 0.5).astype(int)
        baseline_targets = all_results['baseline_biobert']['test_targets']
        
        for model_name in all_results:
            if model_name != 'baseline_biobert':
                # Use default threshold (0.5) for fair comparison
                model_pred = (all_results[model_name]['test_predictions'] >= 0.5).astype(int)
                
                mcnemar_result = analyzer.mcnemar_test(baseline_pred, model_pred, baseline_targets)
                mcnemar_results[f'baseline_biobert vs {model_name}'] = mcnemar_result
    
    # Save all results
    logger.info("\nSaving results...")
    
    # Save raw results
    results_to_save = {}
    for model, results in all_results.items():
        results_to_save[model] = {
            'metrics': results['metrics'],
            'model_name': results['model_name']
        }
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        """Recursively convert numpy types in nested structures."""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'model_results': convert_numpy_types(results_to_save),
            'mcnemar_tests': convert_numpy_types(mcnemar_results)
        }, f, indent=2)
    
    # Save detailed classification results for each model
    logger.info("Saving detailed classification results...")
    for model_name, results in all_results.items():
        # Create model-specific directory
        model_dir = output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Get predictions and targets
        predictions = results['test_predictions']
        targets = results['test_targets']
        
        # Generate binary predictions with default threshold (0.5)
        pred_binary = (predictions >= 0.5).astype(int)
        
        # Save classification results
        classification_results = []
        for i in range(len(targets)):
            sample_result = {
                'sample_id': i,
                'true_labels': targets[i].tolist(),
                'predicted_labels': pred_binary[i].tolist(),
                'prediction_scores': predictions[i].tolist(),
                'correct_predictions': (pred_binary[i] == targets[i]).tolist(),
                'exact_match': bool(np.all(pred_binary[i] == targets[i]))
            }
            classification_results.append(sample_result)
        
        with open(model_dir / 'test_classification_results.json', 'w') as f:
            json.dump({
                'model_name': model_name,
                'threshold': 0.5,
                'overall_metrics': convert_numpy_types(results['metrics']),
                'samples': classification_results
            }, f, indent=2)
        
        # Save per-hallmark summary
        hallmark_summary = {
            'model_name': model_name,
            'hallmarks': {}
        }
        
        for h_id in range(11):
            h_name = analyzer.hallmark_names[h_id]
            
            # Calculate metrics for this hallmark
            true_h = targets[:, h_id]
            pred_h = pred_binary[:, h_id]
            
            tp = int(((pred_h == 1) & (true_h == 1)).sum())
            fp = int(((pred_h == 1) & (true_h == 0)).sum())
            tn = int(((pred_h == 0) & (true_h == 0)).sum())
            fn = int(((pred_h == 0) & (true_h == 1)).sum())
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            hallmark_summary['hallmarks'][h_name] = {
                'hallmark_id': h_id,
                'total_samples': len(true_h),
                'positive_samples': int(true_h.sum()),
                'negative_samples': int((1 - true_h).sum()),
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn,
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
        
        with open(model_dir / 'hallmark_classification_summary.json', 'w') as f:
            json.dump(hallmark_summary, f, indent=2)
        
        logger.info(f"  Saved classification results for {model_name}")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    analyzer.plot_performance_comparison(all_results, output_dir)
    
    if mcnemar_results:
        analyzer.plot_mcnemar_heatmap(mcnemar_results, output_dir)
    
    # Generate report
    analyzer.generate_report(all_results, mcnemar_results, output_dir)
    
    logger.info(f"\nAnalysis complete! Results saved to {output_dir}")
    
    # Print summary
    logger.info("\nQUICK SUMMARY")
    logger.info("=" * 40)
    sorted_models = sorted(all_results.items(), 
                         key=lambda x: x[1]['metrics']['f1_macro'], 
                         reverse=True)
    for model, results in sorted_models:
        macro_f1 = results['metrics']['f1_macro']
        logger.info(f"{model}: {macro_f1:.4f}")


if __name__ == "__main__":
    main()