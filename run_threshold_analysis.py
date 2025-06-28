"""
Threshold Optimization Analysis for Model Comparison

This script optimizes classification thresholds for each model and compares
the improvements gained through threshold optimization.
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
from sklearn.metrics import f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.biokg_biobert import BioKGBioBERT
from src.data.dataset import HoCDataModule

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def optimize_thresholds_for_model(checkpoint_path: str, config_path: str = 'configs/default_config.yaml'):
    """Optimize thresholds for a single model."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize model
    model_config = config['model'].copy()
    
    # Check if checkpoint has auxiliary components
    state_dict_keys = checkpoint['model_state_dict'].keys()
    has_pathway_classifier = any('pathway_classifier' in key for key in state_dict_keys)
    has_consistency_predictor = any('consistency_predictor' in key for key in state_dict_keys)
    
    if has_pathway_classifier and has_consistency_predictor:
        model_config['loss_weights'] = config['training']['loss_weights']
    else:
        model_config['loss_weights'] = {
            'hallmark_loss': 1.0,
            'pathway_loss': 0.0,
            'consistency_loss': 0.0
        }
    
    model = BioKGBioBERT(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    # Get validation predictions
    data_module = HoCDataModule(config)
    data_module.setup()
    val_loader = data_module.val_dataloader()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch_device = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_device[key] = value.to(device)
                elif key == 'graph_data':
                    batch_device[key] = value.to(device) if value is not None else None
                elif key == 'biological_context':
                    moved_context = {}
                    for ctx_key, ctx_value in value.items():
                        if isinstance(ctx_value, torch.Tensor):
                            moved_context[ctx_key] = ctx_value.to(device)
                        else:
                            moved_context[ctx_key] = ctx_value
                    batch_device[key] = moved_context
                else:
                    batch_device[key] = value
            
            outputs = model(**batch_device)
            predictions = torch.sigmoid(outputs['logits'])
            all_predictions.append(predictions.cpu())
            all_targets.append(batch_device['labels'].cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Find optimal thresholds
    optimal_thresholds = {}
    threshold_analysis = []
    
    hallmark_names = {
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
    
    for i in range(11):
        y_true = all_targets[:, i]
        y_scores = all_predictions[:, i]
        
        # Skip if no positive samples
        if y_true.sum() == 0:
            optimal_thresholds[i] = 0.5
            continue
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        
        # Find threshold that maximizes F1
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]
        
        # Calculate F1 at default threshold (0.5)
        default_f1 = f1_score(y_true, (y_scores >= 0.5).astype(int))
        
        optimal_thresholds[i] = float(best_threshold)
        
        threshold_analysis.append({
            'hallmark_id': i,
            'hallmark_name': hallmark_names[i],
            'optimal_threshold': float(best_threshold),
            'optimal_f1': float(best_f1),
            'default_threshold': 0.5,
            'default_f1': float(default_f1),
            'f1_improvement': float(best_f1 - default_f1),
            'support': int(y_true.sum())
        })
    
    # Calculate overall performance with optimal thresholds
    pred_optimal = np.zeros_like(all_predictions)
    for i in range(11):
        pred_optimal[:, i] = (all_predictions[:, i] >= optimal_thresholds[i]).astype(int)
    
    pred_default = (all_predictions >= 0.5).astype(int)
    
    results = {
        'f1_micro_default': f1_score(all_targets, pred_default, average='micro'),
        'f1_macro_default': f1_score(all_targets, pred_default, average='macro'),
        'f1_micro_optimal': f1_score(all_targets, pred_optimal, average='micro'),
        'f1_macro_optimal': f1_score(all_targets, pred_optimal, average='macro'),
        'optimal_thresholds': optimal_thresholds,
        'threshold_analysis': threshold_analysis
    }
    
    return results, all_predictions, all_targets


def plot_threshold_comparison(results_dict: Dict, save_path: str):
    """Plot comparison of threshold optimization benefits across models."""
    
    models = list(results_dict.keys())
    
    # Prepare data for plotting
    default_macro = [results_dict[m]['f1_macro_default'] for m in models]
    optimal_macro = [results_dict[m]['f1_macro_optimal'] for m in models]
    improvements = [results_dict[m]['f1_macro_optimal'] - results_dict[m]['f1_macro_default'] for m in models]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Before/After comparison
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, default_macro, width, label='Default (0.5)', alpha=0.8)
    bars2 = ax1.bar(x + width/2, optimal_macro, width, label='Optimized', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Macro F1 Score')
    ax1.set_title('F1 Score: Default vs Optimized Thresholds')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Improvement amounts
    bars = ax2.bar(models, improvements, color=['green' if imp > 0 else 'red' for imp in improvements])
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0005,
                f'+{imp:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Models')
    ax2.set_ylabel('F1 Score Improvement')
    ax2.set_title('Improvement from Threshold Optimization')
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Threshold comparison plot saved to {save_path}")


def plot_threshold_heatmap(results_dict: Dict, save_path: str):
    """Plot heatmap of optimal thresholds for each model and hallmark."""
    
    models = list(results_dict.keys())
    hallmark_names = {
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
    
    # Create threshold matrix
    threshold_matrix = []
    for model in models:
        thresholds = [results_dict[model]['optimal_thresholds'][i] for i in range(11)]
        threshold_matrix.append(thresholds)
    
    threshold_matrix = np.array(threshold_matrix)
    
    # Create heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        threshold_matrix,
        xticklabels=[hallmark_names[i] for i in range(11)],
        yticklabels=models,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Optimal Threshold'}
    )
    
    plt.title('Optimal Thresholds by Model and Hallmark')
    plt.xlabel('Cancer Hallmarks')
    plt.ylabel('Models')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Threshold heatmap saved to {save_path}")


def main():
    """Main function for threshold analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Threshold optimization analysis")
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                       help='List of checkpoint paths to analyze')
    parser.add_argument('--names', type=str, nargs='+', required=True,
                       help='Names for each model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='threshold_analysis',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    if len(args.checkpoints) != len(args.names):
        raise ValueError("Number of checkpoints must match number of names")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each model
    results_dict = {}
    
    for checkpoint_path, model_name in zip(args.checkpoints, args.names):
        logger.info(f"\nOptimizing thresholds for {model_name}...")
        
        results, predictions, targets = optimize_thresholds_for_model(checkpoint_path, args.config)
        results_dict[model_name] = results
        
        logger.info(f"{model_name}:")
        logger.info(f"  Default: Macro-F1={results['f1_macro_default']:.4f}, Micro-F1={results['f1_micro_default']:.4f}")
        logger.info(f"  Optimal: Macro-F1={results['f1_macro_optimal']:.4f}, Micro-F1={results['f1_micro_optimal']:.4f}")
        logger.info(f"  Improvement: +{results['f1_macro_optimal'] - results['f1_macro_default']:.4f}")
    
    # Save detailed results
    with open(output_dir / 'threshold_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # Generate visualizations
    plot_threshold_comparison(results_dict, output_dir / 'threshold_comparison.png')
    plot_threshold_heatmap(results_dict, output_dir / 'threshold_heatmap.png')
    
    # Generate summary report
    with open(output_dir / 'threshold_report.txt', 'w') as f:
        f.write("Threshold Optimization Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        for model_name, results in results_dict.items():
            f.write(f"\n{model_name}:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Default Thresholds (0.5):\n")
            f.write(f"  Macro-F1: {results['f1_macro_default']:.4f}\n")
            f.write(f"  Micro-F1: {results['f1_micro_default']:.4f}\n")
            f.write(f"\nOptimized Thresholds:\n")
            f.write(f"  Macro-F1: {results['f1_macro_optimal']:.4f} (+{results['f1_macro_optimal'] - results['f1_macro_default']:.4f})\n")
            f.write(f"  Micro-F1: {results['f1_micro_optimal']:.4f} (+{results['f1_micro_optimal'] - results['f1_micro_default']:.4f})\n")
            
            f.write(f"\nPer-Hallmark Improvements:\n")
            df = pd.DataFrame(results['threshold_analysis'])
            top_improvements = df.nlargest(5, 'f1_improvement')[['hallmark_name', 'optimal_threshold', 'f1_improvement']]
            f.write(top_improvements.to_string(index=False))
            f.write("\n")
    
    logger.info(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()