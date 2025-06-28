"""
Threshold Optimization for BioKG-BioBERT Model

This script finds optimal classification thresholds for each cancer hallmark
to maximize F1 scores.
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
from src.evaluation import Evaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_data(checkpoint_path: str, config_path: str = 'configs/default_config.yaml'):
    """Load model and get validation set predictions."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
    
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Initialize data module
    data_module = HoCDataModule(config)
    data_module.setup()
    
    # Get validation dataloader
    val_loader = data_module.val_dataloader()
    
    all_predictions = []
    all_targets = []
    
    logger.info("Getting predictions on validation set...")
    
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
            
            # Forward pass
            outputs = model(**batch_device)
            
            predictions = torch.sigmoid(outputs['logits'])
            all_predictions.append(predictions.cpu())
            all_targets.append(batch_device['labels'].cpu())
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return all_predictions, all_targets, model, config


def find_optimal_thresholds(predictions, targets):
    """Find optimal threshold for each hallmark using validation set."""
    
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
    
    optimal_thresholds = {}
    threshold_analysis = []
    
    for i in range(11):
        y_true = targets[:, i].numpy()
        y_scores = predictions[:, i].numpy()
        
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
        
        logger.info(f"{hallmark_names[i]}: optimal threshold = {best_threshold:.3f} "
                   f"(F1: {default_f1:.4f} -> {best_f1:.4f}, improvement: {best_f1 - default_f1:.4f})")
    
    return optimal_thresholds, pd.DataFrame(threshold_analysis)


def evaluate_with_optimal_thresholds(predictions, targets, thresholds):
    """Evaluate performance using optimal thresholds."""
    
    # Apply optimal thresholds
    pred_binary = torch.zeros_like(predictions)
    for i in range(11):
        pred_binary[:, i] = (predictions[:, i] >= thresholds[i]).float()
    
    # Compute metrics
    evaluator = Evaluator({})
    metrics_optimal = evaluator.compute_metrics(predictions, targets)
    
    # Also compute with binary predictions using optimal thresholds
    y_true = targets.numpy()
    y_pred_optimal = pred_binary.numpy()
    y_pred_default = (predictions >= 0.5).float().numpy()
    
    # Calculate improvement
    f1_micro_optimal = f1_score(y_true, y_pred_optimal, average='micro')
    f1_macro_optimal = f1_score(y_true, y_pred_optimal, average='macro')
    f1_micro_default = f1_score(y_true, y_pred_default, average='micro')
    f1_macro_default = f1_score(y_true, y_pred_default, average='macro')
    
    improvement = {
        'f1_micro_improvement': f1_micro_optimal - f1_micro_default,
        'f1_macro_improvement': f1_macro_optimal - f1_macro_default,
        'f1_micro_optimal': f1_micro_optimal,
        'f1_macro_optimal': f1_macro_optimal,
        'f1_micro_default': f1_micro_default,
        'f1_macro_default': f1_macro_default
    }
    
    return improvement


def plot_threshold_analysis(threshold_df, save_path='threshold_analysis.png'):
    """Plot threshold optimization results."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Optimal thresholds by hallmark
    threshold_df = threshold_df.sort_values('optimal_threshold')
    ax1.barh(threshold_df['hallmark_name'], threshold_df['optimal_threshold'])
    ax1.axvline(x=0.5, color='red', linestyle='--', label='Default threshold')
    ax1.set_xlabel('Optimal Threshold')
    ax1.set_title('Optimal Classification Thresholds by Cancer Hallmark')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: F1 improvement
    threshold_df = threshold_df.sort_values('f1_improvement')
    colors = ['red' if x < 0 else 'green' for x in threshold_df['f1_improvement']]
    ax2.barh(threshold_df['hallmark_name'], threshold_df['f1_improvement'], color=colors)
    ax2.set_xlabel('F1 Score Improvement')
    ax2.set_title('F1 Score Improvement with Optimal Thresholds')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Threshold analysis plot saved to {save_path}")


def main():
    """Main threshold optimization function."""
    
    import argparse
    parser = argparse.ArgumentParser(description="Optimize thresholds for BioKG-BioBERT")
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/simple_kg_test/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, 
                       default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, 
                       default='threshold_optimization',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and get predictions
    predictions, targets, model, config = load_model_and_data(args.checkpoint, args.config)
    
    # Find optimal thresholds
    logger.info("\nFinding optimal thresholds on validation set...")
    optimal_thresholds, threshold_df = find_optimal_thresholds(predictions, targets)
    
    # Evaluate with optimal thresholds
    improvement = evaluate_with_optimal_thresholds(predictions, targets, optimal_thresholds)
    
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE IMPROVEMENT WITH OPTIMAL THRESHOLDS")
    logger.info("="*80)
    logger.info(f"Micro-F1: {improvement['f1_micro_default']:.4f} -> "
               f"{improvement['f1_micro_optimal']:.4f} "
               f"(+{improvement['f1_micro_improvement']:.4f})")
    logger.info(f"Macro-F1: {improvement['f1_macro_default']:.4f} -> "
               f"{improvement['f1_macro_optimal']:.4f} "
               f"(+{improvement['f1_macro_improvement']:.4f})")
    
    # Save optimal thresholds
    thresholds_dict = {
        'thresholds': optimal_thresholds,
        'hallmark_names': {
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
        },
        'performance_improvement': improvement
    }
    
    with open(output_dir / 'optimal_thresholds.json', 'w') as f:
        json.dump(thresholds_dict, f, indent=2)
    
    # Save threshold analysis
    threshold_df.to_csv(output_dir / 'threshold_analysis.csv', index=False)
    
    # Plot results
    plot_threshold_analysis(threshold_df, output_dir / 'threshold_analysis.png')
    
    # Generate report
    report_path = output_dir / 'threshold_optimization_report.txt'
    with open(report_path, 'w') as f:
        f.write("Threshold Optimization Report\n")
        f.write("="*80 + "\n\n")
        
        f.write("OPTIMAL THRESHOLDS BY HALLMARK\n")
        f.write("-"*40 + "\n")
        for _, row in threshold_df.iterrows():
            f.write(f"{row['hallmark_name']:<40} {row['optimal_threshold']:.3f} "
                   f"(F1: {row['default_f1']:.3f} -> {row['optimal_f1']:.3f})\n")
        
        f.write("\n\nOVERALL PERFORMANCE IMPROVEMENT\n")
        f.write("-"*40 + "\n")
        f.write(f"Micro-F1: {improvement['f1_micro_default']:.4f} -> "
               f"{improvement['f1_micro_optimal']:.4f} "
               f"(+{improvement['f1_micro_improvement']:.4f})\n")
        f.write(f"Macro-F1: {improvement['f1_macro_default']:.4f} -> "
               f"{improvement['f1_macro_optimal']:.4f} "
               f"(+{improvement['f1_macro_improvement']:.4f})\n")
        
        f.write("\n\nRECOMMENDATIONS\n")
        f.write("-"*40 + "\n")
        f.write("1. Use these optimal thresholds for production deployment\n")
        f.write("2. Consider per-hallmark thresholds for best performance\n")
        f.write("3. Re-optimize thresholds periodically as data distribution changes\n")
    
    logger.info(f"\nThreshold optimization complete!")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Report available at: {report_path}")


if __name__ == "__main__":
    main()