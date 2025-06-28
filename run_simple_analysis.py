"""
Simple Analysis Script for Trained BioKG-BioBERT Model

This script loads a trained model checkpoint and performs analysis including:
- Model evaluation on test set
- Per-hallmark performance breakdown
- Error analysis
- Attention visualization (if applicable)
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
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_curve

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


def load_model_and_config(checkpoint_path: str, config_path: str = 'configs/default_config.yaml'):
    """Load model from checkpoint and configuration."""
    
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
    
    # Set loss weights based on what's in the checkpoint
    if has_pathway_classifier and has_consistency_predictor:
        model_config['loss_weights'] = config['training']['loss_weights']
    else:
        # Disable auxiliary tasks if not in checkpoint
        model_config['loss_weights'] = {
            'hallmark_loss': 1.0,
            'pathway_loss': 0.0,
            'consistency_loss': 0.0
        }
    
    model = BioKGBioBERT(model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    return model, config, checkpoint


def evaluate_model(model, config):
    """Evaluate model on test set."""
    
    # Initialize data module
    data_module = HoCDataModule(config)
    data_module.setup()
    
    # Get test dataloader
    test_loader = data_module.test_dataloader()
    
    # Initialize evaluator
    evaluator = Evaluator(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_predictions = []
    all_targets = []
    
    logger.info("Running evaluation on test set...")
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            batch_device = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_device[key] = value.to(device)
                elif key == 'graph_data':
                    batch_device[key] = value.to(device) if value is not None else None
                elif key == 'biological_context':
                    # Move biological context tensors
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
    
    # Compute metrics
    metrics = evaluator.compute_metrics(all_predictions, all_targets)
    
    return metrics, all_predictions, all_targets


def analyze_per_hallmark_performance(predictions, targets, threshold=0.5):
    """Analyze performance for each cancer hallmark."""
    
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
    
    pred_binary = (predictions >= threshold).int()
    
    results = []
    
    for i in range(11):
        tp = ((pred_binary[:, i] == 1) & (targets[:, i] == 1)).sum().item()
        tn = ((pred_binary[:, i] == 0) & (targets[:, i] == 0)).sum().item()
        fp = ((pred_binary[:, i] == 1) & (targets[:, i] == 0)).sum().item()
        fn = ((pred_binary[:, i] == 0) & (targets[:, i] == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'hallmark': hallmark_names[i],
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': (targets[:, i] == 1).sum().item()
        })
    
    return pd.DataFrame(results)


def find_optimal_thresholds(predictions, targets):
    """Find optimal threshold for each hallmark."""
    
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
    
    return optimal_thresholds, pd.DataFrame(threshold_analysis)


def plot_confusion_matrices(predictions, targets, threshold=0.5, save_path='confusion_matrices.png', optimal_thresholds=None):
    """Plot confusion matrices for each hallmark."""
    
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
    
    targets_np = targets.numpy()
    
    # Use optimal thresholds if provided, otherwise use default
    if optimal_thresholds is not None:
        pred_binary = np.zeros_like(predictions.numpy())
        for i in range(11):
            pred_binary[:, i] = (predictions[:, i].numpy() >= optimal_thresholds[i]).astype(int)
    else:
        pred_binary = (predictions >= threshold).int().numpy()
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(11):
        cm = confusion_matrix(targets_np[:, i], pred_binary[:, i])
        
        # Add threshold info to title
        if optimal_thresholds is not None:
            title = f'{hallmark_names[i]}\n(threshold={optimal_thresholds[i]:.3f})'
        else:
            title = f'{hallmark_names[i]}\n(threshold={threshold:.3f})'
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(title, fontsize=9)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Hide the 12th subplot
    axes[11].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrices saved to {save_path}")


def main():
    """Main analysis function."""
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Analyze trained BioKG-BioBERT model")
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/simple_kg_test/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, 
                       default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, 
                       default='analysis_results',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and config
    model, config, checkpoint = load_model_and_config(args.checkpoint, args.config)
    
    # Print checkpoint info
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")
    
    # Evaluate model
    metrics, predictions, targets = evaluate_model(model, config)
    
    # Print overall metrics
    logger.info("\n" + "="*80)
    logger.info("OVERALL TEST SET METRICS")
    logger.info("="*80)
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{metric}: {value:.4f}")
    
    # Save metrics
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Analyze per-hallmark performance
    hallmark_df = analyze_per_hallmark_performance(predictions, targets)
    
    logger.info("\n" + "="*80)
    logger.info("PER-HALLMARK PERFORMANCE")
    logger.info("="*80)
    print(hallmark_df.to_string(index=False))
    
    # Save per-hallmark results
    hallmark_df.to_csv(output_dir / 'per_hallmark_performance.csv', index=False)
    
    # Find optimal thresholds
    logger.info("\n" + "="*80)
    logger.info("THRESHOLD OPTIMIZATION")
    logger.info("="*80)
    optimal_thresholds, threshold_df = find_optimal_thresholds(predictions, targets)
    
    # Print threshold analysis
    for _, row in threshold_df.iterrows():
        logger.info(f"{row['hallmark_name']:<40} "
                   f"threshold={row['optimal_threshold']:.3f} "
                   f"(F1: {row['default_f1']:.3f} -> {row['optimal_f1']:.3f}, "
                   f"+{row['f1_improvement']:.3f})")
    
    # Save threshold analysis
    threshold_df.to_csv(output_dir / 'threshold_analysis.csv', index=False)
    
    # Save optimal thresholds
    with open(output_dir / 'optimal_thresholds.json', 'w') as f:
        json.dump({
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
            }
        }, f, indent=2)
    
    # Analyze performance with optimal thresholds
    pred_binary_optimal = np.zeros_like(predictions.numpy())
    for i in range(11):
        pred_binary_optimal[:, i] = (predictions[:, i].numpy() >= optimal_thresholds[i]).astype(int)
    
    pred_binary_default = (predictions >= 0.5).int().numpy()
    
    f1_micro_optimal = f1_score(targets.numpy(), pred_binary_optimal, average='micro')
    f1_macro_optimal = f1_score(targets.numpy(), pred_binary_optimal, average='macro')
    f1_micro_default = f1_score(targets.numpy(), pred_binary_default, average='micro')
    f1_macro_default = f1_score(targets.numpy(), pred_binary_default, average='macro')
    
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE WITH OPTIMAL THRESHOLDS")
    logger.info("="*80)
    logger.info(f"Micro-F1: {f1_micro_default:.4f} -> {f1_micro_optimal:.4f} "
               f"(+{f1_micro_optimal - f1_micro_default:.4f})")
    logger.info(f"Macro-F1: {f1_macro_default:.4f} -> {f1_macro_optimal:.4f} "
               f"(+{f1_macro_optimal - f1_macro_default:.4f})")
    
    # Plot confusion matrices with default thresholds
    plot_confusion_matrices(predictions, targets, save_path=output_dir / 'confusion_matrices_default.png')
    
    # Plot confusion matrices with optimal thresholds
    plot_confusion_matrices(predictions, targets, save_path=output_dir / 'confusion_matrices_optimal.png', 
                          optimal_thresholds=optimal_thresholds)
    
    # Generate summary report
    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("BioKG-BioBERT Model Analysis Report\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model checkpoint: {args.checkpoint}\n")
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-"*40 + "\n")
        for metric, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\n\nPER-HALLMARK PERFORMANCE\n")
        f.write("-"*40 + "\n")
        f.write(hallmark_df.to_string(index=False))
        
        f.write("\n\nTOP PERFORMING HALLMARKS\n")
        f.write("-"*40 + "\n")
        top_hallmarks = hallmark_df.nlargest(5, 'f1')
        f.write(top_hallmarks.to_string(index=False))
        
        f.write("\n\nBOTTOM PERFORMING HALLMARKS\n")
        f.write("-"*40 + "\n")
        bottom_hallmarks = hallmark_df.nsmallest(5, 'f1')
        f.write(bottom_hallmarks.to_string(index=False))
        
        f.write("\n\n\nTHRESHOLD OPTIMIZATION RESULTS\n")
        f.write("="*40 + "\n")
        f.write(f"Micro-F1: {f1_micro_default:.4f} -> {f1_micro_optimal:.4f} "
               f"(+{f1_micro_optimal - f1_micro_default:.4f})\n")
        f.write(f"Macro-F1: {f1_macro_default:.4f} -> {f1_macro_optimal:.4f} "
               f"(+{f1_macro_optimal - f1_macro_default:.4f})\n")
        
        f.write("\n\nOPTIMAL THRESHOLDS BY HALLMARK\n")
        f.write("-"*40 + "\n")
        for _, row in threshold_df.iterrows():
            f.write(f"{row['hallmark_name']:<40} {row['optimal_threshold']:.3f} "
                   f"(F1: {row['default_f1']:.3f} -> {row['optimal_f1']:.3f})\n")
    
    logger.info(f"\nAnalysis complete! Results saved to {output_dir}")
    logger.info(f"Report available at: {report_path}")


if __name__ == "__main__":
    main()