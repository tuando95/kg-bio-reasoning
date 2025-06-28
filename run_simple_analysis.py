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
from sklearn.metrics import confusion_matrix, classification_report

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
    model_config['loss_weights'] = config['training']['loss_weights']
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


def plot_confusion_matrices(predictions, targets, threshold=0.5, save_path='confusion_matrices.png'):
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
    
    pred_binary = (predictions >= threshold).int().numpy()
    targets_np = targets.numpy()
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(11):
        cm = confusion_matrix(targets_np[:, i], pred_binary[:, i])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{hallmark_names[i]}', fontsize=10)
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
    
    # Plot confusion matrices
    plot_confusion_matrices(predictions, targets, save_path=output_dir / 'confusion_matrices.png')
    
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
    
    logger.info(f"\nAnalysis complete! Results saved to {output_dir}")
    logger.info(f"Report available at: {report_path}")


if __name__ == "__main__":
    main()