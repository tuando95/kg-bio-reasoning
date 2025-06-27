#!/usr/bin/env python3
"""
Main entry point for BioKG-BioBERT training and evaluation.

Usage:
    python main.py --config configs/default_config.yaml --mode train
    python main.py --config configs/default_config.yaml --mode evaluate --checkpoint checkpoints/best.pt
"""

import argparse
import logging
import sys
import torch
import yaml
from pathlib import Path

from src.train import train_model
from src.models import BioKGBioBERT
from src.data import HoCDataModule
from src.evaluation import Evaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('biokg_biobert.log')
    ]
)
logger = logging.getLogger(__name__)


def train(config_path: str):
    """Train BioKG-BioBERT model."""
    logger.info(f"Starting training with config: {config_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train model
    results = train_model(config)
    
    logger.info("Training completed!")
    logger.info(f"Final test metrics: {results}")
    
    return results


def evaluate(config_path: str, checkpoint_path: str):
    """Evaluate trained model."""
    logger.info(f"Evaluating model from checkpoint: {checkpoint_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = BioKGBioBERT(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device(config['experiment']['device'])
    model.to(device)
    model.eval()
    
    # Initialize data module
    data_module = HoCDataModule(config)
    data_module.setup()
    
    # Initialize evaluator
    evaluator = Evaluator(config)
    
    # Evaluate on test set
    test_dataloader = data_module.test_dataloader()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Collect predictions
            all_predictions.append(torch.sigmoid(outputs['logits']))
            all_targets.append(batch['labels'])
    
    # Concatenate all predictions
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = evaluator.compute_metrics(predictions, targets)
    
    # Generate report
    report_dir = Path('evaluation_results')
    report_dir.mkdir(exist_ok=True)
    
    report = evaluator.generate_classification_report(
        predictions, targets,
        output_path=report_dir / 'classification_report.txt'
    )
    
    # Plot confusion matrices
    evaluator.plot_confusion_matrices(
        predictions, targets,
        output_dir=report_dir
    )
    
    logger.info("Evaluation completed!")
    logger.info(f"Test metrics: {metrics}")
    print("\n" + report)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="BioKG-BioBERT Training and Evaluation")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], required=True,
                       help='Mode: train or evaluate')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args.config)
    elif args.mode == 'evaluate':
        if not args.checkpoint:
            parser.error("--checkpoint required for evaluation mode")
        evaluate(args.config, args.checkpoint)


if __name__ == "__main__":
    main()