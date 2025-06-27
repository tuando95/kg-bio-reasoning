#!/usr/bin/env python3
"""
Simple training script without AMP to test
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
import yaml
import logging
from tqdm import tqdm
import time
from pathlib import Path
import json

from src.data import HoCDataModule
from src.models import BioKGBioBERT
from src.evaluation import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_simple():
    """Simple training loop without complications"""
    
    # Load config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Simplify config
    config['model']['use_knowledge_graph'] = False
    config['model']['use_bio_attention'] = False
    config['model']['loss_weights'] = {
        'hallmark_loss': 1.0,
        'pathway_loss': 0.0,
        'consistency_loss': 0.0
    }
    config['training']['batch_size'] = 32
    config['training']['max_epochs'] = 1
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data
    logger.info("Setting up data...")
    data_module = HoCDataModule(config)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Model
    logger.info("Initializing model...")
    model = BioKGBioBERT(config['model'])
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {num_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['optimizer']['weight_decay'])
    )
    
    # Evaluator
    evaluator = Evaluator(config)
    
    # Training
    logger.info("Starting training...")
    model.train()
    
    train_losses = []
    start_time = time.time()
    
    progress = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(progress):
        # Move batch to device
        model_inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'labels': batch['labels'].to(device)
        }
        
        # Forward pass
        outputs = model(**model_inputs)
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Track loss
        train_losses.append(loss.item())
        
        # Update progress
        progress.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Log periodically
        if batch_idx % 50 == 0:
            avg_loss = sum(train_losses[-50:]) / len(train_losses[-50:])
            logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {avg_loss:.4f}")
    
    train_time = time.time() - start_time
    avg_train_loss = sum(train_losses) / len(train_losses)
    logger.info(f"Training complete in {train_time:.2f}s")
    logger.info(f"Average train loss: {avg_train_loss:.4f}")
    
    # Validation
    logger.info("Running validation...")
    model.eval()
    
    val_losses = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            model_inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            
            outputs = model(**model_inputs)
            loss = outputs['loss']
            
            val_losses.append(loss.item())
            all_preds.append(torch.sigmoid(outputs['logits']).cpu())
            all_labels.append(batch['labels'].cpu())
    
    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    metrics = evaluator.compute_metrics(all_preds, all_labels)
    avg_val_loss = sum(val_losses) / len(val_losses)
    
    logger.info(f"Validation loss: {avg_val_loss:.4f}")
    logger.info(f"Validation metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
    
    # Save results
    results = {
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'val_metrics': metrics,
        'train_time': train_time,
        'config': config
    }
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'simple_training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Results saved!")
    
    return results


if __name__ == "__main__":
    results = train_simple()
    print(f"\nFinal F1-Macro: {results['val_metrics'].get('f1_macro', 0):.4f}")