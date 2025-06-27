#!/usr/bin/env python3
"""
Simple experiment runner to test the model training
"""

import yaml
import logging
import torch
from pathlib import Path
import json

from src.train import train_model
from src.data import HoCDataModule
from src.models import BioKGBioBERT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_simple_baseline():
    """Run a simple baseline experiment to test the setup"""
    
    # Load base config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for baseline experiment (BioBERT only)
    config['experiment']['name'] = 'simple_baseline_test'
    config['training']['batch_size'] = 8  # Smaller batch for testing
    config['training']['max_epochs'] = 2  # Just 2 epochs for testing
    
    # Disable knowledge graph for baseline
    config['model']['use_knowledge_graph'] = False
    config['model']['use_bio_attention'] = False
    
    # Simplified loss weights
    config['training']['loss_weights'] = {
        'hallmark_loss': 1.0,
        'pathway_loss': 0.0,
        'consistency_loss': 0.0
    }
    
    logger.info("Running simple baseline experiment...")
    
    try:
        # Initialize data module
        logger.info("Initializing data module...")
        data_module = HoCDataModule(config)
        
        # Initialize model
        logger.info("Initializing model...")
        model = BioKGBioBERT(config['model'])
        
        # Check model structure
        logger.info(f"Model initialized. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Get a sample batch
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        
        logger.info(f"Sample batch keys: {batch.keys()}")
        logger.info(f"Input shape: {batch['input_ids'].shape}")
        logger.info(f"Labels shape: {batch['labels'].shape}")
        
        # Try forward pass
        logger.info("Testing forward pass...")
        model.to(config['experiment']['device'])
        model.eval()
        
        with torch.no_grad():
            # Move batch to device
            batch_device = {k: v.to(config['experiment']['device']) if torch.is_tensor(v) else v 
                           for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch_device['input_ids'],
                attention_mask=batch_device['attention_mask'],
                knowledge_graphs=batch_device.get('knowledge_graphs'),
                labels=batch_device['labels']
            )
            
            logger.info(f"Forward pass successful!")
            logger.info(f"Loss: {outputs['loss'].item():.4f}")
            logger.info(f"Logits shape: {outputs['logits'].shape}")
        
        # Now run actual training
        logger.info("\nStarting training...")
        results = train_model(config)
        
        logger.info("\nTraining complete!")
        logger.info(f"Final results: {json.dumps(results, indent=2)}")
        
        # Save results
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / 'simple_baseline_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
    except Exception as e:
        logger.error(f"Error in experiment: {e}", exc_info=True)
        raise


def run_kg_enabled_experiment():
    """Run experiment with KG enabled"""
    
    # Load base config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for KG experiment
    config['experiment']['name'] = 'simple_kg_test'
    config['training']['batch_size'] = 4  # Smaller batch due to KG overhead
    config['training']['max_epochs'] = 2
    
    # Enable knowledge graph
    config['model']['use_knowledge_graph'] = True
    config['model']['use_bio_attention'] = True
    
    # Enable all loss components
    config['training']['loss_weights'] = {
        'hallmark_loss': 1.0,
        'pathway_loss': 0.25,
        'consistency_loss': 0.1
    }
    
    logger.info("Running KG-enabled experiment...")
    
    try:
        results = train_model(config)
        
        logger.info("\nTraining complete!")
        logger.info(f"Final results: {json.dumps(results, indent=2)}")
        
        # Save results
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / 'simple_kg_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error in KG experiment: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # First run baseline
    logger.info("="*80)
    logger.info("RUNNING BASELINE EXPERIMENT")
    logger.info("="*80)
    run_simple_baseline()
    
    # Then run with KG
    logger.info("\n" + "="*80)
    logger.info("RUNNING KG-ENABLED EXPERIMENT")
    logger.info("="*80)
    run_kg_enabled_experiment()