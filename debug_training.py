#!/usr/bin/env python3
"""
Debug training issues
"""

import torch
import yaml
import logging
from src.data import HoCDataModule
from src.models import BioKGBioBERT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_loading():
    """Test if data loads properly"""
    logger.info("Testing data loading...")
    
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['training']['batch_size'] = 2
    
    data_module = HoCDataModule(config)
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    
    logger.info("Getting first batch...")
    batch = next(iter(train_loader))
    
    logger.info(f"Batch keys: {batch.keys()}")
    logger.info(f"Input shape: {batch['input_ids'].shape}")
    logger.info(f"Labels shape: {batch['labels'].shape}")
    
    # Check if any values are problematic
    for key, value in batch.items():
        if torch.is_tensor(value):
            logger.info(f"{key}: min={value.min().item()}, max={value.max().item()}, dtype={value.dtype}")
    
    return batch


def test_forward_pass(batch):
    """Test model forward pass"""
    logger.info("\nTesting forward pass...")
    
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Simple config for testing
    config['model']['use_knowledge_graph'] = False
    config['model']['use_bio_attention'] = False
    config['model']['loss_weights'] = {
        'hallmark_loss': 1.0,
        'pathway_loss': 0.0,
        'consistency_loss': 0.0
    }
    
    model = BioKGBioBERT(config['model'])
    model.eval()
    
    # Move to CPU for debugging
    device = 'cpu'
    model = model.to(device)
    
    # Prepare inputs
    model_inputs = {
        'input_ids': batch['input_ids'].to(device),
        'attention_mask': batch['attention_mask'].to(device),
        'labels': batch['labels'].to(device)
    }
    
    logger.info("Running forward pass...")
    with torch.no_grad():
        outputs = model(**model_inputs)
    
    logger.info(f"Forward pass successful!")
    logger.info(f"Loss: {outputs['loss'].item()}")
    logger.info(f"Logits shape: {outputs['logits'].shape}")
    
    return outputs


def test_backward_pass():
    """Test if backward pass works"""
    logger.info("\nTesting backward pass...")
    
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['model']['use_knowledge_graph'] = False
    config['model']['use_bio_attention'] = False
    config['model']['loss_weights'] = {
        'hallmark_loss': 1.0,
        'pathway_loss': 0.0,
        'consistency_loss': 0.0
    }
    
    model = BioKGBioBERT(config['model'])
    model.train()
    
    # Create simple inputs
    batch_size = 2
    seq_len = 128
    
    inputs = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len),
        'labels': torch.randint(0, 2, (batch_size, 11)).float()
    }
    
    logger.info("Forward pass...")
    outputs = model(**inputs)
    loss = outputs['loss']
    
    logger.info(f"Loss: {loss.item()}")
    
    logger.info("Backward pass...")
    loss.backward()
    
    # Check if gradients are computed
    grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()
    
    logger.info(f"Total gradient norm: {grad_norm}")
    
    if grad_norm > 0:
        logger.info("Gradients computed successfully!")
    else:
        logger.error("No gradients computed!")


if __name__ == "__main__":
    try:
        # Test data loading
        batch = test_data_loading()
        
        # Test forward pass
        test_forward_pass(batch)
        
        # Test backward pass
        test_backward_pass()
        
        logger.info("\nAll tests passed!")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)