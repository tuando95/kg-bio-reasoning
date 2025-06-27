"""
Training Module for BioKG-BioBERT

This module implements the training loop, optimization, and experiment tracking
for the BioKG-BioBERT model on cancer hallmarks classification.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from collections import defaultdict

from .models import BioKGBioBERT
from .data import HoCDataModule
from .evaluation import Evaluator

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for BioKG-BioBERT model.
    
    Handles:
    - Training loop with gradient accumulation
    - Multi-task learning
    - Early stopping
    - Checkpointing
    - Logging and visualization
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer.
        
        Args:
            config: Complete configuration dictionary
        """
        self.config = config
        self.device = torch.device(config['experiment']['device'])
        
        # Create experiment directory
        self.experiment_name = config['experiment']['name']
        self.experiment_dir = Path(config['experiment']['results_dir']) / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpointing
        self.checkpoint_dir = Path(config['experiment']['checkpoint_dir']) / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.log_dir = Path(config['experiment']['log_dir']) / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(self.log_dir / 'tensorboard')
        
        # Training parameters
        self.max_epochs = config['training']['max_epochs']
        self.gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
        self.max_grad_norm = config['training']['max_grad_norm']
        
        # Early stopping
        self.early_stopping_patience = config['training']['early_stopping']['patience']
        self.early_stopping_min_delta = config['training']['early_stopping']['min_delta']
        self.early_stopping_metric = config['training']['early_stopping']['metric']
        
        # Initialize components
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_evaluator()
        
        # Training state
        self.global_step = 0
        self.best_metric = -float('inf')
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized for experiment: {self.experiment_name}")
    
    def _setup_model(self):
        """Initialize model."""
        logger.info("Initializing BioKG-BioBERT model...")
        self.model = BioKGBioBERT(self.config)
        self.model.to(self.device)
        
        # Multi-GPU training
        if self.config['experiment']['num_gpus'] > 1:
            self.model = nn.DataParallel(self.model)
        
        # Mixed precision training
        self.use_amp = self.config['experiment']['mixed_precision']
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _setup_data(self):
        """Initialize data module."""
        logger.info("Setting up data module...")
        self.data_module = HoCDataModule(self.config)
        self.data_module.setup()
    
    def _setup_optimizer(self):
        """Initialize optimizer and scheduler."""
        # Optimizer
        optimizer_config = self.config['training']['optimizer']
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate']),
            weight_decay=float(optimizer_config['weight_decay']),
            eps=float(optimizer_config['eps'])
        )
        
        # Learning rate scheduler
        scheduler_config = self.config['training']['scheduler']
        num_training_steps = (
            len(self.data_module.train_dataloader()) // 
            self.gradient_accumulation_steps * 
            self.max_epochs
        )
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=scheduler_config['num_warmup_steps']
        )
        
        # Main scheduler (cosine)
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - scheduler_config['num_warmup_steps'],
            eta_min=1e-7
        )
        
        # Combine schedulers
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[scheduler_config['num_warmup_steps']]
        )
    
    def _setup_evaluator(self):
        """Initialize evaluation module."""
        from .evaluation import Evaluator
        self.evaluator = Evaluator(self.config)
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        train_dataloader = self.data_module.train_dataloader()
        val_dataloader = self.data_module.val_dataloader()
        
        for epoch in range(self.max_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            
            # Training phase
            train_metrics = self._train_epoch(train_dataloader, epoch)
            
            # Validation phase
            val_metrics = self._validate(val_dataloader, epoch)
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Early stopping check
            if self._check_early_stopping(val_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            # Save checkpoint
            if val_metrics[self.early_stopping_metric] > self.best_metric:
                self.best_metric = val_metrics[self.early_stopping_metric]
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self._save_checkpoint(epoch, val_metrics, is_best=False)
        
        # Final evaluation on test set
        logger.info("\nRunning final evaluation on test set...")
        test_metrics = self._test()
        
        # Save final results
        self._save_final_results(test_metrics)
        
        logger.info("Training completed!")
    
    def _train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = defaultdict(float)
        predictions = []
        targets = []
        
        progress_bar = tqdm(dataloader, desc=f"Training epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs['loss'] / self.gradient_accumulation_steps
            else:
                outputs = self.model(**batch)
                loss = outputs['loss'] / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Track losses
            epoch_losses['total_loss'] += loss.item() * self.gradient_accumulation_steps
            epoch_losses['hallmark_loss'] += outputs.get('hallmark_loss', loss).item()
            
            if 'pathway_loss' in outputs:
                epoch_losses['pathway_loss'] += outputs['pathway_loss'].item()
            if 'consistency_loss' in outputs:
                epoch_losses['consistency_loss'] += outputs['consistency_loss'].item()
            
            # Collect predictions
            predictions.append(torch.sigmoid(outputs['logits']).detach())
            targets.append(batch['labels'])
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to tensorboard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
        
        # Compute epoch metrics
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        
        metrics = self.evaluator.compute_metrics(predictions, targets)
        
        # Add losses to metrics
        num_batches = len(dataloader)
        for key, value in epoch_losses.items():
            metrics[f'{key}'] = value / num_batches
        
        return metrics
    
    def _validate(self, dataloader, epoch: int) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        val_losses = defaultdict(float)
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)
                
                # Track losses
                val_losses['total_loss'] += outputs['loss'].item()
                
                # Collect predictions
                predictions.append(torch.sigmoid(outputs['logits']).detach())
                targets.append(batch['labels'])
        
        # Compute metrics
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        
        metrics = self.evaluator.compute_metrics(predictions, targets)
        
        # Add losses to metrics
        num_batches = len(dataloader)
        for key, value in val_losses.items():
            metrics[f'{key}'] = value / num_batches
        
        return metrics
    
    def _test(self) -> Dict[str, float]:
        """Evaluate on test set."""
        # Load best checkpoint
        self._load_checkpoint('best.pt')
        
        test_dataloader = self.data_module.test_dataloader()
        test_metrics = self._validate(test_dataloader, -1)
        
        return test_metrics
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to device."""
        moved_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            elif key == 'graph_data':
                # Handle PyTorch Geometric batch
                moved_batch[key] = value.to(self.device)
            elif key == 'biological_context':
                # Move biological context tensors
                moved_context = {}
                for ctx_key, ctx_value in value.items():
                    if isinstance(ctx_value, torch.Tensor):
                        moved_context[ctx_key] = ctx_value.to(self.device)
                    else:
                        moved_context[ctx_key] = ctx_value
                moved_batch[key] = moved_context
            else:
                moved_batch[key] = value
        
        return moved_batch
    
    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        """Check early stopping criteria."""
        current_metric = val_metrics[self.early_stopping_metric]
        
        if current_metric - self.best_metric > self.early_stopping_min_delta:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.early_stopping_patience
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint with {self.early_stopping_metric}: {metrics[self.early_stopping_metric]:.4f}")
    
    def _load_checkpoint(self, checkpoint_name: str):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics to tensorboard and console."""
        # Console logging
        logger.info(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                   f"F1-Macro: {train_metrics['f1_macro']:.4f}, "
                   f"F1-Micro: {train_metrics['f1_micro']:.4f}")
        logger.info(f"Val - Loss: {val_metrics['total_loss']:.4f}, "
                   f"F1-Macro: {val_metrics['f1_macro']:.4f}, "
                   f"F1-Micro: {val_metrics['f1_micro']:.4f}")
        
        # Tensorboard logging
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)
        
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)
    
    def _save_final_results(self, test_metrics: Dict):
        """Save final test results."""
        results = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'test_metrics': test_metrics,
            'best_val_metric': self.best_metric,
            'training_time': time.time()
        }
        
        results_path = self.experiment_dir / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved final results to {results_path}")
        logger.info(f"Test F1-Macro: {test_metrics['f1_macro']:.4f}")
        logger.info(f"Test F1-Micro: {test_metrics['f1_micro']:.4f}")


def train_model(config: Dict) -> Dict[str, float]:
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with final test metrics
    """
    # Set random seeds
    torch.manual_seed(config['experiment']['seed'])
    np.random.seed(config['experiment']['seed'])
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['experiment']['seed'])
    
    # Deterministic behavior
    if config['experiment']['deterministic']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Train model
    trainer.train()
    
    # Return test metrics
    return trainer._test()