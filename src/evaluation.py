"""
Evaluation Metrics and Analysis Framework

This module implements comprehensive evaluation metrics for multi-label
cancer hallmark classification with biological consistency analysis.
"""

import torch
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    hamming_loss, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive evaluation framework for BioKG-BioBERT.
    
    Computes:
    - Multi-label classification metrics
    - Per-hallmark performance analysis
    - Biological consistency metrics
    - Pathway prediction accuracy
    """
    
    # Hallmark names for reporting
    HALLMARK_NAMES = {
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
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.threshold = 0.5  # Classification threshold
        
        # Biological constraints
        self.incompatible_pairs = [
            (0, 9),  # Evading growth suppressors vs Sustaining proliferative signaling
        ]
        
        self.synergistic_pairs = [
            (8, 3),  # Inducing angiogenesis & Cellular energetics
            (5, 1),  # Invasion/metastasis & Inflammation
        ]
    
    def compute_metrics(self, 
                       predictions: torch.Tensor,
                       targets: torch.Tensor,
                       pathway_predictions: Optional[torch.Tensor] = None,
                       pathway_targets: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            predictions: Predicted probabilities [batch_size, num_labels]
            targets: Ground truth labels [batch_size, num_labels]
            pathway_predictions: Pathway activation predictions
            pathway_targets: Ground truth pathway activations
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy
        pred_probs = predictions.cpu().numpy()
        true_labels = targets.cpu().numpy()
        
        # Binary predictions
        pred_labels = (pred_probs >= self.threshold).astype(int)
        
        # Basic multi-label metrics
        metrics = {
            'f1_micro': f1_score(true_labels, pred_labels, average='micro'),
            'f1_macro': f1_score(true_labels, pred_labels, average='macro'),
            'f1_weighted': f1_score(true_labels, pred_labels, average='weighted'),
            'precision_micro': precision_score(true_labels, pred_labels, average='micro'),
            'precision_macro': precision_score(true_labels, pred_labels, average='macro'),
            'recall_micro': recall_score(true_labels, pred_labels, average='micro'),
            'recall_macro': recall_score(true_labels, pred_labels, average='macro'),
            'hamming_loss': hamming_loss(true_labels, pred_labels),
            'exact_match_ratio': self._exact_match_ratio(true_labels, pred_labels),
            'subset_accuracy': self._subset_accuracy(true_labels, pred_labels)
        }
        
        # ROC-AUC scores (if applicable)
        try:
            metrics['auc_micro'] = roc_auc_score(true_labels, pred_probs, average='micro')
            metrics['auc_macro'] = roc_auc_score(true_labels, pred_probs, average='macro')
        except:
            logger.warning("Could not compute AUC scores")
        
        # Average precision
        try:
            metrics['ap_micro'] = average_precision_score(true_labels, pred_probs, average='micro')
            metrics['ap_macro'] = average_precision_score(true_labels, pred_probs, average='macro')
        except:
            logger.warning("Could not compute average precision scores")
        
        # Per-hallmark metrics
        per_hallmark_metrics = self._compute_per_hallmark_metrics(true_labels, pred_labels, pred_probs)
        metrics.update(per_hallmark_metrics)
        
        # Biological consistency metrics
        bio_metrics = self._compute_biological_consistency_metrics(pred_labels, true_labels)
        metrics.update(bio_metrics)
        
        # Pathway metrics (if provided)
        if pathway_predictions is not None and pathway_targets is not None:
            pathway_metrics = self._compute_pathway_metrics(
                pathway_predictions.cpu().numpy(),
                pathway_targets.cpu().numpy()
            )
            metrics.update(pathway_metrics)
        
        return metrics
    
    def _exact_match_ratio(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute exact match ratio (all labels correct)."""
        return np.mean(np.all(y_true == y_pred, axis=1))
    
    def _subset_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute subset accuracy."""
        return np.mean(np.sum(y_true == y_pred, axis=1) / y_true.shape[1])
    
    def _compute_per_hallmark_metrics(self, 
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    y_prob: np.ndarray) -> Dict[str, float]:
        """Compute metrics for each hallmark."""
        metrics = {}
        
        for i in range(y_true.shape[1]):
            hallmark_name = self.HALLMARK_NAMES[i]
            prefix = f"hallmark_{i}"
            
            # Skip if no positive samples
            if y_true[:, i].sum() == 0:
                continue
            
            metrics[f"{prefix}_f1"] = f1_score(y_true[:, i], y_pred[:, i])
            metrics[f"{prefix}_precision"] = precision_score(y_true[:, i], y_pred[:, i])
            metrics[f"{prefix}_recall"] = recall_score(y_true[:, i], y_pred[:, i])
            
            try:
                metrics[f"{prefix}_auc"] = roc_auc_score(y_true[:, i], y_prob[:, i])
            except:
                pass
        
        return metrics
    
    def _compute_biological_consistency_metrics(self,
                                              y_pred: np.ndarray,
                                              y_true: np.ndarray) -> Dict[str, float]:
        """Compute biological consistency metrics."""
        metrics = {}
        
        # Incompatible hallmark violations
        violations = 0
        total_incompatible = 0
        
        for i, j in self.incompatible_pairs:
            # Count when both incompatible hallmarks are predicted
            both_predicted = np.sum((y_pred[:, i] == 1) & (y_pred[:, j] == 1))
            violations += both_predicted
            total_incompatible += len(y_pred)
        
        metrics['bio_incompatible_violation_rate'] = violations / max(total_incompatible, 1)
        
        # Synergistic hallmark co-occurrence
        synergy_correct = 0
        total_synergy = 0
        
        for i, j in self.synergistic_pairs:
            # When one is true, check if the other is also predicted
            mask_i = y_true[:, i] == 1
            mask_j = y_true[:, j] == 1
            
            if mask_i.sum() > 0:
                synergy_correct += np.sum(y_pred[mask_i, j] == 1)
                total_synergy += mask_i.sum()
            
            if mask_j.sum() > 0:
                synergy_correct += np.sum(y_pred[mask_j, i] == 1)
                total_synergy += mask_j.sum()
        
        metrics['bio_synergy_capture_rate'] = synergy_correct / max(total_synergy, 1)
        
        # Calculate random baseline for comparison
        # Random model would predict each hallmark with ~0.5 probability
        # So co-occurrence would be ~0.25 for any pair
        metrics['bio_synergy_capture_rate_random_baseline'] = 0.25
        metrics['bio_synergy_capture_rate_improvement'] = (
            metrics['bio_synergy_capture_rate'] - 0.25
        )
        
        # Overall biological plausibility score
        metrics['bio_plausibility_score'] = (
            (1 - metrics['bio_incompatible_violation_rate']) * 0.7 +
            metrics['bio_synergy_capture_rate'] * 0.3
        )
        
        # Add baseline comparisons for plausibility
        # Random model: ~50% avoiding violations * 0.7 + 25% synergy * 0.3 = 0.425
        metrics['bio_plausibility_score_random_baseline'] = 0.425
        metrics['bio_plausibility_score_improvement'] = (
            metrics['bio_plausibility_score'] - 0.425
        )
        
        return metrics
    
    def _compute_pathway_metrics(self,
                               pathway_pred: np.ndarray,
                               pathway_true: np.ndarray) -> Dict[str, float]:
        """Compute pathway prediction metrics."""
        # Binary predictions for pathways
        pathway_pred_binary = (pathway_pred >= self.threshold).astype(int)
        
        metrics = {
            'pathway_f1_micro': f1_score(pathway_true, pathway_pred_binary, average='micro'),
            'pathway_f1_macro': f1_score(pathway_true, pathway_pred_binary, average='macro'),
            'pathway_precision': precision_score(pathway_true, pathway_pred_binary, average='micro'),
            'pathway_recall': recall_score(pathway_true, pathway_pred_binary, average='micro'),
        }
        
        try:
            metrics['pathway_auc'] = roc_auc_score(pathway_true, pathway_pred, average='micro')
        except:
            pass
        
        return metrics
    
    def generate_classification_report(self,
                                     predictions: torch.Tensor,
                                     targets: torch.Tensor,
                                     output_path: Optional[Path] = None) -> str:
        """
        Generate detailed classification report.
        
        Args:
            predictions: Predicted probabilities
            targets: Ground truth labels
            output_path: Optional path to save report
            
        Returns:
            Classification report as string
        """
        pred_labels = (predictions.cpu().numpy() >= self.threshold).astype(int)
        true_labels = targets.cpu().numpy()
        
        # Generate report
        report = "BioKG-BioBERT Classification Report\n"
        report += "=" * 80 + "\n\n"
        
        # Overall metrics
        metrics = self.compute_metrics(predictions, targets)
        report += "Overall Performance:\n"
        report += f"  F1-Micro: {metrics['f1_micro']:.4f}\n"
        report += f"  F1-Macro: {metrics['f1_macro']:.4f}\n"
        report += f"  Hamming Loss: {metrics['hamming_loss']:.4f}\n"
        report += f"  Exact Match Ratio: {metrics['exact_match_ratio']:.4f}\n"
        report += f"  Biological Plausibility Score: {metrics.get('bio_plausibility_score', 0):.4f}\n"
        report += "\n"
        
        # Per-hallmark report
        report += "Per-Hallmark Performance:\n"
        report += "-" * 60 + "\n"
        
        for i in range(len(self.HALLMARK_NAMES)):
            if i == 7:  # Skip "None" class
                continue
                
            hallmark_name = self.HALLMARK_NAMES[i]
            support = true_labels[:, i].sum()
            
            if support > 0:
                precision = precision_score(true_labels[:, i], pred_labels[:, i])
                recall = recall_score(true_labels[:, i], pred_labels[:, i])
                f1 = f1_score(true_labels[:, i], pred_labels[:, i])
                
                report += f"{hallmark_name:40} "
                report += f"P: {precision:.3f}  R: {recall:.3f}  F1: {f1:.3f}  "
                report += f"Support: {support}\n"
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report
    
    def plot_confusion_matrices(self,
                              predictions: torch.Tensor,
                              targets: torch.Tensor,
                              output_dir: Path):
        """Plot confusion matrices for each hallmark."""
        pred_labels = (predictions.cpu().numpy() >= self.threshold).astype(int)
        true_labels = targets.cpu().numpy()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subplots
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, (hallmark_id, hallmark_name) in enumerate(self.HALLMARK_NAMES.items()):
            if i >= 12:  # Only 11 hallmarks + None
                break
                
            # Compute confusion matrix
            cm = confusion_matrix(true_labels[:, hallmark_id], pred_labels[:, hallmark_id])
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f"{hallmark_name}")
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
        
        # Remove empty subplots
        for i in range(11, 12):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_prediction_patterns(self,
                                  predictions: torch.Tensor,
                                  targets: torch.Tensor) -> pd.DataFrame:
        """Analyze co-occurrence patterns in predictions."""
        pred_labels = (predictions.cpu().numpy() >= self.threshold).astype(int)
        
        # Compute co-occurrence matrix
        n_hallmarks = pred_labels.shape[1]
        co_occurrence = np.zeros((n_hallmarks, n_hallmarks))
        
        for i in range(n_hallmarks):
            for j in range(n_hallmarks):
                if i != j:
                    # P(j=1 | i=1)
                    mask = pred_labels[:, i] == 1
                    if mask.sum() > 0:
                        co_occurrence[i, j] = pred_labels[mask, j].mean()
        
        # Create DataFrame
        hallmark_names = [self.HALLMARK_NAMES[i] for i in range(n_hallmarks)]
        df = pd.DataFrame(co_occurrence, index=hallmark_names, columns=hallmark_names)
        
        return df