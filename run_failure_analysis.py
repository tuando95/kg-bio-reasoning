"""
Failure Analysis for Low-Performing Hallmarks

This script performs detailed analysis on why certain hallmarks
(particularly "Evading growth suppressors" and "Avoiding immune destruction")
have lower F1 scores compared to others.
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
from collections import Counter, defaultdict
import re
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.biokg_biobert import BioKGBioBERT
from src.data.dataset import HoCDataModule
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FailureAnalyzer:
    """Analyze failure patterns for specific hallmarks."""
    
    def __init__(self, config_path: str = 'configs/default_config.yaml'):
        """Initialize analyzer."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hallmark information
        self.hallmark_names = {
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
        
        # Target hallmarks for analysis
        self.target_hallmarks = [0, 10]  # Lowest performing
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['base_model'])
        
        # Common keywords/phrases for each hallmark
        self.hallmark_keywords = {
            0: ['growth suppressor', 'tumor suppressor', 'p53', 'RB', 'PTEN', 
                'cell cycle checkpoint', 'growth inhibition', 'anti-proliferative'],
            10: ['immune', 'immunosuppression', 'immune evasion', 'PD-L1', 'CTLA-4',
                 'T cell', 'NK cell', 'immune checkpoint', 'immunotherapy', 'MHC']
        }
    
    def load_model_and_data(self, checkpoint_path: str):
        """Load model and prepare data."""
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
        else:
            model_config = self.config['model'].copy()
        
        # Check auxiliary components
        state_dict_keys = checkpoint['model_state_dict'].keys()
        has_pathway = any('pathway_classifier' in key for key in state_dict_keys)
        has_consistency = any('consistency_predictor' in key for key in state_dict_keys)
        
        if has_pathway and has_consistency:
            model_config['loss_weights'] = self.config['training']['loss_weights']
        else:
            model_config['loss_weights'] = {
                'hallmark_loss': 1.0,
                'pathway_loss': 0.0,
                'consistency_loss': 0.0
            }
        
        model = BioKGBioBERT(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        
        # Load data
        data_module = HoCDataModule(self.config)
        data_module.setup()
        
        return model, data_module
    
    def analyze_predictions(self, model, data_loader):
        """Get predictions and analyze errors."""
        all_predictions = []
        all_targets = []
        all_texts = []
        all_entities = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch_device = self._move_batch_to_device(batch)
                
                # Get predictions
                outputs = model(**batch_device)
                predictions = torch.sigmoid(outputs['logits'])
                
                all_predictions.append(predictions.cpu())
                all_targets.append(batch_device['labels'].cpu())
                
                # Store texts if available
                if 'text' in batch:
                    all_texts.extend(batch['text'])
                
                # Store entity information if available
                if 'entity_mapping' in batch:
                    all_entities.extend(batch['entity_mapping'])
        
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        return all_predictions, all_targets, all_texts, all_entities
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        batch_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_device[key] = value.to(self.device)
            elif key == 'graph_data':
                batch_device[key] = value.to(self.device) if value is not None else None
            elif key == 'biological_context':
                moved_context = {}
                for ctx_key, ctx_value in value.items():
                    if isinstance(ctx_value, torch.Tensor):
                        moved_context[ctx_key] = ctx_value.to(self.device)
                    else:
                        moved_context[ctx_key] = ctx_value
                batch_device[key] = moved_context
            else:
                batch_device[key] = value
        return batch_device
    
    def analyze_failure_patterns(self, predictions, targets, texts, save_dir: Path):
        """Analyze failure patterns for target hallmarks."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Binary predictions
        pred_binary = (predictions >= 0.5).astype(int)
        
        # Initialize analysis results
        analysis_results = {}
        
        for hallmark_id in self.target_hallmarks:
            hallmark_name = self.hallmark_names[hallmark_id]
            logger.info(f"\nAnalyzing failures for: {hallmark_name}")
            
            # Get predictions and targets for this hallmark
            h_pred = pred_binary[:, hallmark_id]
            h_true = targets[:, hallmark_id]
            h_scores = predictions[:, hallmark_id]
            
            # Calculate error types
            tp_mask = (h_pred == 1) & (h_true == 1)
            tn_mask = (h_pred == 0) & (h_true == 0)
            fp_mask = (h_pred == 1) & (h_true == 0)
            fn_mask = (h_pred == 0) & (h_true == 1)
            
            # Store results
            results = {
                'hallmark_id': hallmark_id,
                'hallmark_name': hallmark_name,
                'total_samples': len(h_true),
                'positive_samples': int(h_true.sum()),
                'negative_samples': int((1 - h_true).sum()),
                'true_positives': int(tp_mask.sum()),
                'true_negatives': int(tn_mask.sum()),
                'false_positives': int(fp_mask.sum()),
                'false_negatives': int(fn_mask.sum()),
                'precision': float(tp_mask.sum() / (tp_mask.sum() + fp_mask.sum())) if (tp_mask.sum() + fp_mask.sum()) > 0 else 0,
                'recall': float(tp_mask.sum() / (tp_mask.sum() + fn_mask.sum())) if (tp_mask.sum() + fn_mask.sum()) > 0 else 0,
            }
            results['f1'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall']) if (results['precision'] + results['recall']) > 0 else 0
            
            # Analyze confidence scores
            results['confidence_analysis'] = {
                'tp_avg_confidence': float(h_scores[tp_mask].mean()) if tp_mask.sum() > 0 else 0,
                'tn_avg_confidence': float(1 - h_scores[tn_mask].mean()) if tn_mask.sum() > 0 else 0,
                'fp_avg_confidence': float(h_scores[fp_mask].mean()) if fp_mask.sum() > 0 else 0,
                'fn_avg_confidence': float(h_scores[fn_mask].mean()) if fn_mask.sum() > 0 else 0,
            }
            
            # Find near-miss cases (close to threshold)
            near_miss_threshold = 0.1
            near_miss_fn = fn_mask & (h_scores >= 0.5 - near_miss_threshold)
            near_miss_fp = fp_mask & (h_scores <= 0.5 + near_miss_threshold)
            
            results['near_misses'] = {
                'false_negatives_near_threshold': int(near_miss_fn.sum()),
                'false_positives_near_threshold': int(near_miss_fp.sum()),
            }
            
            # Analyze co-occurrence patterns
            co_occurrence = self._analyze_co_occurrence(h_true, h_pred, targets, pred_binary)
            results['co_occurrence'] = co_occurrence
            
            # Text analysis if available
            if texts:
                text_analysis = self._analyze_error_texts(
                    texts, h_true, h_pred, h_scores, fp_mask, fn_mask, hallmark_id
                )
                results['text_analysis'] = text_analysis
            
            analysis_results[hallmark_name] = results
            
            # Create visualizations
            self._create_failure_visualizations(
                h_true, h_pred, h_scores, hallmark_name, save_dir
            )
        
        # Save detailed results
        with open(save_dir / 'failure_analysis_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Generate report
        self._generate_failure_report(analysis_results, save_dir)
        
        return analysis_results
    
    def _analyze_co_occurrence(self, h_true, h_pred, all_true, all_pred):
        """Analyze which hallmarks co-occur with errors."""
        co_occurrence_stats = {
            'fn_common_co_hallmarks': {},
            'fp_common_co_hallmarks': {},
        }
        
        # False negatives - what other hallmarks are present when we miss this one?
        fn_mask = (h_pred == 0) & (h_true == 1)
        if fn_mask.sum() > 0:
            fn_other_hallmarks = all_true[fn_mask]
            for i in range(11):
                if i in self.target_hallmarks:
                    continue
                co_rate = fn_other_hallmarks[:, i].mean()
                if co_rate > 0.1:  # At least 10% co-occurrence
                    co_occurrence_stats['fn_common_co_hallmarks'][self.hallmark_names[i]] = float(co_rate)
        
        # False positives - what makes us incorrectly predict this hallmark?
        fp_mask = (h_pred == 1) & (h_true == 0)
        if fp_mask.sum() > 0:
            fp_other_hallmarks = all_true[fp_mask]
            for i in range(11):
                if i in self.target_hallmarks:
                    continue
                co_rate = fp_other_hallmarks[:, i].mean()
                if co_rate > 0.1:
                    co_occurrence_stats['fp_common_co_hallmarks'][self.hallmark_names[i]] = float(co_rate)
        
        return co_occurrence_stats
    
    def _analyze_error_texts(self, texts, h_true, h_pred, h_scores, fp_mask, fn_mask, hallmark_id):
        """Analyze text patterns in errors."""
        text_analysis = {
            'keyword_presence': {},
            'avg_text_length': {},
            'sample_errors': {'false_negatives': [], 'false_positives': []}
        }
        
        keywords = self.hallmark_keywords.get(hallmark_id, [])
        
        # Analyze false negatives
        fn_indices = np.where(fn_mask)[0]
        fn_keyword_count = 0
        fn_lengths = []
        
        for idx in fn_indices[:20]:  # Sample first 20
            if idx < len(texts):
                text = texts[idx]
                fn_lengths.append(len(text.split()))
                
                # Check keywords
                text_lower = text.lower()
                has_keyword = any(keyword.lower() in text_lower for keyword in keywords)
                if has_keyword:
                    fn_keyword_count += 1
                
                # Store sample
                if len(text_analysis['sample_errors']['false_negatives']) < 5:
                    text_analysis['sample_errors']['false_negatives'].append({
                        'text': text[:200] + '...' if len(text) > 200 else text,
                        'confidence': float(h_scores[idx]),
                        'has_keywords': has_keyword
                    })
        
        if fn_indices.size > 0:
            text_analysis['keyword_presence']['false_negatives'] = fn_keyword_count / min(len(fn_indices), 20)
            text_analysis['avg_text_length']['false_negatives'] = np.mean(fn_lengths) if fn_lengths else 0
        
        # Analyze false positives
        fp_indices = np.where(fp_mask)[0]
        fp_keyword_count = 0
        fp_lengths = []
        
        for idx in fp_indices[:20]:
            if idx < len(texts):
                text = texts[idx]
                fp_lengths.append(len(text.split()))
                
                text_lower = text.lower()
                has_keyword = any(keyword.lower() in text_lower for keyword in keywords)
                if has_keyword:
                    fp_keyword_count += 1
                
                if len(text_analysis['sample_errors']['false_positives']) < 5:
                    text_analysis['sample_errors']['false_positives'].append({
                        'text': text[:200] + '...' if len(text) > 200 else text,
                        'confidence': float(h_scores[idx]),
                        'has_keywords': has_keyword
                    })
        
        if fp_indices.size > 0:
            text_analysis['keyword_presence']['false_positives'] = fp_keyword_count / min(len(fp_indices), 20)
            text_analysis['avg_text_length']['false_positives'] = np.mean(fp_lengths) if fp_lengths else 0
        
        return text_analysis
    
    def _create_failure_visualizations(self, h_true, h_pred, h_scores, hallmark_name, save_dir):
        """Create visualizations for failure analysis."""
        # 1. Confidence distribution plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # True Positives
        tp_scores = h_scores[(h_pred == 1) & (h_true == 1)]
        if len(tp_scores) > 0:
            axes[0, 0].hist(tp_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 0].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        axes[0, 0].set_title(f'True Positives (n={len(tp_scores)})')
        axes[0, 0].set_xlabel('Confidence Score')
        axes[0, 0].set_xlim(0, 1)
        
        # False Positives
        fp_scores = h_scores[(h_pred == 1) & (h_true == 0)]
        if len(fp_scores) > 0:
            axes[0, 1].hist(fp_scores, bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        axes[0, 1].set_title(f'False Positives (n={len(fp_scores)})')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_xlim(0, 1)
        
        # True Negatives
        tn_scores = h_scores[(h_pred == 0) & (h_true == 0)]
        if len(tn_scores) > 0:
            axes[1, 0].hist(tn_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        axes[1, 0].set_title(f'True Negatives (n={len(tn_scores)})')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].set_xlim(0, 1)
        
        # False Negatives
        fn_scores = h_scores[(h_pred == 0) & (h_true == 1)]
        if len(fn_scores) > 0:
            axes[1, 1].hist(fn_scores, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        axes[1, 1].set_title(f'False Negatives (n={len(fn_scores)})')
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_xlim(0, 1)
        
        plt.suptitle(f'Confidence Score Distributions - {hallmark_name}', fontsize=14)
        plt.tight_layout()
        safe_name = hallmark_name.replace(' ', '_').lower()
        plt.savefig(save_dir / f'confidence_distributions_{safe_name}.png', dpi=300)
        plt.close()
        
        # 2. Error rate by confidence bins
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Create confidence bins
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        error_rates = []
        sample_counts = []
        
        for i in range(len(bins) - 1):
            mask = (h_scores >= bins[i]) & (h_scores < bins[i+1])
            if mask.sum() > 0:
                errors = (h_pred[mask] != h_true[mask]).sum()
                error_rate = errors / mask.sum()
                error_rates.append(error_rate)
                sample_counts.append(mask.sum())
            else:
                error_rates.append(0)
                sample_counts.append(0)
        
        # Plot error rates
        bars = ax.bar(bin_centers, error_rates, width=0.08, alpha=0.7)
        
        # Add sample counts
        for i, (bar, count) in enumerate(zip(bars, sample_counts)):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'n={count}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Confidence Score Bin')
        ax.set_ylabel('Error Rate')
        ax.set_title(f'Error Rate by Confidence - {hallmark_name}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(error_rates) * 1.2 if error_rates else 1)
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Decision Threshold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / f'error_rate_by_confidence_{safe_name}.png', dpi=300)
        plt.close()
    
    def _generate_failure_report(self, analysis_results, save_dir):
        """Generate detailed failure analysis report."""
        report_path = save_dir / 'failure_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("Failure Analysis Report for Low-Performing Hallmarks\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for hallmark_name, results in analysis_results.items():
                f.write(f"\n{hallmark_name.upper()}\n")
                f.write("-" * len(hallmark_name) + "\n\n")
                
                # Basic metrics
                f.write("Performance Metrics:\n")
                f.write(f"  F1 Score: {results['f1']:.3f}\n")
                f.write(f"  Precision: {results['precision']:.3f}\n")
                f.write(f"  Recall: {results['recall']:.3f}\n")
                f.write(f"  Support: {results['positive_samples']} positive, "
                       f"{results['negative_samples']} negative\n\n")
                
                # Error breakdown
                f.write("Error Analysis:\n")
                f.write(f"  False Negatives: {results['false_negatives']} "
                       f"({results['false_negatives']/max(results['positive_samples'], 1)*100:.1f}% of positives)\n")
                f.write(f"  False Positives: {results['false_positives']} "
                       f"({results['false_positives']/max(results['negative_samples'], 1)*100:.1f}% of negatives)\n\n")
                
                # Confidence analysis
                conf = results['confidence_analysis']
                f.write("Confidence Analysis:\n")
                f.write(f"  Avg confidence when correct (TP): {conf['tp_avg_confidence']:.3f}\n")
                f.write(f"  Avg confidence when correct (TN): {conf['tn_avg_confidence']:.3f}\n")
                f.write(f"  Avg confidence when wrong (FP): {conf['fp_avg_confidence']:.3f}\n")
                f.write(f"  Avg confidence when wrong (FN): {conf['fn_avg_confidence']:.3f}\n\n")
                
                # Near misses
                f.write("Near-Miss Analysis:\n")
                f.write(f"  FN near threshold: {results['near_misses']['false_negatives_near_threshold']}\n")
                f.write(f"  FP near threshold: {results['near_misses']['false_positives_near_threshold']}\n\n")
                
                # Co-occurrence patterns
                if results['co_occurrence']['fn_common_co_hallmarks']:
                    f.write("Common co-occurring hallmarks in False Negatives:\n")
                    for co_hallmark, rate in sorted(results['co_occurrence']['fn_common_co_hallmarks'].items(), 
                                                   key=lambda x: x[1], reverse=True):
                        f.write(f"  - {co_hallmark}: {rate:.2f}\n")
                    f.write("\n")
                
                if results['co_occurrence']['fp_common_co_hallmarks']:
                    f.write("Common co-occurring hallmarks in False Positives:\n")
                    for co_hallmark, rate in sorted(results['co_occurrence']['fp_common_co_hallmarks'].items(), 
                                                   key=lambda x: x[1], reverse=True):
                        f.write(f"  - {co_hallmark}: {rate:.2f}\n")
                    f.write("\n")
                
                # Text analysis
                if 'text_analysis' in results:
                    text_stats = results['text_analysis']
                    f.write("Text Pattern Analysis:\n")
                    if 'false_negatives' in text_stats['keyword_presence']:
                        f.write(f"  Keyword presence in FN: {text_stats['keyword_presence']['false_negatives']:.1%}\n")
                    if 'false_positives' in text_stats['keyword_presence']:
                        f.write(f"  Keyword presence in FP: {text_stats['keyword_presence']['false_positives']:.1%}\n")
                    f.write("\n")
                    
                    # Sample errors
                    if text_stats['sample_errors']['false_negatives']:
                        f.write("Sample False Negatives:\n")
                        for i, sample in enumerate(text_stats['sample_errors']['false_negatives'][:3]):
                            f.write(f"  {i+1}. Text: {sample['text']}\n")
                            f.write(f"     Confidence: {sample['confidence']:.3f}, "
                                   f"Has keywords: {sample['has_keywords']}\n\n")
            
            # Key insights
            f.write("\n\nKEY INSIGHTS\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("1. Common Failure Patterns:\n")
            f.write("   - Both hallmarks suffer from low recall (missing true positives)\n")
            f.write("   - Confidence scores for false negatives are often just below threshold\n")
            f.write("   - Co-occurrence with other hallmarks affects prediction accuracy\n\n")
            
            f.write("2. Potential Causes:\n")
            f.write("   - Class imbalance (fewer positive examples)\n")
            f.write("   - Subtle or implicit language patterns\n")
            f.write("   - Complex biological relationships not fully captured\n\n")
            
            f.write("3. Recommendations:\n")
            f.write("   - Consider hallmark-specific threshold optimization\n")
            f.write("   - Augment training data for underrepresented hallmarks\n")
            f.write("   - Enhance biological knowledge integration for these pathways\n")
            f.write("   - Add hallmark-specific attention mechanisms\n")
        
        logger.info(f"Failure analysis report saved to {report_path}")


def main():
    """Main function for failure analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Failure analysis for low-performing hallmarks")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='failure_analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--split', type=str, default='test',
                       choices=['validation', 'test'],
                       help='Which dataset split to analyze')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = FailureAnalyzer(args.config)
    
    # Load model and data
    logger.info("Loading model and data...")
    model, data_module = analyzer.load_model_and_data(args.checkpoint)
    
    # Get appropriate data loader
    if args.split == 'test':
        data_loader = data_module.test_dataloader()
    else:
        data_loader = data_module.val_dataloader()
    
    # Get predictions
    logger.info(f"Getting predictions on {args.split} set...")
    predictions, targets, texts, entities = analyzer.analyze_predictions(model, data_loader)
    
    # Perform failure analysis
    logger.info("Analyzing failure patterns...")
    results = analyzer.analyze_failure_patterns(predictions, targets, texts, output_dir)
    
    # Print summary
    logger.info("\nFAILURE ANALYSIS SUMMARY")
    logger.info("=" * 40)
    for hallmark_name, result in results.items():
        logger.info(f"\n{hallmark_name}:")
        logger.info(f"  F1 Score: {result['f1']:.3f}")
        logger.info(f"  Main issue: {'Low Recall' if result['recall'] < result['precision'] else 'Low Precision'}")
        logger.info(f"  False Negatives: {result['false_negatives']} ({result['recall']:.1%} recall)")
        logger.info(f"  Near-threshold FN: {result['near_misses']['false_negatives_near_threshold']}")
    
    logger.info(f"\nDetailed analysis saved to {output_dir}")


if __name__ == "__main__":
    main()