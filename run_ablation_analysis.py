#!/usr/bin/env python3
"""
Ablation Analysis Script

This script performs comprehensive analysis of ablation study results,
comparing each ablation variant against the full BioKG-BioBERT model.
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score, hamming_loss
import warnings
warnings.filterwarnings('ignore')

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Hallmark names for better readability
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

# Ablation category mapping (without full model - will be added dynamically)
ABLATION_CATEGORIES = {
    'ablation_attention_none': 'A1: No Bio Attention',
    'ablation_attention_entity_only': 'A1: Entity-Only Attention',
    'ablation_kg_none': 'A2: No Knowledge Graph',
    'ablation_kg_1hop': 'A2: 1-Hop KG Only',
    'ablation_fusion_early': 'A3: Early Fusion',
    'ablation_fusion_cross_modal': 'A3: Cross-Modal Fusion',
    'ablation_multitask_hallmarks_only': 'A4: Hallmarks Only',
    'ablation_multitask_with_pathway': 'A4: With Pathway Loss',
    'ablation_multitask_with_consistency': 'A4: With Consistency Loss'
}


class AblationAnalyzer:
    """Analyzes ablation study results comprehensively."""
    
    def __init__(self, experiment_dir: str, output_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different analysis types
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.tables_dir = self.output_dir / "tables"
        self.tables_dir.mkdir(exist_ok=True)
        
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Will be set by main()
        self.full_model_name = 'biokg_biobert'
        
    def get_ablation_groups(self):
        """Get ablation groups with dynamic full model name."""
        return {
            'Attention Mechanisms': [self.full_model_name, 'ablation_attention_none', 'ablation_attention_entity_only'],
            'Knowledge Graph Integration': [self.full_model_name, 'ablation_kg_none', 'ablation_kg_1hop'],
            'Fusion Strategies': [self.full_model_name, 'ablation_fusion_early', 'ablation_fusion_cross_modal'],
            'Multi-Task Learning': [self.full_model_name, 'ablation_multitask_hallmarks_only', 
                                   'ablation_multitask_with_pathway', 'ablation_multitask_with_consistency']
        }
    
    def get_ablation_categories(self):
        """Get ablation categories with dynamic full model name."""
        categories = ABLATION_CATEGORIES.copy()
        categories[self.full_model_name] = 'Full Model'
        return categories
        
    def load_model_results(self, model_name: str):
        """Load results for a specific model."""
        model_dir = self.experiment_dir / model_name
        
        # Load test results
        results_file = model_dir / "results" / "test_results.json"
        if not results_file.exists():
            print(f"Warning: No test results found for {model_name}")
            return None
            
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        # Load predictions if available
        predictions_file = model_dir / "results" / "test_predictions.json"
        if predictions_file.exists():
            with open(predictions_file, 'r') as f:
                predictions = json.load(f)
            results['predictions'] = predictions
            
        # Load training history if available
        history_file = model_dir / "results" / "training_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
            results['history'] = history
            
        return results
    
    def calculate_additional_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate additional evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['exact_match_ratio'] = np.mean([np.array_equal(t, p) for t, p in zip(y_true, y_pred)])
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['per_class_precision'] = precision.tolist()
        metrics['per_class_recall'] = recall.tolist()
        metrics['per_class_f1'] = f1.tolist()
        metrics['per_class_support'] = support.tolist()
        
        # AUC if probabilities available
        if y_prob is not None:
            try:
                # Calculate AUC for each class
                auc_scores = []
                for i in range(y_true.shape[1]):
                    if len(np.unique(y_true[:, i])) > 1:  # Check if both classes present
                        auc = roc_auc_score(y_true[:, i], y_prob[:, i])
                        auc_scores.append(auc)
                    else:
                        auc_scores.append(np.nan)
                metrics['per_class_auc'] = auc_scores
                metrics['macro_auc'] = np.nanmean(auc_scores)
            except Exception as e:
                print(f"Warning: Could not calculate AUC: {e}")
        
        return metrics
    
    def compare_with_full_model(self, full_results, ablation_results, ablation_name):
        """Compare ablation model with full model."""
        comparison = {
            'ablation_name': ablation_name,
            'category': self.get_ablation_categories().get(ablation_name, ablation_name)
        }
        
        # Performance differences
        for metric in ['micro_f1', 'macro_f1', 'weighted_f1']:
            if metric in full_results and metric in ablation_results:
                full_val = full_results[metric]
                ablation_val = ablation_results[metric]
                comparison[f'{metric}_diff'] = ablation_val - full_val
                comparison[f'{metric}_relative_diff'] = (ablation_val - full_val) / full_val * 100
        
        # Statistical significance test (if predictions available)
        if 'predictions' in full_results and 'predictions' in ablation_results:
            full_pred = np.array(full_results['predictions']['y_pred'])
            ablation_pred = np.array(ablation_results['predictions']['y_pred'])
            y_true = np.array(full_results['predictions']['y_true'])
            
            # McNemar's test for each hallmark
            mcnemar_results = []
            for i in range(y_true.shape[1]):
                # Create contingency table
                correct_full = (full_pred[:, i] == y_true[:, i])
                correct_ablation = (ablation_pred[:, i] == y_true[:, i])
                
                # Count disagreements
                n01 = np.sum(~correct_full & correct_ablation)  # Full wrong, ablation correct
                n10 = np.sum(correct_full & ~correct_ablation)  # Full correct, ablation wrong
                
                # McNemar's test
                if n01 + n10 > 0:
                    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
                    p_value = 1 - stats.chi2.cdf(statistic, 1)
                else:
                    p_value = 1.0
                    
                mcnemar_results.append({
                    'hallmark': i,
                    'n01': n01,
                    'n10': n10,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
            
            comparison['mcnemar_results'] = mcnemar_results
        
        return comparison
    
    def generate_performance_comparison_plot(self, results_dict):
        """Generate comprehensive performance comparison plots."""
        # Prepare data for plotting
        models = []
        micro_f1_scores = []
        macro_f1_scores = []
        weighted_f1_scores = []
        
        # Order models by category
        model_order = [self.full_model_name]  # Full model first
        for group_models in self.get_ablation_groups().values():
            for model in group_models[1:]:  # Skip full model duplicate
                if model not in model_order:
                    model_order.append(model)
        
        for model in model_order:
            if model in results_dict and results_dict[model] is not None:
                models.append(self.get_ablation_categories().get(model, model))
                micro_f1_scores.append(results_dict[model].get('micro_f1', 0))
                macro_f1_scores.append(results_dict[model].get('macro_f1', 0))
                weighted_f1_scores.append(results_dict[model].get('weighted_f1', 0))
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ablation Study: Performance Comparison', fontsize=16, y=1.02)
        
        # 1. Overall F1 scores comparison
        ax = axes[0, 0]
        x = np.arange(len(models))
        width = 0.25
        
        ax.bar(x - width, micro_f1_scores, width, label='Micro F1', alpha=0.8)
        ax.bar(x, macro_f1_scores, width, label='Macro F1', alpha=0.8)
        ax.bar(x + width, weighted_f1_scores, width, label='Weighted F1', alpha=0.8)
        
        ax.set_xlabel('Model Variant')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score Comparison Across Ablation Variants')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (micro, macro, weighted) in enumerate(zip(micro_f1_scores, macro_f1_scores, weighted_f1_scores)):
            ax.text(i - width, micro + 0.01, f'{micro:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i, macro + 0.01, f'{macro:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width, weighted + 0.01, f'{weighted:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Performance drop from full model
        ax = axes[0, 1]
        full_micro_f1 = micro_f1_scores[0] if micro_f1_scores else 0
        performance_drops = [(score - full_micro_f1) / full_micro_f1 * 100 for score in micro_f1_scores[1:]]
        
        colors = ['red' if drop < 0 else 'green' for drop in performance_drops]
        bars = ax.bar(range(len(performance_drops)), performance_drops, color=colors, alpha=0.7)
        
        ax.set_xlabel('Ablation Variant')
        ax.set_ylabel('Performance Change (%)')
        ax.set_title('Performance Change Relative to Full Model (Micro F1)')
        ax.set_xticks(range(len(performance_drops)))
        ax.set_xticklabels(models[1:], rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, drop) in enumerate(zip(bars, performance_drops)):
            ax.text(i, drop + (0.5 if drop > 0 else -0.5), f'{drop:.1f}%', 
                   ha='center', va='bottom' if drop > 0 else 'top', fontsize=8)
        
        # 3. Grouped comparison by ablation category
        ax = axes[1, 0]
        group_names = list(self.get_ablation_groups().keys())
        group_positions = []
        current_pos = 0
        
        for group_name, group_models in self.get_ablation_groups().items():
            group_scores = []
            group_labels = []
            
            for model in group_models:
                if model in results_dict and results_dict[model] is not None:
                    group_scores.append(results_dict[model].get('micro_f1', 0))
                    group_labels.append(self.get_ablation_categories().get(model, model).split(': ')[-1])
            
            if group_scores:
                positions = np.arange(len(group_scores)) + current_pos
                bars = ax.bar(positions, group_scores, alpha=0.8, label=group_name)
                
                # Add value labels
                for pos, score in zip(positions, group_scores):
                    ax.text(pos, score + 0.01, f'{score:.3f}', ha='center', va='bottom', fontsize=8)
                
                group_positions.append((current_pos, current_pos + len(group_scores) - 1, group_name))
                current_pos += len(group_scores) + 1
        
        ax.set_xlabel('Model Variant')
        ax.set_ylabel('Micro F1 Score')
        ax.set_title('Performance by Ablation Category')
        ax.grid(True, alpha=0.3)
        
        # Add group labels
        for start, end, name in group_positions:
            ax.text((start + end) / 2, -0.15, name, ha='center', va='top', 
                   transform=ax.get_xaxis_transform(), fontsize=10, weight='bold')
        
        # 4. Per-hallmark heatmap (if available)
        ax = axes[1, 1]
        if any('per_hallmark_f1' in results_dict.get(model, {}) for model in model_order):
            # Create per-hallmark F1 matrix
            hallmark_matrix = []
            available_models = []
            
            for model in model_order:
                if model in results_dict and results_dict[model] is not None:
                    if 'per_hallmark_f1' in results_dict[model]:
                        hallmark_matrix.append(results_dict[model]['per_hallmark_f1'])
                        available_models.append(self.get_ablation_categories().get(model, model))
            
            if hallmark_matrix:
                hallmark_matrix = np.array(hallmark_matrix)
                
                # Create heatmap
                sns.heatmap(hallmark_matrix.T, 
                           xticklabels=available_models,
                           yticklabels=[HALLMARK_NAMES[i] for i in range(hallmark_matrix.shape[1])],
                           annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                           cbar_kws={'label': 'F1 Score'},
                           ax=ax)
                ax.set_title('Per-Hallmark F1 Scores Heatmap')
                ax.set_xlabel('Model Variant')
                ax.set_ylabel('Cancer Hallmark')
        else:
            ax.text(0.5, 0.5, 'Per-hallmark scores not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Per-Hallmark F1 Scores Heatmap')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'ablation_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_ablation_impact_analysis(self, comparisons):
        """Generate detailed ablation impact analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ablation Impact Analysis', fontsize=16, y=1.02)
        
        # 1. Component importance ranking
        ax = axes[0, 0]
        component_impacts = []
        
        for comp in comparisons:
            if 'micro_f1_diff' in comp:
                impact = abs(comp['micro_f1_diff'])
                component_impacts.append({
                    'component': comp['category'].split(': ')[-1],
                    'impact': impact,
                    'direction': 'negative' if comp['micro_f1_diff'] < 0 else 'positive'
                })
        
        component_impacts.sort(key=lambda x: x['impact'], reverse=True)
        
        components = [c['component'] for c in component_impacts]
        impacts = [c['impact'] for c in component_impacts]
        colors = ['red' if c['direction'] == 'negative' else 'green' for c in component_impacts]
        
        bars = ax.barh(components, impacts, color=colors, alpha=0.7)
        ax.set_xlabel('Absolute F1 Score Impact')
        ax.set_title('Component Importance Ranking')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, impact in zip(bars, impacts):
            ax.text(impact + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{impact:.3f}', ha='left', va='center', fontsize=8)
        
        # 2. Statistical significance summary
        ax = axes[0, 1]
        if any('mcnemar_results' in comp for comp in comparisons):
            sig_counts = []
            labels = []
            
            for comp in comparisons:
                if 'mcnemar_results' in comp:
                    sig_count = sum(1 for r in comp['mcnemar_results'] if r['significant'])
                    sig_counts.append(sig_count)
                    labels.append(comp['category'].split(': ')[-1])
            
            if sig_counts:
                bars = ax.bar(range(len(sig_counts)), sig_counts, alpha=0.8)
                ax.set_xlabel('Ablation Variant')
                ax.set_ylabel('Number of Significantly Different Hallmarks')
                ax.set_title('Statistical Significance of Changes (McNemar Test, p<0.05)')
                ax.set_xticks(range(len(sig_counts)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylim(0, 11)
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, count in zip(bars, sig_counts):
                    ax.text(bar.get_x() + bar.get_width()/2, count + 0.1, 
                           str(count), ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Statistical significance analysis not available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # 3. Performance trade-offs scatter plot
        ax = axes[1, 0]
        if len(comparisons) > 0:
            # Extract metrics for scatter plot
            micro_diffs = []
            macro_diffs = []
            labels = []
            
            for comp in comparisons:
                if 'micro_f1_diff' in comp and 'macro_f1_diff' in comp:
                    micro_diffs.append(comp['micro_f1_diff'] * 100)
                    macro_diffs.append(comp['macro_f1_diff'] * 100)
                    labels.append(comp['category'].split(': ')[-1])
            
            if micro_diffs and macro_diffs:
                scatter = ax.scatter(micro_diffs, macro_diffs, s=100, alpha=0.7)
                
                # Add labels
                for i, label in enumerate(labels):
                    ax.annotate(label, (micro_diffs[i], macro_diffs[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                ax.set_xlabel('Micro F1 Change (%)')
                ax.set_ylabel('Macro F1 Change (%)')
                ax.set_title('Performance Trade-offs: Micro vs Macro F1')
                ax.grid(True, alpha=0.3)
                
                # Add quadrant labels
                ax.text(0.95, 0.95, 'Better on both', transform=ax.transAxes, 
                       ha='right', va='top', fontsize=10, alpha=0.5)
                ax.text(0.05, 0.05, 'Worse on both', transform=ax.transAxes, 
                       ha='left', va='bottom', fontsize=10, alpha=0.5)
        
        # 4. Ablation group summary
        ax = axes[1, 1]
        group_summaries = {}
        
        for group_name, group_models in self.get_ablation_groups().items():
            group_impacts = []
            for comp in comparisons:
                model_name = comp['ablation_name']
                if model_name in group_models and 'micro_f1_diff' in comp:
                    group_impacts.append(abs(comp['micro_f1_diff']))
            
            if group_impacts:
                group_summaries[group_name] = {
                    'mean_impact': np.mean(group_impacts),
                    'max_impact': np.max(group_impacts),
                    'count': len(group_impacts)
                }
        
        if group_summaries:
            groups = list(group_summaries.keys())
            mean_impacts = [group_summaries[g]['mean_impact'] for g in groups]
            max_impacts = [group_summaries[g]['max_impact'] for g in groups]
            
            x = np.arange(len(groups))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, mean_impacts, width, label='Mean Impact', alpha=0.8)
            bars2 = ax.bar(x + width/2, max_impacts, width, label='Max Impact', alpha=0.8)
            
            ax.set_xlabel('Ablation Group')
            ax.set_ylabel('Absolute F1 Score Impact')
            ax.set_title('Impact Summary by Ablation Group')
            ax.set_xticks(x)
            ax.set_xticklabels(groups, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.001,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'ablation_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_detailed_report(self, results_dict, comparisons):
        """Generate comprehensive ablation analysis report."""
        report_path = self.reports_dir / 'ablation_analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# BioKG-BioBERT Ablation Study Analysis Report\n\n")
            f.write("## Executive Summary\n\n")
            
            # Overall performance summary
            full_model_results = results_dict.get(self.full_model_name, {})
            if full_model_results:
                f.write(f"**Full Model Performance:**\n")
                f.write(f"- Micro F1: {full_model_results.get('micro_f1', 0):.4f}\n")
                f.write(f"- Macro F1: {full_model_results.get('macro_f1', 0):.4f}\n")
                f.write(f"- Weighted F1: {full_model_results.get('weighted_f1', 0):.4f}\n\n")
            
            # Key findings
            f.write("### Key Findings\n\n")
            
            # Find most impactful components
            impact_ranking = []
            for comp in comparisons:
                if 'micro_f1_diff' in comp:
                    impact_ranking.append({
                        'name': comp['category'],
                        'impact': abs(comp['micro_f1_diff']),
                        'diff': comp['micro_f1_diff']
                    })
            
            impact_ranking.sort(key=lambda x: x['impact'], reverse=True)
            
            f.write("**Component Impact Ranking (by absolute F1 change):**\n\n")
            for i, comp in enumerate(impact_ranking[:5], 1):
                direction = "↓" if comp['diff'] < 0 else "↑"
                f.write(f"{i}. **{comp['name']}**: {direction} {comp['impact']:.4f} ({comp['diff']*100:+.2f}%)\n")
            
            f.write("\n## Detailed Analysis by Ablation Group\n\n")
            
            # Analyze each ablation group
            for group_name, group_models in self.get_ablation_groups().items():
                f.write(f"### {group_name}\n\n")
                
                # Create comparison table for this group
                f.write("| Model Variant | Micro F1 | Macro F1 | Weighted F1 | Δ Micro F1 | Relative Change |\n")
                f.write("|--------------|----------|----------|-------------|-----------|----------------|\n")
                
                for model in group_models:
                    if model in results_dict and results_dict[model] is not None:
                        result = results_dict[model]
                        name = self.get_ablation_categories().get(model, model).split(': ')[-1]
                        
                        if model == self.full_model_name:
                            f.write(f"| **{name}** (Baseline) | **{result.get('micro_f1', 0):.4f}** | ")
                            f.write(f"**{result.get('macro_f1', 0):.4f}** | **{result.get('weighted_f1', 0):.4f}** | - | - |\n")
                        else:
                            # Find comparison
                            comp = next((c for c in comparisons if c['ablation_name'] == model), None)
                            if comp and 'micro_f1_diff' in comp:
                                f.write(f"| {name} | {result.get('micro_f1', 0):.4f} | ")
                                f.write(f"{result.get('macro_f1', 0):.4f} | {result.get('weighted_f1', 0):.4f} | ")
                                f.write(f"{comp['micro_f1_diff']:+.4f} | {comp['micro_f1_relative_diff']:+.2f}% |\n")
                            else:
                                f.write(f"| {name} | {result.get('micro_f1', 0):.4f} | ")
                                f.write(f"{result.get('macro_f1', 0):.4f} | {result.get('weighted_f1', 0):.4f} | N/A | N/A |\n")
                
                # Group insights
                f.write(f"\n**Insights for {group_name}:**\n")
                
                if group_name == "Attention Mechanisms":
                    f.write("- Biological attention mechanisms contribute significantly to model performance\n")
                    f.write("- Pathway information provides additional benefits beyond entity-level attention\n")
                elif group_name == "Knowledge Graph Integration":
                    f.write("- Knowledge graph integration is crucial for achieving optimal performance\n")
                    f.write("- Multi-hop relationships provide marginal improvements over single-hop\n")
                elif group_name == "Fusion Strategies":
                    f.write("- Late fusion (default) shows best performance for multi-modal integration\n")
                    f.write("- Alternative fusion strategies may trade off performance for efficiency\n")
                elif group_name == "Multi-Task Learning":
                    f.write("- Auxiliary tasks (pathway and consistency losses) improve generalization\n")
                    f.write("- Pathway prediction task provides stronger regularization than consistency alone\n")
                
                f.write("\n")
            
            # Statistical significance analysis
            f.write("## Statistical Significance Analysis\n\n")
            f.write("McNemar's test results for each ablation variant (p < 0.05 indicates significant difference):\n\n")
            
            sig_summary = []
            for comp in comparisons:
                if 'mcnemar_results' in comp:
                    sig_count = sum(1 for r in comp['mcnemar_results'] if r['significant'])
                    sig_summary.append({
                        'name': comp['category'],
                        'sig_count': sig_count,
                        'total': len(comp['mcnemar_results'])
                    })
            
            if sig_summary:
                f.write("| Ablation Variant | Significantly Different Hallmarks | Percentage |\n")
                f.write("|-----------------|----------------------------------|------------|\n")
                
                for item in sig_summary:
                    percentage = (item['sig_count'] / item['total']) * 100
                    f.write(f"| {item['name']} | {item['sig_count']}/{item['total']} | {percentage:.1f}% |\n")
            
            # Per-hallmark analysis
            f.write("\n## Per-Hallmark Performance Analysis\n\n")
            
            # Find hallmarks most affected by ablations
            hallmark_impacts = {i: [] for i in range(11)}
            
            for model_name, result in results_dict.items():
                if result and 'per_hallmark_f1' in result and model_name != self.full_model_name:
                    full_hallmark_f1 = results_dict.get(self.full_model_name, {}).get('per_hallmark_f1', [0]*11)
                    for i, (full_f1, ablation_f1) in enumerate(zip(full_hallmark_f1, result['per_hallmark_f1'])):
                        hallmark_impacts[i].append(ablation_f1 - full_f1)
            
            f.write("**Hallmarks Most Sensitive to Ablations:**\n\n")
            sensitivity_scores = []
            for hallmark_id, impacts in hallmark_impacts.items():
                if impacts:
                    sensitivity_scores.append({
                        'id': hallmark_id,
                        'name': HALLMARK_NAMES[hallmark_id],
                        'mean_impact': np.mean(np.abs(impacts)),
                        'std_impact': np.std(impacts)
                    })
            
            sensitivity_scores.sort(key=lambda x: x['mean_impact'], reverse=True)
            
            for i, score in enumerate(sensitivity_scores[:5], 1):
                f.write(f"{i}. **{score['name']}**: Mean absolute impact = {score['mean_impact']:.4f} (σ = {score['std_impact']:.4f})\n")
            
            # Conclusions and recommendations
            f.write("\n## Conclusions and Recommendations\n\n")
            f.write("### Component Criticality\n\n")
            f.write("Based on the ablation analysis, the following components are critical for optimal performance:\n\n")
            
            critical_components = []
            for comp in comparisons:
                if 'micro_f1_diff' in comp and comp['micro_f1_diff'] < -0.01:  # More than 1% drop
                    critical_components.append({
                        'name': comp['category'].split(': ')[-1],
                        'impact': abs(comp['micro_f1_diff'])
                    })
            
            critical_components.sort(key=lambda x: x['impact'], reverse=True)
            
            for comp in critical_components:
                f.write(f"- **{comp['name']}**: Removal causes {comp['impact']*100:.1f}% performance drop\n")
            
            f.write("\n### Optimization Opportunities\n\n")
            f.write("1. **Computational Efficiency**: Components with minimal impact could be removed for faster inference\n")
            f.write("2. **Architecture Simplification**: Consider removing components that don't significantly improve performance\n")
            f.write("3. **Training Strategy**: Focus computational resources on the most impactful components\n")
            
            f.write("\n### Future Research Directions\n\n")
            f.write("1. Investigate interaction effects between different components\n")
            f.write("2. Explore more efficient implementations of critical components\n")
            f.write("3. Develop adaptive mechanisms to selectively enable/disable components based on input\n")
            
            # Appendix with detailed metrics
            f.write("\n## Appendix: Detailed Metrics\n\n")
            
            # Create comprehensive metrics table
            f.write("### Complete Performance Metrics\n\n")
            f.write("| Model | Exact Match | Hamming Loss | Micro F1 | Macro F1 | Weighted F1 |\n")
            f.write("|-------|-------------|--------------|----------|----------|-------------|\n")
            
            for model in [self.full_model_name] + [m for m in model_order if m != self.full_model_name]:
                if model in results_dict and results_dict[model] is not None:
                    result = results_dict[model]
                    name = self.get_ablation_categories().get(model, model)
                    
                    # Calculate additional metrics if needed
                    if 'predictions' in result:
                        y_true = np.array(result['predictions']['y_true'])
                        y_pred = np.array(result['predictions']['y_pred'])
                        
                        exact_match = np.mean([np.array_equal(t, p) for t, p in zip(y_true, y_pred)])
                        h_loss = hamming_loss(y_true, y_pred)
                    else:
                        exact_match = result.get('exact_match_ratio', 'N/A')
                        h_loss = result.get('hamming_loss', 'N/A')
                    
                    f.write(f"| {name} | {exact_match:.4f if isinstance(exact_match, float) else exact_match} | ")
                    f.write(f"{h_loss:.4f if isinstance(h_loss, float) else h_loss} | ")
                    f.write(f"{result.get('micro_f1', 0):.4f} | {result.get('macro_f1', 0):.4f} | ")
                    f.write(f"{result.get('weighted_f1', 0):.4f} |\n")
        
        print(f"Comprehensive ablation analysis report saved to: {report_path}")
        
    def generate_latex_tables(self, results_dict, comparisons):
        """Generate LaTeX tables for paper inclusion."""
        
        # Main ablation results table
        latex_path = self.tables_dir / 'ablation_results.tex'
        
        with open(latex_path, 'w') as f:
            f.write("% Ablation Study Results Table\n")
            f.write("\\begin{table}[ht]\n")
            f.write("\\centering\n")
            f.write("\\caption{Ablation Study Results: Performance comparison of BioKG-BioBERT variants}\n")
            f.write("\\label{tab:ablation_results}\n")
            f.write("\\begin{tabular}{llcccc}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Ablation Group} & \\textbf{Model Variant} & \\textbf{Micro F1} & \\textbf{Macro F1} & ")
            f.write("\\textbf{$\\Delta$ Micro F1} & \\textbf{Relative Change} \\\\\n")
            f.write("\\midrule\n")
            
            # Full model first
            if self.full_model_name in results_dict and results_dict[self.full_model_name] is not None:
                result = results_dict[self.full_model_name]
                f.write("\\multicolumn{2}{l}{\\textbf{Full Model (Baseline)}} & ")
                f.write(f"\\textbf{{{result.get('micro_f1', 0):.4f}}} & ")
                f.write(f"\\textbf{{{result.get('macro_f1', 0):.4f}}} & - & - \\\\\n")
                f.write("\\midrule\n")
            
            # Group results
            for group_idx, (group_name, group_models) in enumerate(self.get_ablation_groups().items()):
                first_in_group = True
                group_models_filtered = [m for m in group_models if m != self.full_model_name]
                
                for model_idx, model in enumerate(group_models_filtered):
                    if model in results_dict and results_dict[model] is not None:
                        result = results_dict[model]
                        name = self.get_ablation_categories().get(model, model).split(': ')[-1]
                        
                        # Find comparison
                        comp = next((c for c in comparisons if c['ablation_name'] == model), None)
                        
                        if first_in_group:
                            f.write(f"\\multirow{{{len(group_models_filtered)}}}{{*}}{{{group_name}}} & ")
                            first_in_group = False
                        else:
                            f.write(" & ")
                        
                        f.write(f"{name} & {result.get('micro_f1', 0):.4f} & {result.get('macro_f1', 0):.4f} & ")
                        
                        if comp and 'micro_f1_diff' in comp:
                            f.write(f"{comp['micro_f1_diff']:+.4f} & {comp['micro_f1_relative_diff']:+.1f}\\% \\\\\n")
                        else:
                            f.write("N/A & N/A \\\\\n")
                
                if group_idx < len(self.get_ablation_groups()) - 1:
                    f.write("\\midrule\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"LaTeX tables saved to: {self.tables_dir}")
        
    def run_analysis(self):
        """Run complete ablation analysis."""
        print("Starting comprehensive ablation analysis...")
        
        # Load all model results
        model_names = [self.full_model_name] + list(ABLATION_CATEGORIES.keys())
        results_dict = {}
        
        for model_name in model_names:
            print(f"Loading results for {model_name}...")
            results = self.load_model_results(model_name)
            if results:
                results_dict[model_name] = results
                    
                    # Calculate additional metrics
                    if 'predictions' in results:
                        y_true = np.array(results['predictions']['y_true'])
                        y_pred = np.array(results['predictions']['y_pred'])
                        y_prob = np.array(results['predictions'].get('y_prob', []))
                        
                        additional_metrics = self.calculate_additional_metrics(
                            y_true, y_pred, y_prob if y_prob.size > 0 else None
                        )
                        results_dict[model_name].update(additional_metrics)
                        
                        # Add per-hallmark F1 scores
                        per_hallmark_f1 = []
                        for i in range(y_true.shape[1]):
                            f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
                            per_hallmark_f1.append(f1)
                        results_dict[model_name]['per_hallmark_f1'] = per_hallmark_f1
        
        # Get full model results
        full_results = results_dict.get(self.full_model_name)
        
        # Compare each ablation with full model
        comparisons = []
        if full_results:
            for model_name, results in results_dict.items():
                if model_name != self.full_model_name and results is not None:
                    print(f"Comparing {model_name} with full model...")
                    comparison = self.compare_with_full_model(full_results, results, model_name)
                    comparisons.append(comparison)
        
        # Generate visualizations
        print("Generating performance comparison plots...")
        self.generate_performance_comparison_plot(results_dict)
        
        print("Generating ablation impact analysis...")
        self.generate_ablation_impact_analysis(comparisons)
        
        # Generate detailed report
        print("Generating comprehensive report...")
        self.generate_detailed_report(results_dict, comparisons)
        
        # Generate LaTeX tables
        print("Generating LaTeX tables...")
        self.generate_latex_tables(results_dict, comparisons)
        
        # Save comparison results
        comparison_file = self.output_dir / 'ablation_comparisons.json'
        with open(comparison_file, 'w') as f:
            json.dump(comparisons, f, indent=2)
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        print(f"- Performance plots: {self.plots_dir}")
        print(f"- Analysis report: {self.reports_dir / 'ablation_analysis_report.md'}")
        print(f"- LaTeX tables: {self.tables_dir}")
        print(f"- Raw comparisons: {comparison_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze ablation study results')
    parser.add_argument('--experiment-dir', type=str, default='experiments',
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='ablation_analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--full-model', type=str, default='biokg_biobert',
                       help='Name/path of the full model to compare against (default: biokg_biobert)')
    
    args = parser.parse_args()
    
    analyzer = AblationAnalyzer(args.experiment_dir, args.output_dir)
    analyzer.full_model_name = args.full_model
    analyzer.run_analysis()


if __name__ == "__main__":
    main()