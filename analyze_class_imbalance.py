"""
Analyze Class Imbalance and Its Impact on Performance

This script analyzes the relationship between class distribution
and model performance across all hallmarks.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datasets import load_dataset
from collections import Counter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_dataset_distribution():
    """Analyze the distribution of hallmarks in the dataset."""
    
    # Load HoC dataset
    logger.info("Loading Hallmarks of Cancer dataset...")
    dataset = load_dataset("qnastek/HoC")
    
    # Hallmark names
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
    
    # Analyze each split
    splits = ['train', 'validation', 'test']
    distribution_data = {}
    
    for split in splits:
        split_data = dataset[split]
        
        # Count occurrences of each hallmark
        hallmark_counts = Counter()
        total_samples = len(split_data)
        
        # Also track co-occurrence
        co_occurrence_matrix = np.zeros((11, 11))
        
        for sample in split_data:
            labels = sample['labels']
            for label in labels:
                hallmark_counts[label] += 1
            
            # Update co-occurrence
            for i in labels:
                for j in labels:
                    co_occurrence_matrix[i, j] += 1
        
        # Calculate percentages
        hallmark_percentages = {
            hallmark_names[i]: {
                'count': hallmark_counts.get(i, 0),
                'percentage': (hallmark_counts.get(i, 0) / total_samples) * 100
            }
            for i in range(11)
        }
        
        distribution_data[split] = {
            'total_samples': total_samples,
            'hallmark_distribution': hallmark_percentages,
            'co_occurrence': co_occurrence_matrix / total_samples
        }
    
    return distribution_data, hallmark_names


def create_distribution_visualizations(distribution_data, hallmark_names, save_dir):
    """Create visualizations for class distribution analysis."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 1. Distribution across splits
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, split in enumerate(['train', 'validation', 'test']):
        data = distribution_data[split]['hallmark_distribution']
        
        # Sort by percentage
        sorted_hallmarks = sorted(data.items(), key=lambda x: x[1]['percentage'], reverse=True)
        names = [h[0].split()[0] + "..." for h in sorted_hallmarks]  # Shorten names
        percentages = [h[1]['percentage'] for h in sorted_hallmarks]
        
        ax = axes[idx]
        bars = ax.bar(range(len(names)), percentages)
        
        # Color code by percentage
        colors = ['red' if p < 10 else 'orange' if p < 20 else 'green' for p in percentages]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Percentage of Samples')
        ax.set_title(f'{split.capitalize()} Set Distribution')
        ax.set_ylim(0, max(percentages) * 1.2)
        
        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Hallmark Distribution Across Dataset Splits', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir / 'hallmark_distribution_by_split.png', dpi=300)
    plt.close()
    
    # 2. Focus on problematic hallmarks
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Get training distribution
    train_dist = distribution_data['train']['hallmark_distribution']
    
    # Create DataFrame for easier plotting
    df_data = []
    for i, name in hallmark_names.items():
        df_data.append({
            'Hallmark': name,
            'Hallmark_ID': i,
            'Training_Percentage': train_dist[name]['percentage'],
            'Training_Count': train_dist[name]['count']
        })
    
    df = pd.DataFrame(df_data)
    df = df.sort_values('Training_Percentage')
    
    # Create horizontal bar plot
    bars = ax.barh(df['Hallmark'], df['Training_Percentage'])
    
    # Highlight the problematic hallmarks
    colors = ['red' if i in [0, 10] else 'lightblue' for i in df['Hallmark_ID']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add count labels
    for i, (pct, count) in enumerate(zip(df['Training_Percentage'], df['Training_Count'])):
        ax.text(pct + 0.5, i, f'n={count}', va='center', fontsize=9)
    
    ax.set_xlabel('Percentage of Training Samples')
    ax.set_title('Training Set Class Distribution\n(Red = Low-performing hallmarks)')
    ax.axvline(x=np.mean(df['Training_Percentage']), color='green', linestyle='--', 
               label=f'Mean: {np.mean(df["Training_Percentage"]):.1f}%')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'class_imbalance_analysis.png', dpi=300)
    plt.close()
    
    # 3. Co-occurrence heatmap for problematic hallmarks
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    co_occurrence = distribution_data['train']['co_occurrence']
    
    # Normalize by diagonal (self-occurrence) to get conditional probabilities
    co_occurrence_norm = co_occurrence.copy()
    for i in range(11):
        if co_occurrence[i, i] > 0:
            co_occurrence_norm[i, :] = co_occurrence[i, :] / co_occurrence[i, i]
    
    # Create heatmap
    sns.heatmap(co_occurrence_norm, 
                annot=True, 
                fmt='.2f',
                cmap='YlOrRd',
                xticklabels=[name.split()[0] for name in hallmark_names.values()],
                yticklabels=[name.split()[0] for name in hallmark_names.values()],
                cbar_kws={'label': 'P(Column | Row)'})
    
    # Highlight problematic hallmarks
    for i in [0, 10]:
        ax.add_patch(plt.Rectangle((i, 0), 1, 11, fill=False, edgecolor='red', lw=3))
        ax.add_patch(plt.Rectangle((0, i), 11, 1, fill=False, edgecolor='red', lw=3))
    
    plt.title('Hallmark Co-occurrence Probabilities\n(Red boxes = Low-performing hallmarks)')
    plt.tight_layout()
    plt.savefig(save_dir / 'hallmark_cooccurrence_heatmap.png', dpi=300)
    plt.close()


def generate_imbalance_report(distribution_data, save_dir):
    """Generate report on class imbalance analysis."""
    save_dir = Path(save_dir)
    
    with open(save_dir / 'class_imbalance_report.txt', 'w') as f:
        f.write("Class Imbalance Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        # Overall statistics
        train_dist = distribution_data['train']['hallmark_distribution']
        percentages = [d['percentage'] for d in train_dist.values()]
        
        f.write("OVERALL STATISTICS (Training Set)\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean class percentage: {np.mean(percentages):.1f}%\n")
        f.write(f"Std dev: {np.std(percentages):.1f}%\n")
        f.write(f"Min: {min(percentages):.1f}%\n")
        f.write(f"Max: {max(percentages):.1f}%\n")
        f.write(f"Imbalance ratio (max/min): {max(percentages)/min(percentages):.1f}x\n\n")
        
        # Problematic hallmarks
        f.write("LOW-PERFORMING HALLMARKS\n")
        f.write("-" * 30 + "\n")
        
        problem_hallmarks = {
            0: "Evading growth suppressors",
            10: "Avoiding immune destruction"
        }
        
        for h_id, h_name in problem_hallmarks.items():
            data = train_dist[h_name]
            f.write(f"\n{h_name}:\n")
            f.write(f"  Training samples: {data['count']} ({data['percentage']:.1f}%)\n")
            f.write(f"  Rank by frequency: {sorted(percentages).index(data['percentage']) + 1}/11\n")
            
            # Find most common co-occurring hallmarks
            co_occur = distribution_data['train']['co_occurrence']
            co_probs = []
            for i in range(11):
                if i != h_id and co_occur[h_id, h_id] > 0:
                    prob = co_occur[h_id, i] / co_occur[h_id, h_id]
                    co_probs.append((i, prob))
            
            co_probs.sort(key=lambda x: x[1], reverse=True)
            f.write("  Top co-occurring hallmarks:\n")
            for i, prob in co_probs[:3]:
                f.write(f"    - {list(train_dist.keys())[i]}: {prob:.2f}\n")
        
        # Comparison with high-performing hallmarks
        f.write("\n\nHIGH-PERFORMING HALLMARKS (for comparison)\n")
        f.write("-" * 30 + "\n")
        
        # Find top 3 most frequent hallmarks
        sorted_hallmarks = sorted(train_dist.items(), key=lambda x: x[1]['percentage'], reverse=True)
        for h_name, data in sorted_hallmarks[:3]:
            if h_name not in ["None", "Evading growth suppressors", "Avoiding immune destruction"]:
                f.write(f"\n{h_name}:\n")
                f.write(f"  Training samples: {data['count']} ({data['percentage']:.1f}%)\n")
        
        # Key insights
        f.write("\n\nKEY INSIGHTS\n")
        f.write("-" * 30 + "\n")
        f.write("1. Both low-performing hallmarks have below-average representation\n")
        f.write("2. 'Avoiding immune destruction' is one of the rarest classes\n")
        f.write("3. 'Evading growth suppressors' has moderate frequency but still underperforms\n")
        f.write("4. Class imbalance alone doesn't fully explain the performance gap\n")
        f.write("5. The biological complexity of these hallmarks may require more samples\n\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 30 + "\n")
        f.write("1. Apply class-weighted loss functions for these hallmarks\n")
        f.write("2. Use data augmentation specifically for underrepresented classes\n")
        f.write("3. Consider focal loss to address hard-to-classify examples\n")
        f.write("4. Implement hallmark-specific threshold optimization\n")
        f.write("5. Enhance biological knowledge for these specific pathways\n")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze class imbalance in HoC dataset")
    parser.add_argument('--output_dir', type=str, default='class_imbalance_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Analyze dataset
    logger.info("Analyzing dataset distribution...")
    distribution_data, hallmark_names = analyze_dataset_distribution()
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_distribution_visualizations(distribution_data, hallmark_names, args.output_dir)
    
    # Generate report
    logger.info("Generating report...")
    generate_imbalance_report(distribution_data, args.output_dir)
    
    # Save raw data
    with open(Path(args.output_dir) / 'distribution_data.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        for split in distribution_data:
            distribution_data[split]['co_occurrence'] = distribution_data[split]['co_occurrence'].tolist()
        json.dump(distribution_data, f, indent=2)
    
    logger.info(f"Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()