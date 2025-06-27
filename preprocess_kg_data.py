#!/usr/bin/env python3
"""
Pre-process and Cache Knowledge Graphs for All Dataset Samples

This script builds knowledge graphs for all samples in the dataset before training,
avoiding redundant API calls and speeding up experiments.
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
import pickle
import json
from tqdm import tqdm
from datasets import load_dataset
import asyncio
from datetime import datetime
import pandas as pd
from collections import defaultdict

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.kg_construction import BiologicalKGPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeGraphPreprocessor:
    """
    Pre-processes and caches knowledge graphs for the entire dataset.
    """
    
    # Hallmark label mapping (exact names from the dataset)
    HALLMARK_LABELS = {
        0: "evading_growth_suppressors",
        1: "tumor_promoting_inflammation",
        2: "enabling_replicative_immortality",
        3: "cellular_energetics",
        4: "resisting_cell_death",
        5: "activating_invasion_metastasis",
        6: "genomic_instability",
        7: "none",
        8: "inducing_angiogenesis",
        9: "sustaining_proliferative_signaling",
        10: "avoiding_immune_destruction"
    }
    
    def __init__(self, config_path: str, cache_dir: str = "cache/kg_preprocessed"):
        """
        Initialize the preprocessor.
        
        Args:
            config_path: Path to configuration file
            cache_dir: Directory to save cached knowledge graphs
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize KG pipeline
        logger.info("Initializing Knowledge Graph Pipeline...")
        self.kg_pipeline = BiologicalKGPipeline(self.config['knowledge_graph'])
        
        # Track statistics
        self.stats = defaultdict(int)
        
    def preprocess_all_splits(self):
        """Process all dataset splits (train, validation, test)."""
        splits = ['train', 'validation', 'test']
        
        for split in splits:
            logger.info(f"\nProcessing {split} split...")
            self.preprocess_split(split)
        
        # Save statistics
        self._save_statistics()
        
        logger.info("\nPreprocessing completed!")
        self._print_summary()
    
    def preprocess_split(self, split: str):
        """
        Process a single dataset split.
        
        Args:
            split: Dataset split name ('train', 'validation', 'test')
        """
        # Create split-specific cache directory
        split_cache_dir = self.cache_dir / split
        split_cache_dir.mkdir(exist_ok=True)
        
        # Load dataset
        logger.info(f"Loading {split} dataset...")
        dataset = load_dataset("qanastek/HoC", split=split)
        
        # Check for existing cache
        cache_index_path = split_cache_dir / "index.json"
        if cache_index_path.exists():
            with open(cache_index_path, 'r') as f:
                cache_index = json.load(f)
            logger.info(f"Found existing cache with {len(cache_index)} entries")
        else:
            cache_index = {}
        
        # Process samples
        new_entries = 0
        failed_entries = 0
        
        for idx in tqdm(range(len(dataset)), desc=f"Building KGs for {split}"):
            # Skip if already cached
            if str(idx) in cache_index:
                self.stats['cached'] += 1
                continue
            
            try:
                sample = dataset[idx]
                
                # Extract text and labels
                text = sample['text']
                labels = sample['label']  # 'label' not 'labels' in HoC dataset
                
                # Get hallmarks for this sample (convert to lowercase with underscores for KG processing)
                hallmarks = [
                    self.HALLMARK_LABELS[i].lower().replace(' ', '_').replace('and_', '') 
                    for i in labels 
                    if i != 7  # Skip "none"
                ]
                
                # Process through KG pipeline
                kg_output = self.kg_pipeline.process_text(text, hallmarks)
                
                # Prepare data to cache
                cache_data = {
                    'idx': idx,
                    'text': text,
                    'labels': labels,
                    'hallmarks': hallmarks,
                    'entities': self._serialize_entities(kg_output.entities),
                    'knowledge_graph': self._serialize_graph(kg_output.knowledge_graph),
                    'stats': {
                        'num_entities': len(kg_output.entities),
                        'num_nodes': kg_output.knowledge_graph.number_of_nodes(),
                        'num_edges': kg_output.knowledge_graph.number_of_edges(),
                    }
                }
                
                # Save individual cache file
                cache_file = split_cache_dir / f"sample_{idx}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                
                # Update index
                cache_index[str(idx)] = {
                    'file': f"sample_{idx}.pkl",
                    'stats': cache_data['stats']
                }
                
                new_entries += 1
                self.stats['processed'] += 1
                
                # Update statistics
                self.stats['total_entities'] += cache_data['stats']['num_entities']
                self.stats['total_nodes'] += cache_data['stats']['num_nodes']
                self.stats['total_edges'] += cache_data['stats']['num_edges']
                
                # Save index periodically
                if new_entries % 100 == 0:
                    with open(cache_index_path, 'w') as f:
                        json.dump(cache_index, f)
                
            except Exception as e:
                logger.error(f"Failed to process sample {idx}: {e}")
                failed_entries += 1
                self.stats['failed'] += 1
        
        # Save final index
        with open(cache_index_path, 'w') as f:
            json.dump(cache_index, f)
        
        logger.info(f"Completed {split}: {new_entries} new, {failed_entries} failed")
    
    def _serialize_entities(self, entities):
        """Serialize entity objects for caching."""
        return [
            {
                'text': e.text,
                'start': e.start,
                'end': e.end,
                'entity_type': e.entity_type,
                'normalized_ids': e.normalized_ids,
                'confidence': e.confidence,
                'context': e.context
            }
            for e in entities
        ]
    
    def _serialize_graph(self, graph):
        """Serialize NetworkX graph for caching."""
        import networkx as nx
        
        return {
            'nodes': list(graph.nodes(data=True)),
            'edges': list(graph.edges(data=True)),
            'graph': graph.graph
        }
    
    def verify_cache(self):
        """Verify the integrity of cached data."""
        logger.info("Verifying cache integrity...")
        
        splits = ['train', 'validation', 'test']
        total_verified = 0
        total_errors = 0
        
        for split in splits:
            split_cache_dir = self.cache_dir / split
            cache_index_path = split_cache_dir / "index.json"
            
            if not cache_index_path.exists():
                logger.warning(f"No cache index found for {split}")
                continue
            
            with open(cache_index_path, 'r') as f:
                cache_index = json.load(f)
            
            logger.info(f"Verifying {len(cache_index)} entries in {split}...")
            
            for idx, info in tqdm(cache_index.items(), desc=f"Verifying {split}"):
                cache_file = split_cache_dir / info['file']
                
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Basic integrity checks
                    assert 'knowledge_graph' in data
                    assert 'entities' in data
                    assert data['idx'] == int(idx)
                    
                    total_verified += 1
                    
                except Exception as e:
                    logger.error(f"Error verifying {cache_file}: {e}")
                    total_errors += 1
        
        logger.info(f"Verification complete: {total_verified} valid, {total_errors} errors")
    
    def generate_statistics_report(self):
        """Generate detailed statistics about the cached knowledge graphs."""
        logger.info("Generating statistics report...")
        
        all_stats = []
        
        for split in ['train', 'validation', 'test']:
            split_cache_dir = self.cache_dir / split
            cache_index_path = split_cache_dir / "index.json"
            
            if not cache_index_path.exists():
                continue
            
            with open(cache_index_path, 'r') as f:
                cache_index = json.load(f)
            
            # Collect statistics
            for idx, info in cache_index.items():
                stats = info['stats']
                stats['split'] = split
                stats['idx'] = idx
                all_stats.append(stats)
        
        # Create DataFrame
        df = pd.DataFrame(all_stats)
        
        # Generate report
        report_path = self.cache_dir / "kg_statistics_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Knowledge Graph Statistics Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Average entities per sample: {df['num_entities'].mean():.2f}\n")
            f.write(f"Average nodes per KG: {df['num_nodes'].mean():.2f}\n")
            f.write(f"Average edges per KG: {df['num_edges'].mean():.2f}\n\n")
            
            # Per-split statistics
            f.write("Per-Split Statistics:\n")
            for split in ['train', 'validation', 'test']:
                split_df = df[df['split'] == split]
                if len(split_df) > 0:
                    f.write(f"\n{split.upper()}:\n")
                    f.write(f"  Samples: {len(split_df)}\n")
                    f.write(f"  Entities: {split_df['num_entities'].mean():.2f} ± {split_df['num_entities'].std():.2f}\n")
                    f.write(f"  Nodes: {split_df['num_nodes'].mean():.2f} ± {split_df['num_nodes'].std():.2f}\n")
                    f.write(f"  Edges: {split_df['num_edges'].mean():.2f} ± {split_df['num_edges'].std():.2f}\n")
            
            # Distribution analysis
            f.write("\n\nDistribution Analysis:\n")
            f.write(f"Min entities: {df['num_entities'].min()}\n")
            f.write(f"Max entities: {df['num_entities'].max()}\n")
            f.write(f"Min nodes: {df['num_nodes'].min()}\n")
            f.write(f"Max nodes: {df['num_nodes'].max()}\n")
            f.write(f"Min edges: {df['num_edges'].min()}\n")
            f.write(f"Max edges: {df['num_edges'].max()}\n")
        
        # Save detailed statistics as CSV
        df.to_csv(self.cache_dir / "kg_statistics.csv", index=False)
        
        logger.info(f"Statistics report saved to {report_path}")
    
    def _save_statistics(self):
        """Save processing statistics."""
        stats_path = self.cache_dir / "preprocessing_stats.json"
        
        stats_dict = dict(self.stats)
        stats_dict['timestamp'] = datetime.now().isoformat()
        
        with open(stats_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
    
    def _print_summary(self):
        """Print preprocessing summary."""
        print("\nPreprocessing Summary:")
        print("=" * 40)
        print(f"Total processed: {self.stats['processed']}")
        print(f"From cache: {self.stats['cached']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Total entities: {self.stats['total_entities']}")
        print(f"Total nodes: {self.stats['total_nodes']}")
        print(f"Total edges: {self.stats['total_edges']}")
        
        if self.stats['processed'] > 0:
            print(f"\nAverages per sample:")
            print(f"  Entities: {self.stats['total_entities'] / self.stats['processed']:.2f}")
            print(f"  Nodes: {self.stats['total_nodes'] / self.stats['processed']:.2f}")
            print(f"  Edges: {self.stats['total_edges'] / self.stats['processed']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Pre-process knowledge graphs for HoC dataset")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--cache_dir', type=str, default='cache/kg_preprocessed',
                       help='Directory to save cached knowledge graphs')
    parser.add_argument('--verify', action='store_true',
                       help='Verify existing cache integrity')
    parser.add_argument('--stats', action='store_true',
                       help='Generate statistics report')
    parser.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'],
                       help='Dataset splits to process')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = KnowledgeGraphPreprocessor(args.config, args.cache_dir)
    
    if args.verify:
        # Verify existing cache
        preprocessor.verify_cache()
    elif args.stats:
        # Generate statistics report
        preprocessor.generate_statistics_report()
    else:
        # Process all splits
        preprocessor.preprocess_all_splits()
        
        # Generate statistics after processing
        preprocessor.generate_statistics_report()


if __name__ == "__main__":
    main()