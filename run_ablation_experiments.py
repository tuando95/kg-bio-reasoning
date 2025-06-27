"""
Ablation Study Experiment Runner for BioKG-BioBERT

This script runs comprehensive ablation studies to evaluate the contribution
of each component in the BioKG-BioBERT architecture.
"""

import os
import logging
import yaml
import argparse
from typing import Dict, List, Any
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import itertools
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AblationExperimentRunner:
    """
    Manages and runs ablation experiments for BioKG-BioBERT.
    
    Ablation studies include:
    1. Attention mechanism variants
    2. Knowledge graph integration levels
    3. Fusion strategy variants
    4. Multi-task learning components
    """
    
    def __init__(self, base_config_path: str, output_dir: str):
        """
        Initialize the experiment runner.
        
        Args:
            base_config_path: Path to base configuration file
            output_dir: Directory to save experiment results
        """
        # Load base configuration
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment tracking
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"ablation_{self.experiment_id}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results tracking
        self.results = []
        
        logger.info(f"Initialized ablation experiment runner")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Output directory: {self.experiment_dir}")
    
    def generate_ablation_configs(self) -> List[Dict[str, Any]]:
        """
        Generate all ablation experiment configurations based on the base config.
        
        Returns:
            List of configuration dictionaries for each ablation experiment
        """
        ablation_configs = []
        ablation_settings = self.base_config['ablation_studies']
        
        # A1: Attention Mechanism Variants
        logger.info("Generating attention mechanism ablation configs...")
        for variant in ablation_settings['attention_variants']:
            config = self._create_ablation_config(
                name=f"attention_{variant['name']}",
                ablation_type="attention_mechanism",
                modifications={
                    'model.bio_attention.enabled': variant.get('bio_attention', True),
                    'model.bio_attention.use_pathways': variant.get('pathway_attention', True)
                }
            )
            ablation_configs.append(config)
        
        # A2: Knowledge Graph Integration Levels
        logger.info("Generating KG integration level ablation configs...")
        for variant in ablation_settings['kg_integration_levels']:
            modifications = {
                'knowledge_graph.enabled': variant.get('use_kg', True)
            }
            if variant.get('use_kg', True):
                modifications['knowledge_graph.graph_construction.max_hops'] = variant.get('max_hops', 2)
            
            config = self._create_ablation_config(
                name=f"kg_integration_{variant['name']}",
                ablation_type="kg_integration",
                modifications=modifications
            )
            ablation_configs.append(config)
        
        # A3: Fusion Strategy Variants
        logger.info("Generating fusion strategy ablation configs...")
        for strategy in ablation_settings['fusion_strategies']:
            config = self._create_ablation_config(
                name=f"fusion_{strategy}",
                ablation_type="fusion_strategy",
                modifications={
                    'model.fusion.strategy': strategy
                }
            )
            ablation_configs.append(config)
        
        # A4: Multi-Task Learning Components
        logger.info("Generating multi-task learning ablation configs...")
        for variant in ablation_settings['multitask_variants']:
            config = self._create_ablation_config(
                name=f"multitask_{variant['name']}",
                ablation_type="multitask_learning",
                modifications={
                    'training.loss_weights.hallmark_loss': 1.0 if variant['hallmark_loss'] else 0.0,
                    'training.loss_weights.pathway_loss': 0.25 if variant.get('pathway_loss', False) else 0.0,
                    'training.loss_weights.consistency_loss': 0.1 if variant.get('consistency_loss', False) else 0.0
                }
            )
            ablation_configs.append(config)
        
        # Knowledge Source Ablations
        logger.info("Generating knowledge source ablation configs...")
        databases = ['KEGG', 'STRING', 'Reactome', 'GO']
        
        # Single database ablations
        for db in databases:
            config = self._create_ablation_config(
                name=f"knowledge_source_{db.lower()}_only",
                ablation_type="knowledge_source",
                modifications={
                    f'knowledge_graph.databases.{i}.enabled': (db_config['name'] == db)
                    for i, db_config in enumerate(self.base_config['knowledge_graph']['databases'])
                }
            )
            ablation_configs.append(config)
        
        # Database combination ablations
        for r in range(2, len(databases)):
            for combo in itertools.combinations(databases, r):
                combo_name = '_'.join([db.lower() for db in combo])
                config = self._create_ablation_config(
                    name=f"knowledge_source_{combo_name}",
                    ablation_type="knowledge_source",
                    modifications={
                        f'knowledge_graph.databases.{i}.enabled': (db_config['name'] in combo)
                        for i, db_config in enumerate(self.base_config['knowledge_graph']['databases'])
                    }
                )
                ablation_configs.append(config)
        
        # Hyperparameter sensitivity configs
        if self.base_config['hyperparameter_search']['enabled']:
            logger.info("Generating hyperparameter sensitivity configs...")
            search_space = self.base_config['hyperparameter_search']['search_space']
            
            # Fusion weight sensitivity
            for fusion_weight in search_space['fusion_weight']:
                config = self._create_ablation_config(
                    name=f"hyperparam_fusion_weight_{fusion_weight}",
                    ablation_type="hyperparameter",
                    modifications={
                        'model.bio_attention.fusion_weight': fusion_weight
                    }
                )
                ablation_configs.append(config)
            
            # Loss weight sensitivity
            for pathway_weight in search_space['pathway_loss_weight']:
                for consistency_weight in search_space['consistency_loss_weight']:
                    config = self._create_ablation_config(
                        name=f"hyperparam_loss_weights_p{pathway_weight}_c{consistency_weight}",
                        ablation_type="hyperparameter",
                        modifications={
                            'training.loss_weights.pathway_loss': pathway_weight,
                            'training.loss_weights.consistency_loss': consistency_weight
                        }
                    )
                    ablation_configs.append(config)
        
        logger.info(f"Generated {len(ablation_configs)} ablation configurations")
        return ablation_configs
    
    def _create_ablation_config(self, name: str, ablation_type: str, 
                               modifications: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an ablation configuration by modifying the base config.
        
        Args:
            name: Name of the ablation experiment
            ablation_type: Type of ablation (attention, kg_integration, etc.)
            modifications: Dictionary of config paths and values to modify
            
        Returns:
            Complete configuration dictionary for the ablation
        """
        import copy
        config = copy.deepcopy(self.base_config)
        
        # Apply modifications
        for path, value in modifications.items():
            self._set_nested_dict_value(config, path, value)
        
        # Add ablation metadata
        config['ablation_metadata'] = {
            'name': name,
            'type': ablation_type,
            'modifications': modifications,
            'experiment_id': self.experiment_id
        }
        
        # Update experiment name
        config['experiment']['name'] = f"{config['experiment']['name']}_{name}"
        
        return config
    
    def _set_nested_dict_value(self, d: Dict, path: str, value: Any):
        """Set a value in a nested dictionary using dot notation path"""
        keys = path.split('.')
        for key in keys[:-1]:
            if key.isdigit():
                key = int(key)
            d = d[key]
        
        final_key = keys[-1]
        if final_key.isdigit():
            final_key = int(final_key)
        d[final_key] = value
    
    def run_single_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single ablation experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Dictionary containing experiment results
        """
        experiment_name = config['ablation_metadata']['name']
        logger.info(f"Running experiment: {experiment_name}")
        
        # Save configuration
        config_path = self.experiment_dir / f"{experiment_name}_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        try:
            # Import and run the actual experiment
            # This is a placeholder - in practice would import and run the model
            from src.train import train_model
            
            results = train_model(config)
            
            # Add experiment metadata to results
            results['experiment_name'] = experiment_name
            results['ablation_type'] = config['ablation_metadata']['type']
            results['modifications'] = config['ablation_metadata']['modifications']
            results['config_path'] = str(config_path)
            
            # Save individual experiment results
            results_path = self.experiment_dir / f"{experiment_name}_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Completed experiment: {experiment_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error running experiment {experiment_name}: {e}")
            return {
                'experiment_name': experiment_name,
                'ablation_type': config['ablation_metadata']['type'],
                'error': str(e),
                'status': 'failed'
            }
    
    def run_all_experiments(self, parallel: bool = True, num_workers: int = None):
        """
        Run all ablation experiments.
        
        Args:
            parallel: Whether to run experiments in parallel
            num_workers: Number of parallel workers (None for CPU count)
        """
        # Generate all ablation configurations
        ablation_configs = self.generate_ablation_configs()
        
        logger.info(f"Starting {len(ablation_configs)} ablation experiments")
        
        if parallel:
            if num_workers is None:
                num_workers = min(mp.cpu_count(), 4)
            
            logger.info(f"Running experiments in parallel with {num_workers} workers")
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_config = {
                    executor.submit(self.run_single_experiment, config): config
                    for config in ablation_configs
                }
                
                for future in tqdm(as_completed(future_to_config), 
                                 total=len(ablation_configs),
                                 desc="Running ablations"):
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                    except Exception as e:
                        logger.error(f"Experiment failed: {e}")
                        self.results.append({
                            'experiment_name': config['ablation_metadata']['name'],
                            'status': 'failed',
                            'error': str(e)
                        })
        else:
            logger.info("Running experiments sequentially")
            for config in tqdm(ablation_configs, desc="Running ablations"):
                result = self.run_single_experiment(config)
                self.results.append(result)
        
        # Save combined results
        self._save_combined_results()
        
        # Generate summary report
        self._generate_summary_report()
        
        logger.info("All ablation experiments completed")
    
    def _save_combined_results(self):
        """Save all experiment results in a combined format"""
        # Save as JSON
        combined_results_path = self.experiment_dir / "all_results.json"
        with open(combined_results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Convert to DataFrame and save as CSV
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.experiment_dir / "all_results.csv", index=False)
        
        logger.info(f"Saved combined results to {combined_results_path}")
    
    def _generate_summary_report(self):
        """Generate a summary report of all ablation experiments"""
        report_path = self.experiment_dir / "ablation_summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"BioKG-BioBERT Ablation Study Summary Report\n")
            f.write(f"Experiment ID: {self.experiment_id}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Group results by ablation type
            results_by_type = {}
            for result in self.results:
                ablation_type = result.get('ablation_type', 'unknown')
                if ablation_type not in results_by_type:
                    results_by_type[ablation_type] = []
                results_by_type[ablation_type].append(result)
            
            # Write summary for each ablation type
            for ablation_type, type_results in results_by_type.items():
                f.write(f"\n{ablation_type.upper()} ABLATIONS\n")
                f.write("-" * 40 + "\n")
                
                # Sort by performance metric (assuming F1 macro)
                valid_results = [r for r in type_results if 'val_f1_macro' in r]
                valid_results.sort(key=lambda x: x.get('val_f1_macro', 0), reverse=True)
                
                for result in valid_results:
                    f.write(f"\nExperiment: {result['experiment_name']}\n")
                    f.write(f"  Val F1 Macro: {result.get('val_f1_macro', 'N/A'):.4f}\n")
                    f.write(f"  Val F1 Micro: {result.get('val_f1_micro', 'N/A'):.4f}\n")
                    f.write(f"  Modifications: {result.get('modifications', {})}\n")
            
            # Write best performing configuration
            f.write("\n\nBEST PERFORMING CONFIGURATIONS\n")
            f.write("=" * 40 + "\n")
            
            all_valid_results = [r for r in self.results if 'val_f1_macro' in r]
            if all_valid_results:
                all_valid_results.sort(key=lambda x: x.get('val_f1_macro', 0), reverse=True)
                
                f.write(f"\nTop 5 Configurations by F1 Macro:\n")
                for i, result in enumerate(all_valid_results[:5]):
                    f.write(f"\n{i+1}. {result['experiment_name']}\n")
                    f.write(f"   F1 Macro: {result['val_f1_macro']:.4f}\n")
                    f.write(f"   Type: {result['ablation_type']}\n")
        
        logger.info(f"Generated summary report: {report_path}")
    
    def analyze_ablation_impact(self):
        """
        Analyze the impact of each ablation on model performance.
        
        This method computes statistics about how each component affects performance.
        """
        analysis_results = {}
        
        # Find baseline performance (full model)
        baseline = None
        for result in self.results:
            if result.get('experiment_name') == 'attention_full_biokg':
                baseline = result.get('val_f1_macro', 0)
                break
        
        if baseline is None:
            logger.warning("No baseline (full model) results found")
            return
        
        # Analyze impact by ablation type
        results_by_type = {}
        for result in self.results:
            ablation_type = result.get('ablation_type', 'unknown')
            if ablation_type not in results_by_type:
                results_by_type[ablation_type] = []
            
            if 'val_f1_macro' in result:
                impact = result['val_f1_macro'] - baseline
                results_by_type[ablation_type].append({
                    'name': result['experiment_name'],
                    'performance': result['val_f1_macro'],
                    'impact': impact,
                    'relative_impact': impact / baseline * 100
                })
        
        # Compute statistics for each ablation type
        for ablation_type, impacts in results_by_type.items():
            if impacts:
                performances = [r['performance'] for r in impacts]
                relative_impacts = [r['relative_impact'] for r in impacts]
                
                analysis_results[ablation_type] = {
                    'mean_performance': np.mean(performances),
                    'std_performance': np.std(performances),
                    'min_performance': np.min(performances),
                    'max_performance': np.max(performances),
                    'mean_relative_impact': np.mean(relative_impacts),
                    'most_important_component': min(impacts, key=lambda x: x['impact'])
                }
        
        # Save analysis results
        analysis_path = self.experiment_dir / "ablation_impact_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info(f"Saved ablation impact analysis to {analysis_path}")
        
        return analysis_results


def main():
    parser = argparse.ArgumentParser(description="Run BioKG-BioBERT ablation experiments")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to base configuration file')
    parser.add_argument('--output_dir', type=str, default='experiments/ablations',
                       help='Directory to save experiment results')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Run experiments in parallel')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze existing results without running experiments')
    
    args = parser.parse_args()
    
    # Initialize experiment runner
    runner = AblationExperimentRunner(args.config, args.output_dir)
    
    if args.analyze_only:
        # Load existing results and analyze
        logger.info("Analyzing existing results...")
        # This would load results from disk
        runner.analyze_ablation_impact()
    else:
        # Run all experiments
        runner.run_all_experiments(
            parallel=args.parallel,
            num_workers=args.num_workers
        )
        
        # Analyze impact
        runner.analyze_ablation_impact()


if __name__ == "__main__":
    main()