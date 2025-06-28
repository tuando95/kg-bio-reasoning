"""
Experiment Runner for BioKG-BioBERT

This script manages and runs all experiments including ablation studies,
hyperparameter search, and final evaluation.
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import json
import copy
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import train_model
from src.models.biokg_biobert import BioKGBioBERT
from src.data.dataset import HoCDataModule
from src.evaluation import Evaluator
from sklearn.metrics import f1_score, precision_recall_curve

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Manages experiments for BioKG-BioBERT including ablations and hyperparameter search.
    """
    
    def __init__(self, base_config_path: str, output_dir: str):
        """
        Initialize experiment runner.
        
        Args:
            base_config_path: Path to base configuration file
            output_dir: Directory for experiment outputs
        """
        # Load base configuration
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Turn off AMP globally
        self.base_config['experiment']['mixed_precision'] = False
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment tracking
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"experiments_{self.experiment_id}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Results tracking
        self.results = []
        
        logger.info(f"Initialized experiment runner")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Output directory: {self.experiment_dir}")
    
    def run_baseline(self):
        """Run baseline experiments."""
        logger.info("Running baseline experiments...")
        
        baseline_configs = [
            # BioBERT baseline (no KG, no bio attention)
            self._create_config(
                "baseline_biobert",
                {
                    'model.use_knowledge_graph': False,
                    'model.use_bio_attention': False,
                    'training.loss_weights.pathway_loss': 0.0,
                    'training.loss_weights.consistency_loss': 0.0
                }
            ),
            
            # BioBERT + Entity features
            self._create_config(
                "baseline_biobert_entities",
                {
                    'model.use_knowledge_graph': False,
                    'model.use_bio_attention': False,
                    'training.loss_weights.pathway_loss': 0.0,
                    'training.loss_weights.consistency_loss': 0.0
                }
            ),
            
            # BioBERT + Simple KG (no bio attention)
            self._create_config(
                "baseline_biobert_simple_kg",
                {
                    'model.use_knowledge_graph': True,
                    'model.use_bio_attention': False,
                    'training.loss_weights.pathway_loss': 0.0,
                    'training.loss_weights.consistency_loss': 0.0
                }
            )
        ]
        
        for config in baseline_configs:
            result = self._run_single_experiment(config)
            self.results.append(result)
    
    def run_ablation_studies(self):
        """Run comprehensive ablation studies."""
        logger.info("Running ablation studies...")
        
        ablation_configs = []
        
        # A1: Attention Mechanism Variants
        logger.info("Generating attention mechanism ablations...")
        ablation_configs.extend([
            self._create_config(
                "ablation_attention_full",
                {
                    'model.use_bio_attention': True,
                    'model.bio_attention.use_pathways': True
                }
            ),
            self._create_config(
                "ablation_attention_none",
                {
                    'model.use_bio_attention': False
                }
            ),
            self._create_config(
                "ablation_attention_entity_only",
                {
                    'model.use_bio_attention': True,
                    'model.bio_attention.use_pathways': False
                }
            )
        ])
        
        # A2: Knowledge Graph Integration Levels
        logger.info("Generating KG integration ablations...")
        ablation_configs.extend([
            self._create_config(
                "ablation_kg_none",
                {
                    'model.use_knowledge_graph': False
                }
            ),
            self._create_config(
                "ablation_kg_1hop",
                {
                    'model.use_knowledge_graph': True,
                    'knowledge_graph.graph_construction.max_hops': 1
                }
            ),
            self._create_config(
                "ablation_kg_2hop",
                {
                    'model.use_knowledge_graph': True,
                    'knowledge_graph.graph_construction.max_hops': 2
                }
            )
        ])
        
        # A3: Fusion Strategy Variants
        logger.info("Generating fusion strategy ablations...")
        for strategy in ['early', 'late', 'cross_modal']:
            ablation_configs.append(
                self._create_config(
                    f"ablation_fusion_{strategy}",
                    {
                        'model.fusion.strategy': strategy
                    }
                )
            )
        
        # A4: Multi-Task Learning Components
        logger.info("Generating multi-task ablations...")
        ablation_configs.extend([
            self._create_config(
                "ablation_multitask_hallmarks_only",
                {
                    'training.loss_weights.pathway_loss': 0.0,
                    'training.loss_weights.consistency_loss': 0.0
                }
            ),
            self._create_config(
                "ablation_multitask_with_pathway",
                {
                    'training.loss_weights.pathway_loss': 0.25,
                    'training.loss_weights.consistency_loss': 0.0
                }
            ),
            self._create_config(
                "ablation_multitask_with_consistency",
                {
                    'training.loss_weights.pathway_loss': 0.0,
                    'training.loss_weights.consistency_loss': 0.1
                }
            ),
            self._create_config(
                "ablation_multitask_full",
                {
                    'training.loss_weights.pathway_loss': 0.25,
                    'training.loss_weights.consistency_loss': 0.1
                }
            )
        ])
        
        # Run all ablation experiments
        for config in tqdm(ablation_configs, desc="Running ablations"):
            result = self._run_single_experiment(config)
            self.results.append(result)
    
    def run_hyperparameter_search(self):
        """Run hyperparameter sensitivity analysis."""
        if not self.base_config['hyperparameter_search']['enabled']:
            logger.info("Hyperparameter search disabled in config")
            return
        
        logger.info("Running hyperparameter search...")
        
        search_space = self.base_config['hyperparameter_search']['search_space']
        hyperparam_configs = []
        
        # Fusion weight search
        for fusion_weight in search_space['fusion_weight']:
            hyperparam_configs.append(
                self._create_config(
                    f"hyperparam_fusion_weight_{fusion_weight}",
                    {
                        'model.bio_attention.fusion_weight': fusion_weight
                    }
                )
            )
        
        # Loss weight search
        for pathway_weight in search_space['pathway_loss_weight']:
            for consistency_weight in search_space['consistency_loss_weight']:
                hyperparam_configs.append(
                    self._create_config(
                        f"hyperparam_loss_p{pathway_weight}_c{consistency_weight}",
                        {
                            'training.loss_weights.pathway_loss': pathway_weight,
                            'training.loss_weights.consistency_loss': consistency_weight
                        }
                    )
                )
        
        # GNN architecture search
        for num_layers in search_space['gnn_layers']:
            hyperparam_configs.append(
                self._create_config(
                    f"hyperparam_gnn_layers_{num_layers}",
                    {
                        'model.gnn.num_layers': num_layers
                    }
                )
            )
        
        # Learning rate search
        for lr in search_space['learning_rate']:
            hyperparam_configs.append(
                self._create_config(
                    f"hyperparam_lr_{lr}",
                    {
                        'training.learning_rate': lr
                    }
                )
            )
        
        # Batch size search
        for batch_size in search_space['batch_size']:
            hyperparam_configs.append(
                self._create_config(
                    f"hyperparam_batch_{batch_size}",
                    {
                        'training.batch_size': batch_size
                    }
                )
            )
        
        # Dropout rate search
        for dropout in search_space['dropout_rate']:
            hyperparam_configs.append(
                self._create_config(
                    f"hyperparam_dropout_{dropout}",
                    {
                        'model.dropout_rate': dropout
                    }
                )
            )
        
        # Run hyperparameter experiments
        for config in tqdm(hyperparam_configs, desc="Hyperparameter search"):
            result = self._run_single_experiment(config)
            self.results.append(result)
    
    def run_final_model(self, num_seeds: int = 5):
        """Run final model with multiple seeds for statistical significance."""
        logger.info(f"Running final model with {num_seeds} seeds...")
        
        final_results = []
        
        for seed in range(num_seeds):
            config = copy.deepcopy(self.base_config)
            config['experiment']['seed'] = 42 + seed
            config['experiment']['name'] = f"final_model_seed_{42 + seed}"
            
            result = self._run_single_experiment(config)
            final_results.append(result)
        
        # Compute statistics
        metrics = ['val_f1_macro', 'val_f1_micro', 'test_f1_macro', 'test_f1_micro']
        stats = {}
        
        for metric in metrics:
            values = [r.get(metric, 0) for r in final_results if metric in r]
            if values:
                stats[f"{metric}_mean"] = np.mean(values)
                stats[f"{metric}_std"] = np.std(values)
        
        # Save final results
        final_report = {
            'num_seeds': num_seeds,
            'individual_results': final_results,
            'statistics': stats
        }
        
        with open(self.experiment_dir / 'final_model_results.json', 'w') as f:
            json.dump(final_report, f, indent=2)
        
        logger.info(f"Final model results:")
        logger.info(f"  Test F1-Macro: {stats.get('test_f1_macro_mean', 0):.4f} ± {stats.get('test_f1_macro_std', 0):.4f}")
        logger.info(f"  Test F1-Micro: {stats.get('test_f1_micro_mean', 0):.4f} ± {stats.get('test_f1_micro_std', 0):.4f}")
    
    def _create_config(self, name: str, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Create modified configuration."""
        config = copy.deepcopy(self.base_config)
        
        # Apply modifications
        for path, value in modifications.items():
            self._set_nested_value(config, path, value)
        
        # Update experiment name
        config['experiment']['name'] = name
        
        # Add metadata
        config['experiment_metadata'] = {
            'name': name,
            'modifications': modifications,
            'experiment_id': self.experiment_id
        }
        
        return config
    
    def _set_nested_value(self, d: Dict, path: str, value: Any):
        """Set value in nested dictionary using dot notation."""
        keys = path.split('.')
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = value
    
    def _optimize_thresholds(self, checkpoint_path: Path, config: Dict) -> Dict[str, float]:
        """Optimize classification thresholds on validation set."""
        device = torch.device(config['experiment']['device'])
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Initialize model
        model_config = config['model'].copy()
        model_config['loss_weights'] = config['training']['loss_weights']
        model = BioKGBioBERT(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
        
        # Get validation predictions
        data_module = HoCDataModule(config)
        data_module.setup()
        val_loader = data_module.val_dataloader()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch_device = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch_device[key] = value.to(device)
                    elif key == 'graph_data':
                        batch_device[key] = value.to(device) if value is not None else None
                    elif key == 'biological_context':
                        moved_context = {}
                        for ctx_key, ctx_value in value.items():
                            if isinstance(ctx_value, torch.Tensor):
                                moved_context[ctx_key] = ctx_value.to(device)
                            else:
                                moved_context[ctx_key] = ctx_value
                        batch_device[key] = moved_context
                    else:
                        batch_device[key] = value
                
                outputs = model(**batch_device)
                predictions = torch.sigmoid(outputs['logits'])
                all_predictions.append(predictions.cpu())
                all_targets.append(batch_device['labels'].cpu())
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Find optimal thresholds
        optimal_thresholds = {}
        for i in range(11):
            y_true = all_targets[:, i].numpy()
            y_scores = all_predictions[:, i].numpy()
            
            if y_true.sum() == 0:
                optimal_thresholds[i] = 0.5
                continue
            
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            optimal_thresholds[i] = float(thresholds[best_idx] if best_idx < len(thresholds) else 0.5)
        
        # Calculate performance with optimal thresholds
        pred_optimal = np.zeros_like(all_predictions.numpy())
        for i in range(11):
            pred_optimal[:, i] = (all_predictions[:, i].numpy() >= optimal_thresholds[i]).astype(int)
        
        pred_default = (all_predictions >= 0.5).float().numpy()
        
        return {
            'f1_micro_optimal': f1_score(all_targets.numpy(), pred_optimal, average='micro'),
            'f1_macro_optimal': f1_score(all_targets.numpy(), pred_optimal, average='macro'),
            'optimal_thresholds': optimal_thresholds
        }
    
    def _run_single_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment."""
        experiment_name = config['experiment']['name']
        logger.info(f"Running experiment: {experiment_name}")
        
        # Save configuration
        config_path = self.experiment_dir / f"{experiment_name}_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        try:
            # Run training
            results = train_model(config)
            
            # Add threshold optimization results if available
            checkpoint_dir = Path(config['experiment']['checkpoint_dir']) / config['experiment']['name']
            best_checkpoint = checkpoint_dir / 'best.pt'
            
            if best_checkpoint.exists() and config.get('optimize_thresholds', True):
                logger.info(f"Running threshold optimization for {experiment_name}...")
                threshold_results = self._optimize_thresholds(best_checkpoint, config)
                results.update(threshold_results)
            
            # Add metadata
            results['experiment_name'] = experiment_name
            results['config_path'] = str(config_path)
            results['modifications'] = config.get('experiment_metadata', {}).get('modifications', {})
            
            # Save results
            results_path = self.experiment_dir / f"{experiment_name}_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            if 'f1_macro_optimal' in results:
                logger.info(f"Completed {experiment_name}: F1-Macro={results.get('f1_macro', 0):.4f} "
                           f"(Optimal: {results.get('f1_macro_optimal', 0):.4f})")
            else:
                logger.info(f"Completed {experiment_name}: F1-Macro={results.get('f1_macro', 0):.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in experiment {experiment_name}: {e}")
            return {
                'experiment_name': experiment_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def generate_report(self):
        """Generate comprehensive experiment report."""
        logger.info("Generating experiment report...")
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save as CSV
        df.to_csv(self.experiment_dir / 'all_results.csv', index=False)
        
        # Generate text report
        report_path = self.experiment_dir / 'experiment_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("BioKG-BioBERT Experiment Report\n")
            f.write(f"Experiment ID: {self.experiment_id}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Best performing model
            if 'f1_macro' in df.columns:
                best_idx = df['f1_macro'].idxmax()
                best_model = df.loc[best_idx]
                
                f.write("BEST PERFORMING MODEL\n")
                f.write("-" * 40 + "\n")
                f.write(f"Experiment: {best_model['experiment_name']}\n")
                f.write(f"F1-Macro: {best_model['f1_macro']:.4f}\n")
                f.write(f"F1-Micro: {best_model.get('f1_micro', 0):.4f}\n")
                
                if 'f1_macro_optimal' in best_model:
                    f.write(f"\nWith Optimal Thresholds:\n")
                    f.write(f"F1-Macro: {best_model['f1_macro_optimal']:.4f} "
                           f"(+{best_model['f1_macro_optimal'] - best_model['f1_macro']:.4f})\n")
                    f.write(f"F1-Micro: {best_model['f1_micro_optimal']:.4f} "
                           f"(+{best_model['f1_micro_optimal'] - best_model.get('f1_micro', 0):.4f})\n")
                f.write(f"Modifications: {best_model.get('modifications', {})}\n\n")
            
            # Ablation analysis
            f.write("ABLATION STUDY RESULTS\n")
            f.write("-" * 40 + "\n")
            
            ablation_results = df[df['experiment_name'].str.contains('ablation')]
            if not ablation_results.empty:
                ablation_summary = ablation_results.groupby(
                    ablation_results['experiment_name'].str.split('_').str[1]
                )['f1_macro'].agg(['mean', 'std', 'max'])
                f.write(ablation_summary.to_string())
                f.write("\n\n")
            
            # All results summary
            f.write("ALL EXPERIMENT RESULTS\n")
            f.write("-" * 40 + "\n")
            summary_cols = ['experiment_name', 'f1_macro', 'f1_micro', 'hamming_loss']
            available_cols = [col for col in summary_cols if col in df.columns]
            f.write(df[available_cols].to_string(index=False))
        
        logger.info(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run BioKG-BioBERT experiments")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to base configuration file')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Directory for experiment outputs')
    parser.add_argument('--run_baselines', action='store_true',
                       help='Run baseline experiments')
    parser.add_argument('--run_ablations', action='store_true',
                       help='Run ablation studies')
    parser.add_argument('--run_hyperparam', action='store_true',
                       help='Run hyperparameter search')
    parser.add_argument('--run_final', action='store_true',
                       help='Run final model with multiple seeds')
    parser.add_argument('--num_seeds', type=int, default=5,
                       help='Number of seeds for final model')
    parser.add_argument('--run_all', action='store_true',
                       help='Run all experiments')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ExperimentRunner(args.config, args.output_dir)
    
    # Run requested experiments
    if args.run_all or args.run_baselines:
        runner.run_baseline()
    
    if args.run_all or args.run_ablations:
        runner.run_ablation_studies()
    
    if args.run_all or args.run_hyperparam:
        runner.run_hyperparameter_search()
    
    if args.run_all or args.run_final:
        runner.run_final_model(args.num_seeds)
    
    # Generate report
    runner.generate_report()
    
    logger.info("All experiments completed!")


if __name__ == "__main__":
    main()