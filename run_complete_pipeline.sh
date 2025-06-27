#!/bin/bash
# Complete pipeline for BioKG-BioBERT experiments

# Exit on error
set -e

# Configuration
CONFIG_FILE="configs/default_config.yaml"
CACHE_DIR="cache/kg_preprocessed"
EXPERIMENT_DIR="experiments"
LOG_DIR="logs"

# Create directories
mkdir -p $CACHE_DIR $EXPERIMENT_DIR $LOG_DIR

echo "=========================================="
echo "BioKG-BioBERT Complete Pipeline"
echo "=========================================="
echo

# Step 1: Pre-process and cache knowledge graphs
echo "Step 1: Pre-processing knowledge graphs..."
echo "This will cache KGs for all dataset samples to speed up experiments."
echo

if [ -d "$CACHE_DIR/train" ] && [ -d "$CACHE_DIR/validation" ] && [ -d "$CACHE_DIR/test" ]; then
    echo "Cache already exists. Verifying integrity..."
    python preprocess_kg_data.py --config $CONFIG_FILE --verify
    
    read -p "Do you want to regenerate the cache? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python preprocess_kg_data.py --config $CONFIG_FILE
    fi
else
    echo "Building knowledge graph cache..."
    python preprocess_kg_data.py --config $CONFIG_FILE
fi

# Generate statistics report
echo
echo "Generating KG statistics report..."
python preprocess_kg_data.py --config $CONFIG_FILE --stats

# Step 2: Run baseline experiments
echo
echo "=========================================="
echo "Step 2: Running baseline experiments..."
echo "=========================================="
echo

python run_experiments.py \
    --config $CONFIG_FILE \
    --output_dir $EXPERIMENT_DIR \
    --run_baselines \
    2>&1 | tee $LOG_DIR/baselines.log

# Step 3: Run ablation studies
echo
echo "=========================================="
echo "Step 3: Running ablation studies..."
echo "=========================================="
echo

python run_experiments.py \
    --config $CONFIG_FILE \
    --output_dir $EXPERIMENT_DIR \
    --run_ablations \
    2>&1 | tee $LOG_DIR/ablations.log

# Step 4: Run hyperparameter search (optional)
echo
echo "=========================================="
echo "Step 4: Hyperparameter search (optional)..."
echo "=========================================="
echo

read -p "Run hyperparameter search? This will take considerable time. (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python run_experiments.py \
        --config $CONFIG_FILE \
        --output_dir $EXPERIMENT_DIR \
        --run_hyperparam \
        2>&1 | tee $LOG_DIR/hyperparam.log
fi

# Step 5: Run final model with multiple seeds
echo
echo "=========================================="
echo "Step 5: Running final model evaluation..."
echo "=========================================="
echo

python run_experiments.py \
    --config $CONFIG_FILE \
    --output_dir $EXPERIMENT_DIR \
    --run_final \
    --num_seeds 5 \
    2>&1 | tee $LOG_DIR/final_model.log

# Step 6: Generate comprehensive report
echo
echo "=========================================="
echo "Step 6: Generating final report..."
echo "=========================================="
echo

# Find the latest experiment directory
LATEST_EXP=$(ls -td $EXPERIMENT_DIR/experiments_* | head -1)

if [ -d "$LATEST_EXP" ]; then
    echo "Experiment results saved in: $LATEST_EXP"
    echo
    echo "Key files:"
    echo "  - All results: $LATEST_EXP/all_results.csv"
    echo "  - Summary report: $LATEST_EXP/experiment_report.txt"
    echo "  - Final model results: $LATEST_EXP/final_model_results.json"
    
    # Display summary
    if [ -f "$LATEST_EXP/experiment_report.txt" ]; then
        echo
        echo "=== SUMMARY ==="
        tail -n 20 "$LATEST_EXP/experiment_report.txt"
    fi
else
    echo "Warning: Could not find experiment results directory"
fi

echo
echo "=========================================="
echo "Pipeline completed!"
echo "=========================================="