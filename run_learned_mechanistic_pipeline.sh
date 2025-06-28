#!/bin/bash

# Pipeline for data-driven mechanistic interpretability

echo "=== Data-Driven Mechanistic Interpretability Pipeline ==="
echo

# Step 1: Learn hallmark associations from training data
echo "Step 1: Learning hallmark associations from cached training data..."
python learn_hallmark_associations.py \
    --config configs/default_config.yaml \
    --split train \
    --output_dir learned_associations

echo
echo "Step 2: Waiting for association learning to complete..."
# Check if the associations file was created
while [ ! -f "learned_associations/hallmark_associations.json" ]; do
    sleep 2
done

echo
echo "Step 3: Running mechanistic interpretability with learned associations..."
# Assuming you have a trained model checkpoint
python run_mechanistic_interpretability_learned.py \
    --checkpoint experiments/biokg_biobert/seed_42/best_model.pt \
    --config configs/default_config.yaml \
    --associations learned_associations/hallmark_associations.json \
    --num_samples 10 \
    --output_dir learned_mechanistic_analysis

echo
echo "=== Pipeline Complete ==="
echo "Results:"
echo "  - Learned associations: learned_associations/"
echo "  - Mechanistic analysis: learned_mechanistic_analysis/"
echo
echo "Key outputs:"
echo "  - learned_associations/hallmark_associations.json: Data-driven associations"
echo "  - learned_associations/hallmark_pathway_associations.png: Pathway heatmap"
echo "  - learned_associations/hallmark_gene_network.png: Gene network"
echo "  - learned_mechanistic_analysis/sample_*/: Individual sample analyses"
echo "  - learned_mechanistic_analysis/learned_mechanistic_summary.txt: Summary report"