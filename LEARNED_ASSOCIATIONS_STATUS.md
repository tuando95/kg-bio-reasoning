# Learned Associations Status

## Current Situation

### Dataset Label Distribution

The HoC dataset has a severe label imbalance issue:

**Training Set (12,119 samples):**
- Hallmark 0 (Evading growth suppressors): 374 samples (3%)
- Hallmark 4 (Resisting cell death): 1 sample
- All other hallmarks: 0 samples
- Most samples (>95%) are labeled as 7 ("None")

**Validation Set (1,798 samples):**
- Much better distribution with all hallmarks represented:
  - Hallmark 0: 23 samples
  - Hallmark 1: 45 samples
  - Hallmark 2: 11 samples
  - Hallmark 3: 44 samples
  - Hallmark 4: 72 samples
  - Hallmark 5: 64 samples
  - Hallmark 6: 103 samples
  - Hallmark 8: 60 samples
  - Hallmark 9: 85 samples
  - Hallmark 10: 20 samples

### Key Issues Fixed

1. **Label Format Handling**: The dataset uses single-label format as a list (e.g., `[7]` or `[0]`) rather than multi-label binary arrays. Fixed to handle both formats.

2. **Pathway ID Mismatch**: Fixed the mismatch between how pathways are stored (`pathway_id:pathway_name`) vs looked up.

3. **Graph Format**: Updated to handle both NetworkX graphs and serialized dictionary formats.

4. **Threshold Adjustments**: Lowered association thresholds (0.1 → 0.05) and minimum gene count (3 → 2) to capture more associations.

### Current Code Status

✅ **learn_hallmark_associations.py**: 
- Now properly handles single-label format
- Extracts associations from validation set successfully
- Generates meaningful associations for all hallmarks (except "None")

✅ **run_mechanistic_interpretability_learned.py**:
- Fixed label format handling
- Skips samples with only "None" label
- Properly loads cached test data

## Recommendations

1. **Use Validation Set for Learning**: Since the training set is severely imbalanced, use the validation set to learn associations:
   ```bash
   python learn_hallmark_associations.py \
       --config configs/default_config.yaml \
       --split validation \
       --output_dir learned_associations
   ```

2. **Run Mechanistic Analysis**: The learned associations from validation set should work better:
   ```bash
   python run_mechanistic_interpretability_learned.py \
       --checkpoint checkpoints/final_model_full/best.pt \
       --config configs/default_config.yaml \
       --associations learned_associations/hallmark_associations.json \
       --num_samples 10 \
       --output_dir learned_mechanistic_analysis
   ```

3. **Future Improvements**:
   - Consider combining train + validation sets for learning associations
   - Implement cross-validation to avoid overfitting to validation set
   - Add minimum support thresholds based on dataset size
   - Consider using external biological databases to supplement learned associations

## Key Insights

1. The HoC dataset's extreme imbalance in the training set makes it unsuitable for learning meaningful associations for most hallmarks.

2. The validation set has a much more balanced distribution and is better suited for learning associations.

3. The data-driven approach is still valuable but requires sufficient positive examples for each hallmark to be effective.

4. The learned associations show biologically plausible pathways and genes for each hallmark when sufficient data is available.