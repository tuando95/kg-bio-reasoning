# Data-Driven Mechanistic Interpretability

This document explains the data-driven approach for mechanistic interpretability in BioKG-BioBERT, which learns hallmark-pathway-gene associations from the training data instead of using hardcoded biological knowledge.

## Overview

The data-driven approach consists of two main components:

1. **Association Learning**: Analyze cached knowledge graphs and training labels to learn which pathways and genes are statistically associated with each cancer hallmark
2. **Learned Interpretability**: Use these learned associations to provide mechanistic explanations for model predictions

## Benefits

- **Adaptive**: Automatically adapts to the specific dataset and domain
- **Data-driven**: Based on actual patterns in the training data, not predefined knowledge
- **Statistically grounded**: Uses proper statistical measures (support, confidence, lift, chi-square)
- **Maintainable**: No hardcoded biological knowledge to update
- **Transparent**: Shows the statistical basis for each association

## Running the Pipeline

### Step 1: Learn Associations from Training Data

```bash
python learn_hallmark_associations.py \
    --config configs/default_config.yaml \
    --split train \
    --output_dir learned_associations
```

This will:
- Load cached training data with knowledge graphs
- Extract pathways, genes, and other entities from each sample
- Calculate association scores between entities and hallmarks
- Generate visualizations and reports
- Save associations to `learned_associations/hallmark_associations.json`

### Step 2: Run Mechanistic Interpretability with Learned Associations

```bash
python run_mechanistic_interpretability_learned.py \
    --checkpoint experiments/biokg_biobert/seed_42/best_model.pt \
    --config configs/default_config.yaml \
    --associations learned_associations/hallmark_associations.json \
    --num_samples 10 \
    --output_dir learned_mechanistic_analysis
```

This will:
- Load the learned associations
- Analyze test samples using these associations
- Generate interpretations based on data-driven evidence
- Create visualizations showing pathway activation and gene contributions

### Complete Pipeline

Run the entire pipeline with:

```bash
./run_learned_mechanistic_pipeline.sh
```

## Output Files

### Association Learning Outputs

- `learned_associations/hallmark_associations.json`: Main associations file containing:
  - Pathway associations with scores
  - Gene associations with importance scores
  - Other entity associations
  - Metadata about the learning process

- `learned_associations/hallmark_pathway_associations.png`: Heatmap showing pathway-hallmark associations
- `learned_associations/hallmark_gene_network.png`: Network visualization of gene-hallmark relationships
- `learned_associations/association_statistics.png`: Statistics about learned associations
- `learned_associations/learned_associations_report.txt`: Detailed text report

### Mechanistic Analysis Outputs

- `learned_mechanistic_analysis/sample_*/`: Individual sample analyses containing:
  - `learned_pathway_activation.png`: Visualization of pathway evidence
  - `learned_gene_contributions.png`: Gene importance visualization
  - `molecular_network.html`: Interactive network visualization
  - `learned_interpretation.txt`: Natural language interpretation
  - `learned_analysis.json`: Structured analysis results

- `learned_mechanistic_analysis/learned_mechanistic_summary.txt`: Summary report across all samples

## Association Scoring

The system uses multiple statistical measures to score associations:

1. **Support**: How frequently the entity appears with the hallmark
   - `support = count(entity & hallmark) / count(hallmark)`

2. **Confidence**: Probability of hallmark given the entity
   - `confidence = count(entity & hallmark) / count(entity)`

3. **Lift**: How much more likely the association is than random
   - `lift = confidence / P(hallmark)`

4. **Chi-square test**: Statistical significance of the association

5. **Combined Score**: Weighted combination of all measures
   - `score = 0.3*support + 0.3*confidence + 0.2*lift + 0.2*chi2`

## Configuration

The biological knowledge configuration in `configs/default_config.yaml`:

```yaml
biological_knowledge:
  learned_associations_path: 'learned_associations/hallmark_associations.json'
  use_learned_associations: true
  min_association_score: 0.15
  pathway_evidence_weight: 0.3
  top_k_pathways: 20
  top_k_genes: 30
```

## Example Learned Associations

The system might learn associations like:

**Evading growth suppressors (Hallmark 0)**:
- Top pathways:
  - Cell cycle (hsa04110): score=0.82, support=0.65
  - p53 signaling (hsa04115): score=0.75, support=0.58
  - DNA damage checkpoint (R-HSA-69278): score=0.68, support=0.45
- Top genes:
  - TP53: score=0.91, mentioned in 78% of samples
  - RB1: score=0.84, mentioned in 62% of samples
  - CDKN2A: score=0.76, mentioned in 54% of samples

These associations are learned entirely from the data, making them more reliable and dataset-specific than hardcoded knowledge.

## Comparison with Hardcoded Approach

| Aspect | Hardcoded | Data-Driven |
|--------|-----------|-------------|
| Source | Literature/Databases | Training Data |
| Adaptability | Fixed | Adapts to dataset |
| Maintenance | Manual updates | Automatic |
| Transparency | Black box | Shows statistics |
| Coverage | Predefined | Data-dependent |
| Reliability | Expert knowledge | Statistical evidence |

## Future Enhancements

1. **Online Learning**: Update associations as new data becomes available
2. **Cross-Dataset Transfer**: Learn associations that generalize across datasets
3. **Confidence Intervals**: Add statistical confidence intervals to associations
4. **Temporal Analysis**: Track how associations change over time
5. **Multi-Modal Integration**: Combine text, KG, and other modalities for learning

## Troubleshooting

If associations are not being learned properly:
1. Check that cached datasets exist in `cache/kg_preprocessed/`
2. Ensure sufficient samples per hallmark (minimum ~50 recommended)
3. Adjust `min_association_score` threshold if too few associations
4. Check that knowledge graphs contain pathway/gene information

## Citation

If you use this data-driven mechanistic interpretability approach, please acknowledge that it learns associations from data rather than using predefined biological knowledge.