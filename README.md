# BioKG-BioBERT: Biological Knowledge Graph-Enhanced Transformer Architecture

This repository implements BioKG-BioBERT, a novel approach for cancer hallmarks classification that integrates structured biological knowledge directly into transformer attention mechanisms.

## Overview

BioKG-BioBERT enhances traditional transformer models by:
- Extracting biological entities and constructing sentence-specific knowledge graphs
- Implementing biological pathway-guided attention mechanisms
- Fusing textual and graph representations through multi-modal learning
- Enforcing biological consistency constraints during training

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/biokg-biobert.git
cd biokg-biobert

# Install dependencies
pip install -r requirements.txt

# Download ScispaCy model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz
```

## Quick Start

### Training

```bash
# Train the full BioKG-BioBERT model
python main.py --config configs/default_config.yaml --mode train

# Run specific experiments
python run_experiments.py --config configs/default_config.yaml --run_all
```

### Evaluation

```bash
# Evaluate a trained model
python main.py --config configs/default_config.yaml --mode evaluate --checkpoint checkpoints/best.pt
```

## Project Structure

```
biokg-biobert/
├── configs/
│   └── default_config.yaml      # Main configuration file
├── src/
│   ├── models/                  # Model architectures
│   │   ├── biobert_base.py     # BioBERT with entity awareness
│   │   ├── gnn_module.py       # Graph neural network components
│   │   ├── bio_attention.py    # Biological pathway-guided attention
│   │   └── biokg_biobert.py    # Complete model
│   ├── kg_construction/         # Knowledge graph construction
│   │   ├── bio_entity_extractor.py
│   │   ├── kg_builder.py
│   │   └── pipeline.py
│   ├── data/                    # Data loading and preprocessing
│   │   └── dataset.py
│   ├── train.py                 # Training logic
│   └── evaluation.py            # Evaluation metrics
├── run_experiments.py           # Experiment runner for ablations
├── main.py                      # Main entry point
└── requirements.txt
```

## Key Features

### 1. Biological Entity Extraction
- Uses ScispaCy for biomedical NER
- Normalizes entities to standard database identifiers (HGNC, UniProt, CHEBI, etc.)
- Maps entities across multiple biological databases

### 2. Dynamic Knowledge Graph Construction
- Queries KEGG, STRING, Reactome, and GO databases
- Builds sentence-specific subgraphs with biological relationships
- Incorporates hallmark-specific pathway information

### 3. Biological Pathway-Guided Attention
- Replaces standard self-attention with biologically-informed attention
- Integrates pathway relevance scores into attention computation
- Learnable fusion between textual and biological attention

### 4. Multi-Task Learning
- Primary task: Cancer hallmark classification
- Auxiliary task 1: Pathway activation prediction
- Auxiliary task 2: Biological consistency validation

## Configuration

The model behavior is controlled through `configs/default_config.yaml`:

```yaml
model:
  base_model: "dmis-lab/biobert-base-cased-v1.1"
  bio_attention:
    enabled: true
    fusion_strategy: "learned"
    
knowledge_graph:
  databases:
    - name: "KEGG"
      enabled: true
    - name: "STRING"
      enabled: true
      
training:
  batch_size: 16
  max_epochs: 50
  learning_rate: 2e-5
```

## Ablation Studies

Run comprehensive ablation studies to evaluate component contributions:

```bash
# Run all ablation studies
python run_experiments.py --config configs/default_config.yaml --run_ablations

# Run specific ablation types
python run_experiments.py --run_baselines  # Baseline models
python run_experiments.py --run_hyperparam  # Hyperparameter search
```

## Evaluation Metrics

The model is evaluated using:
- Multi-label classification metrics (F1-micro/macro, Hamming loss)
- Per-hallmark performance analysis
- Biological consistency metrics
- Pathway prediction accuracy

## Citation

If you use this code in your research, please cite:

```bibtex
@article{biokg-biobert,
  title={BioKG-BioBERT: Biological Knowledge Graph-Enhanced Transformer Architecture for Cancer Hallmarks Classification},
  author={Your Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.