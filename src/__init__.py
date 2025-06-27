"""
BioKG-BioBERT: Biological Knowledge Graph-Enhanced Transformer Architecture

A novel approach for cancer hallmarks classification that integrates structured
biological knowledge directly into transformer attention mechanisms.
"""

from .models import BioKGBioBERT
from .data import HoCDataModule
from .train import Trainer, train_model
from .evaluation import Evaluator

__version__ = "1.0.0"

__all__ = [
    'BioKGBioBERT',
    'HoCDataModule',
    'Trainer',
    'train_model',
    'Evaluator'
]