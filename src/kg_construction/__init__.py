"""
Biological Knowledge Graph Construction Module

This module provides functionality for:
- Extracting biological entities from text
- Building knowledge graphs from biological databases
- Integrating structured knowledge into downstream models
"""

from .bio_entity_extractor import BioEntityExtractor, BioEntity
from .kg_builder import BiologicalKGBuilder, KGNode, KGEdge
from .simple_kg_builder import SimpleKGBuilder
from .pipeline import BiologicalKGPipeline, KGPipelineOutput

__all__ = [
    'BioEntityExtractor',
    'BioEntity',
    'BiologicalKGBuilder',
    'SimpleKGBuilder',
    'KGNode',
    'KGEdge',
    'BiologicalKGPipeline',
    'KGPipelineOutput'
]