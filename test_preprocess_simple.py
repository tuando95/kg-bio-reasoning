#!/usr/bin/env python3
"""
Simple test of KG preprocessing
"""

import yaml
import asyncio
from datasets import load_dataset
from src.kg_construction import BiologicalKGPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_preprocessing():
    # Load config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize pipeline
    pipeline = BiologicalKGPipeline(config['knowledge_graph'])
    
    # Load dataset
    dataset = load_dataset("qanastek/HoC", split="train", trust_remote_code=True)
    
    # Process a few samples
    successful = 0
    total_nodes = 0
    total_edges = 0
    
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        text = sample['text']
        label = sample['label']
        
        print(f"\nSample {i}:")
        print(f"Text: {text[:100]}...")
        print(f"Label: {label}")
        
        try:
            # Extract entities
            entities = pipeline.entity_extractor.extract_entities(text)
            print(f"Entities: {len(entities)}")
            
            if entities:
                # Build KG
                kg_output = await pipeline.process_text(text)
                nodes = kg_output.knowledge_graph.number_of_nodes()
                edges = kg_output.knowledge_graph.number_of_edges()
                
                print(f"KG: {nodes} nodes, {edges} edges")
                successful += 1
                total_nodes += nodes
                total_edges += edges
                
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\nSummary:")
    print(f"Successful: {successful}/5")
    print(f"Avg nodes: {total_nodes/successful if successful > 0 else 0:.1f}")
    print(f"Avg edges: {total_edges/successful if successful > 0 else 0:.1f}")


if __name__ == "__main__":
    asyncio.run(test_preprocessing())