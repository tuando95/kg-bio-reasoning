"""
Test KG pipeline with actual HoC dataset samples
"""

import asyncio
import logging
from datasets import load_dataset
from src.kg_construction.pipeline import BiologicalKGPipeline
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_hoc_samples():
    """Test with real HoC dataset samples"""
    
    # Load config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create pipeline
    pipeline = BiologicalKGPipeline(config['knowledge_graph'])
    
    # Load HoC dataset
    print("Loading HoC dataset...")
    dataset = load_dataset("qanastek/HoC", split="train", trust_remote_code=True)
    
    # Test first 10 samples
    print(f"\nTesting first 10 samples from HoC dataset...")
    print("=" * 80)
    
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        text = sample['text']
        labels = sample['label']
        
        print(f"\nSample {i+1}:")
        print(f"Text: {text[:150]}...")  # First 150 chars
        print(f"Labels: {labels}")
        
        # Process text
        result = await pipeline.process_text(text)
        
        # Show extracted entities
        print(f"\nExtracted {len(result.entities)} entities:")
        gene_protein_entities = []
        for entity in result.entities[:10]:  # Show first 10
            print(f"  - '{entity.text}' | Type: {entity.entity_type}")
            if entity.entity_type in ['GENE', 'PROTEIN']:
                gene_protein_entities.append(entity)
        
        print(f"\nGene/Protein entities: {len(gene_protein_entities)}")
        for entity in gene_protein_entities:
            print(f"  - '{entity.text}' | IDs: {entity.normalized_ids}")
        
        # Show KG stats
        kg = result.knowledge_graph
        print(f"\nKnowledge Graph:")
        print(f"  Nodes: {kg.number_of_nodes()}")
        print(f"  Edges: {kg.number_of_edges()}")
        
        # Count edge types
        edge_types = {}
        for u, v, data in kg.edges(data=True):
            edge_type = f"{data.get('edge_type')} ({data.get('database', 'N/A')})"
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        if edge_types:
            print(f"  Edge types: {dict(edge_types)}")
        
        print("-" * 80)
        
        # Brief pause between samples
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(test_hoc_samples())