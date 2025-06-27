"""
Debug KG pipeline to see what entities are being extracted
"""

import asyncio
import logging
from src.kg_construction.pipeline import KGPipeline
import yaml

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def debug_pipeline():
    """Debug the KG pipeline with sample texts"""
    
    # Load config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create pipeline
    pipeline = KGPipeline(config['knowledge_graph'])
    
    # Test texts with known genes
    test_texts = [
        "TP53 mutations lead to cancer progression through apoptosis resistance.",
        "BRCA1 and BRCA2 are involved in DNA repair mechanisms.",
        "The EGFR pathway is activated in many tumors.",
        "p53 regulates cell cycle and induces apoptosis.",
        "HER2 overexpression drives tumor growth.",
        "The interaction between KRAS and PIK3CA promotes cell proliferation."
    ]
    
    hallmarks = ['evading_growth_suppressors', 'resisting_cell_death']
    
    for i, text in enumerate(test_texts):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {text}")
        print('='*60)
        
        # Process text
        result = await pipeline.process_text(text, hallmarks)
        
        # Show extracted entities
        print(f"\nExtracted {len(result.entities)} entities:")
        for entity in result.entities:
            print(f"  - Text: '{entity.text}' | Type: {entity.entity_type} | Normalized IDs: {entity.normalized_ids}")
        
        # Show KG stats
        kg = result.knowledge_graph
        print(f"\nKnowledge Graph:")
        print(f"  Nodes: {kg.number_of_nodes()}")
        print(f"  Edges: {kg.number_of_edges()}")
        
        # Show node details
        if kg.number_of_nodes() > 0:
            print("\nNodes in graph:")
            for node_id, data in list(kg.nodes(data=True))[:10]:  # First 10 nodes
                node_type = data.get('node_type', 'unknown')
                name = data.get('name', 'N/A')
                print(f"    {node_id} | type: {node_type} | name: {name}")
        
        # Show edge details
        if kg.number_of_edges() > 0:
            print("\nEdges in graph:")
            for u, v, data in list(kg.edges(data=True))[:10]:  # First 10 edges
                edge_type = data.get('edge_type', 'unknown')
                database = data.get('database', 'N/A')
                print(f"    {u} -> {v} | type: {edge_type} | database: {database}")

if __name__ == "__main__":
    asyncio.run(debug_pipeline())