#!/usr/bin/env python3
"""
Test KG construction with mock API responses to isolate the issue
"""

import yaml
import logging
import asyncio
from src.kg_construction import BiologicalKGPipeline
from src.kg_construction.kg_builder import KGEdge
import aiohttp
from unittest.mock import patch, MagicMock

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def mock_string_response():
    """Mock STRING API response"""
    return [
        {
            "stringId_A": "9606.ENSP00000269305",
            "stringId_B": "9606.ENSP00000353178", 
            "preferredName_A": "TP53",
            "preferredName_B": "EGFR",
            "ncbiTaxonId": 9606,
            "score": 900,
            "nscore": 0.0,
            "fscore": 0.9,
            "pscore": 0.0,
            "ascore": 0.0,
            "escore": 0.0,
            "dscore": 0.0,
            "tscore": 0.0
        },
        {
            "stringId_A": "9606.ENSP00000269305",
            "stringId_B": "9606.ENSP00000078429",
            "preferredName_A": "TP53", 
            "preferredName_B": "VEGFA",
            "ncbiTaxonId": 9606,
            "score": 800
        }
    ]


async def test_with_mock_apis():
    """Test KG construction with mocked API responses"""
    
    # Load config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize pipeline
    pipeline = BiologicalKGPipeline(config['knowledge_graph'])
    
    # Test text
    text = "TP53 and EGFR are important cancer genes. VEGF promotes angiogenesis."
    hallmarks = ['resisting_cell_death', 'inducing_angiogenesis']
    
    print(f"\nInput text: {text}")
    print(f"Hallmarks: {hallmarks}")
    print("="*80)
    
    # Extract entities first
    entities = pipeline.entity_extractor.extract_entities(text)
    print(f"\nExtracted {len(entities)} entities:")
    for e in entities:
        print(f"  - {e.text} ({e.entity_type})")
    
    # Mock the STRING API call
    original_fetch = pipeline.kg_builder._fetch_string_interactions
    
    async def mock_fetch_string(nodes):
        logger.info(f"Mock STRING API called with {len(nodes)} nodes")
        
        # Create edges based on mock response
        edges = []
        node_map = {node.name.upper(): node for node in nodes}
        
        for interaction in mock_string_response():
            name_a = interaction['preferredName_A']
            name_b = interaction['preferredName_B']
            
            if name_a in node_map and name_b in node_map:
                edge = KGEdge(
                    source=node_map[name_a].node_id,
                    target=node_map[name_b].node_id,
                    edge_type='interacts',
                    properties={
                        'score': interaction['score'],
                        'database': 'STRING'
                    },
                    confidence=interaction['score'] / 1000.0
                )
                edges.append(edge)
                logger.info(f"Created edge: {name_a} -> {name_b}")
        
        return edges
    
    # Patch the method
    pipeline.kg_builder._fetch_string_interactions = mock_fetch_string
    
    # Build KG
    print("\nBuilding knowledge graph with mocked APIs...")
    kg = await pipeline.kg_builder.build_knowledge_graph(entities, hallmarks)
    
    print(f"\nFinal KG: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
    
    # Show edges
    print("\nEdges in graph:")
    for u, v, data in kg.edges(data=True):
        print(f"  {u} -> {v}: type={data.get('edge_type')}, confidence={data.get('confidence')}")
    
    # Restore original method
    pipeline.kg_builder._fetch_string_interactions = original_fetch


async def test_direct_api():
    """Test STRING API directly"""
    print("\n\nTesting STRING API directly:")
    print("="*80)
    
    url = "https://string-db.org/api/json/network"
    params = {
        'identifiers': '9606.TP53%0d9606.EGFR%0d9606.VEGFA',
        'species': 9606,
        'required_score': 400,
        'network_type': 'functional',
        'caller_identity': 'biokg_biobert_test'
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"Got {len(data)} interactions")
                    for i, interaction in enumerate(data[:3]):
                        print(f"\nInteraction {i+1}:")
                        for key, val in interaction.items():
                            print(f"  {key}: {val}")
                else:
                    print(f"Error: {await response.text()}")
    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    # Test with mocked APIs
    asyncio.run(test_with_mock_apis())
    
    # Test real API
    asyncio.run(test_direct_api())