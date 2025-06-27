"""
Test script to verify fixed API integrations for biological knowledge graph construction
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the fixed KG builder
import sys
sys.path.append('src')
from kg_construction.kg_builder_fixed import BiologicalKGBuilder, KGNode


# Mock BioEntity for testing
@dataclass
class MockBioEntity:
    text: str
    start: int
    end: int
    entity_type: str
    normalized_ids: Dict[str, str]
    confidence: float
    context: str = None


async def test_string_api():
    """Test STRING API with known working examples"""
    print("\n=== Testing STRING API ===")
    
    # Create test entities - well-known cancer genes
    entities = [
        MockBioEntity("TP53", 0, 4, "GENE", {}, 1.0),
        MockBioEntity("BRCA1", 10, 15, "GENE", {}, 1.0),
        MockBioEntity("BRCA2", 20, 25, "GENE", {}, 1.0),
        MockBioEntity("EGFR", 30, 34, "GENE", {}, 1.0),
        MockBioEntity("KRAS", 40, 44, "GENE", {}, 1.0)
    ]
    
    # Configure KG builder
    config = {
        'databases': [
            {
                'name': 'STRING',
                'enabled': True,
                'api_endpoint': 'https://string-db.org/api',
                'confidence_threshold': 400
            }
        ],
        'graph_construction': {
            'max_hops': 0,
            'max_neighbors': 50,
            'include_pathways': False
        }
    }
    
    builder = BiologicalKGBuilder(config)
    
    # Build knowledge graph
    kg = await builder.build_knowledge_graph(entities)
    
    print(f"Nodes: {kg.number_of_nodes()}")
    print(f"Edges: {kg.number_of_edges()}")
    
    # Display edges
    if kg.number_of_edges() > 0:
        print("\nSTRING interactions found:")
        for u, v, data in kg.edges(data=True):
            if data.get('edge_type') == 'interacts':
                print(f"  {u} <-> {v} (score: {data.get('score', 'N/A')})")
    else:
        print("No STRING interactions found!")
    
    return kg.number_of_edges() > 0


async def test_kegg_api():
    """Test KEGG API with known genes"""
    print("\n=== Testing KEGG API ===")
    
    # Create test entities
    entities = [
        MockBioEntity("TP53", 0, 4, "GENE", {"NCBI_GENE": "7157"}, 1.0),
        MockBioEntity("EGFR", 10, 14, "GENE", {"NCBI_GENE": "1956"}, 1.0),
        MockBioEntity("KRAS", 20, 24, "GENE", {"NCBI_GENE": "3845"}, 1.0)
    ]
    
    # Configure KG builder
    config = {
        'databases': [
            {
                'name': 'KEGG',
                'enabled': True,
                'api_endpoint': 'https://rest.kegg.jp'
            }
        ],
        'graph_construction': {
            'max_hops': 0,
            'max_neighbors': 50,
            'include_pathways': True
        }
    }
    
    builder = BiologicalKGBuilder(config)
    
    # Build knowledge graph
    kg = await builder.build_knowledge_graph(entities)
    
    print(f"Nodes: {kg.number_of_nodes()}")
    print(f"Edges: {kg.number_of_edges()}")
    
    # Display pathway memberships
    pathway_edges = [(u, v, data) for u, v, data in kg.edges(data=True) 
                     if data.get('edge_type') == 'pathway_member' and data.get('database') == 'KEGG']
    
    if pathway_edges:
        print("\nKEGG pathway memberships found:")
        for u, v, data in pathway_edges[:10]:  # Show first 10
            print(f"  {u} -> {v}")
    else:
        print("No KEGG pathway memberships found!")
    
    return len(pathway_edges) > 0


async def test_reactome_api():
    """Test Reactome API with known genes"""
    print("\n=== Testing Reactome API ===")
    
    # Create test entities
    entities = [
        MockBioEntity("TP53", 0, 4, "GENE", {}, 1.0),
        MockBioEntity("BRCA1", 10, 15, "GENE", {}, 1.0),
        MockBioEntity("EGFR", 20, 24, "GENE", {}, 1.0)
    ]
    
    # Configure KG builder
    config = {
        'databases': [
            {
                'name': 'Reactome',
                'enabled': True,
                'api_endpoint': 'https://reactome.org/ContentService'
            }
        ],
        'graph_construction': {
            'max_hops': 0,
            'max_neighbors': 50,
            'include_pathways': True
        }
    }
    
    builder = BiologicalKGBuilder(config)
    
    # Build knowledge graph
    kg = await builder.build_knowledge_graph(entities)
    
    print(f"Nodes: {kg.number_of_nodes()}")
    print(f"Edges: {kg.number_of_edges()}")
    
    # Display pathway memberships
    pathway_edges = [(u, v, data) for u, v, data in kg.edges(data=True) 
                     if data.get('edge_type') == 'pathway_member' and data.get('database') == 'Reactome']
    
    if pathway_edges:
        print("\nReactome pathway memberships found:")
        for u, v, data in pathway_edges[:10]:  # Show first 10
            pathway_name = data.get('pathway_name', 'Unknown')
            print(f"  {u} -> {pathway_name}")
    else:
        print("No Reactome pathway memberships found!")
    
    return len(pathway_edges) > 0


async def test_all_apis():
    """Test all APIs together"""
    print("\n=== Testing All APIs Together ===")
    
    # Create comprehensive test entities
    entities = [
        MockBioEntity("TP53", 0, 4, "GENE", {"NCBI_GENE": "7157"}, 1.0),
        MockBioEntity("BRCA1", 10, 15, "GENE", {"NCBI_GENE": "672"}, 1.0),
        MockBioEntity("EGFR", 20, 24, "GENE", {"NCBI_GENE": "1956"}, 1.0),
        MockBioEntity("KRAS", 30, 34, "GENE", {"NCBI_GENE": "3845"}, 1.0),
        MockBioEntity("PIK3CA", 40, 46, "GENE", {"NCBI_GENE": "5290"}, 1.0),
        MockBioEntity("PTEN", 50, 54, "GENE", {"NCBI_GENE": "5728"}, 1.0)
    ]
    
    # Configure KG builder with all databases
    config = {
        'databases': [
            {
                'name': 'STRING',
                'enabled': True,
                'api_endpoint': 'https://string-db.org/api',
                'confidence_threshold': 400
            },
            {
                'name': 'KEGG',
                'enabled': True,
                'api_endpoint': 'https://rest.kegg.jp'
            },
            {
                'name': 'Reactome',
                'enabled': True,
                'api_endpoint': 'https://reactome.org/ContentService'
            }
        ],
        'graph_construction': {
            'max_hops': 0,
            'max_neighbors': 100,
            'include_pathways': True
        }
    }
    
    builder = BiologicalKGBuilder(config)
    
    # Build knowledge graph with hallmarks
    hallmarks = ['evading_growth_suppressors', 'sustaining_proliferative_signaling']
    kg = await builder.build_knowledge_graph(entities, hallmarks)
    
    print(f"Total nodes: {kg.number_of_nodes()}")
    print(f"Total edges: {kg.number_of_edges()}")
    
    # Count edge types
    edge_types = {}
    for u, v, data in kg.edges(data=True):
        edge_type = data.get('edge_type', 'unknown')
        database = data.get('database', 'unknown')
        key = f"{edge_type} ({database})"
        edge_types[key] = edge_types.get(key, 0) + 1
    
    print("\nEdge type distribution:")
    for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {edge_type}: {count}")
    
    # Show sample edges
    print("\nSample edges:")
    for i, (u, v, data) in enumerate(kg.edges(data=True)):
        if i >= 5:
            break
        edge_type = data.get('edge_type', 'unknown')
        database = data.get('database', 'unknown')
        print(f"  {u} -> {v} [{edge_type}, {database}]")
    
    return kg.number_of_edges() > 10


async def main():
    """Run all tests"""
    print("Testing Fixed Biological Knowledge Graph API Integration")
    print("=" * 60)
    
    results = {
        "STRING": False,
        "KEGG": False,
        "Reactome": False,
        "Combined": False
    }
    
    try:
        # Test individual APIs
        results["STRING"] = await test_string_api()
        await asyncio.sleep(1)  # Brief pause between tests
        
        results["KEGG"] = await test_kegg_api()
        await asyncio.sleep(1)
        
        results["Reactome"] = await test_reactome_api()
        await asyncio.sleep(1)
        
        results["Combined"] = await test_all_apis()
        
    except Exception as e:
        logger.error(f"Test error: {e}", exc_info=True)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY:")
    print("=" * 60)
    
    for api, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{api:12} {status}")
    
    total_passed = sum(results.values())
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All API integrations are working correctly!")
    else:
        print("\n⚠️  Some API integrations need attention.")


if __name__ == "__main__":
    asyncio.run(main())