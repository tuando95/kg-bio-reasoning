"""
Final test of all API integrations with fixes applied
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the fixed KG builder
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


async def test_complete_integration():
    """Test complete integration with all APIs"""
    print("\n" + "=" * 80)
    print("FINAL API INTEGRATION TEST - ALL DATABASES")
    print("=" * 80)
    
    # Create comprehensive test entities - major cancer genes
    entities = [
        MockBioEntity("TP53", 0, 4, "GENE", {"NCBI_GENE": "7157"}, 1.0),
        MockBioEntity("BRCA1", 10, 15, "GENE", {"NCBI_GENE": "672"}, 1.0),
        MockBioEntity("EGFR", 20, 24, "GENE", {"NCBI_GENE": "1956"}, 1.0),
        MockBioEntity("KRAS", 30, 34, "GENE", {"NCBI_GENE": "3845"}, 1.0),
        MockBioEntity("PIK3CA", 40, 46, "GENE", {"NCBI_GENE": "5290"}, 1.0),
        MockBioEntity("PTEN", 50, 54, "GENE", {"NCBI_GENE": "5728"}, 1.0),
        MockBioEntity("MYC", 60, 63, "GENE", {"NCBI_GENE": "4609"}, 1.0),
        MockBioEntity("BRCA2", 70, 75, "GENE", {"NCBI_GENE": "675"}, 1.0)
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
            'max_neighbors': 200,
            'include_pathways': True
        }
    }
    
    builder = BiologicalKGBuilder(config)
    
    # Build knowledge graph with cancer hallmarks
    hallmarks = ['evading_growth_suppressors', 'sustaining_proliferative_signaling', 'resisting_cell_death']
    
    print("\nBuilding knowledge graph...")
    print(f"Input: {len(entities)} genes")
    print(f"Hallmarks: {len(hallmarks)} cancer hallmarks")
    
    kg = await builder.build_knowledge_graph(entities, hallmarks)
    
    # Analyze results
    print("\n" + "-" * 60)
    print("RESULTS:")
    print("-" * 60)
    print(f"Total nodes: {kg.number_of_nodes()}")
    print(f"Total edges: {kg.number_of_edges()}")
    
    # Count node types
    node_types = {}
    for node_id, data in kg.nodes(data=True):
        node_type = data.get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\nNode type distribution:")
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type}: {count}")
    
    # Count edge types by database
    edge_stats = {}
    string_edges = 0
    kegg_edges = 0
    reactome_edges = 0
    
    for u, v, data in kg.edges(data=True):
        edge_type = data.get('edge_type', 'unknown')
        database = data.get('database', 'unknown')
        
        key = f"{edge_type} ({database})"
        edge_stats[key] = edge_stats.get(key, 0) + 1
        
        if database == 'STRING':
            string_edges += 1
        elif database == 'KEGG':
            kegg_edges += 1
        elif database == 'Reactome':
            reactome_edges += 1
    
    print("\nEdge type distribution:")
    for edge_type, count in sorted(edge_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {edge_type}: {count}")
    
    # API-specific results
    print("\n" + "-" * 60)
    print("API-SPECIFIC RESULTS:")
    print("-" * 60)
    
    # STRING results
    print("\nSTRING Database:")
    print(f"  Protein-protein interactions: {string_edges}")
    if string_edges > 0:
        # Show sample interactions
        print("  Sample interactions:")
        count = 0
        for u, v, data in kg.edges(data=True):
            if data.get('database') == 'STRING' and count < 5:
                score = data.get('score', 'N/A')
                print(f"    {u} <-> {v} (score: {score})")
                count += 1
    
    # KEGG results
    print("\nKEGG Database:")
    print(f"  Pathway memberships: {kegg_edges}")
    if kegg_edges > 0:
        # Count unique pathways
        kegg_pathways = set()
        for u, v, data in kg.edges(data=True):
            if data.get('database') == 'KEGG':
                kegg_pathways.add(v)
        print(f"  Unique pathways: {len(kegg_pathways)}")
        
        # Show sample pathways
        print("  Sample pathways:")
        for pathway in list(kegg_pathways)[:5]:
            pathway_name = kg.nodes[pathway].get('name', 'Unknown')
            print(f"    {pathway}: {pathway_name}")
    
    # Reactome results
    print("\nReactome Database:")
    print(f"  Pathway memberships: {reactome_edges}")
    if reactome_edges > 0:
        # Count unique pathways
        reactome_pathways = set()
        for u, v, data in kg.edges(data=True):
            if data.get('database') == 'Reactome':
                reactome_pathways.add(v)
        print(f"  Unique pathways: {len(reactome_pathways)}")
        
        # Show sample pathways
        print("  Sample pathways:")
        for pathway in list(reactome_pathways)[:5]:
            pathway_name = kg.nodes[pathway].get('name', 'Unknown')
            print(f"    {pathway}: {pathway_name}")
    
    # Test success criteria
    print("\n" + "=" * 80)
    print("TEST SUMMARY:")
    print("=" * 80)
    
    tests = {
        "Graph has nodes": kg.number_of_nodes() > len(entities),
        "Graph has edges": kg.number_of_edges() > 10,
        "STRING interactions found": string_edges > 0,
        "KEGG pathways found": kegg_edges > 0,
        "Reactome pathways found": reactome_edges > 0,
        "Multiple edge types": len(edge_stats) > 1,
        "Pathway nodes created": node_types.get('pathway', 0) > 0,
        "Hallmark nodes created": node_types.get('hallmark', 0) == len(hallmarks)
    }
    
    all_passed = True
    for test_name, passed in tests.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! The biological knowledge graph construction is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the logs for details.")
    
    # Export sample graph structure
    print("\n" + "-" * 60)
    print("SAMPLE GRAPH STRUCTURE (first 10 edges):")
    print("-" * 60)
    for i, (u, v, data) in enumerate(kg.edges(data=True)):
        if i >= 10:
            break
        edge_type = data.get('edge_type', 'unknown')
        database = data.get('database', 'unknown')
        confidence = data.get('confidence', 'N/A')
        print(f"{i+1}. {u} -> {v}")
        print(f"   Type: {edge_type}, Database: {database}, Confidence: {confidence}")


async def test_reactome_specific():
    """Test Reactome API specifically"""
    print("\n" + "=" * 80)
    print("REACTOME-SPECIFIC TEST")
    print("=" * 80)
    
    # Test with well-known cancer genes
    entities = [
        MockBioEntity("TP53", 0, 4, "GENE", {}, 1.0),
        MockBioEntity("BRCA1", 10, 15, "GENE", {}, 1.0),
        MockBioEntity("EGFR", 20, 24, "GENE", {}, 1.0)
    ]
    
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
            'max_neighbors': 100,
            'include_pathways': True
        }
    }
    
    builder = BiologicalKGBuilder(config)
    kg = await builder.build_knowledge_graph(entities)
    
    print(f"\nNodes: {kg.number_of_nodes()}")
    print(f"Edges: {kg.number_of_edges()}")
    
    # Check for Reactome pathways
    reactome_pathways = [(u, v, data) for u, v, data in kg.edges(data=True) 
                         if data.get('database') == 'Reactome']
    
    print(f"\nReactome pathway memberships: {len(reactome_pathways)}")
    
    if reactome_pathways:
        print("\nSample Reactome pathways:")
        for u, v, data in reactome_pathways[:10]:
            pathway_name = data.get('pathway_name', 'Unknown')
            print(f"  {u} -> {pathway_name}")
        return True
    else:
        print("No Reactome pathways found!")
        return False


async def main():
    """Run all tests"""
    print("BIOLOGICAL KNOWLEDGE GRAPH API INTEGRATION - FINAL TEST")
    print("=" * 80)
    
    # Test Reactome specifically first
    reactome_success = await test_reactome_specific()
    
    # Brief pause
    await asyncio.sleep(2)
    
    # Run complete integration test
    await test_complete_integration()


if __name__ == "__main__":
    asyncio.run(main())