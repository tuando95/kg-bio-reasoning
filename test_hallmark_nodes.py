"""
Test specifically for hallmark nodes
"""

import asyncio
import sys
sys.path.append('src')
from kg_construction.kg_builder_fixed import BiologicalKGBuilder
from dataclasses import dataclass
from typing import Dict

@dataclass
class MockBioEntity:
    text: str
    start: int
    end: int
    entity_type: str
    normalized_ids: Dict[str, str]
    confidence: float
    context: str = None


async def test_hallmarks():
    """Test hallmark node creation"""
    print("Testing Hallmark Node Creation")
    print("=" * 60)
    
    # Create simple test entities
    entities = [
        MockBioEntity("TP53", 0, 4, "GENE", {}, 1.0),
        MockBioEntity("KRAS", 10, 14, "GENE", {}, 1.0)
    ]
    
    # Configure with minimal settings
    config = {
        'databases': [],  # No databases, just hallmarks
        'graph_construction': {
            'max_hops': 0,
            'max_neighbors': 200,
            'include_pathways': True
        }
    }
    
    builder = BiologicalKGBuilder(config)
    
    # Build with hallmarks
    hallmarks = ['evading_growth_suppressors', 'sustaining_proliferative_signaling', 'resisting_cell_death']
    kg = await builder.build_knowledge_graph(entities, hallmarks)
    
    print(f"\nTotal nodes: {kg.number_of_nodes()}")
    print(f"Total edges: {kg.number_of_edges()}")
    
    # Count node types
    node_types = {}
    hallmark_nodes = []
    
    print("\nAll nodes in graph:")
    for node_id, data in kg.nodes(data=True):
        node_type = data.get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print(f"  {node_id}: type={node_type}, name={data.get('name', 'N/A')}")
        
        if node_type == 'hallmark':
            hallmark_nodes.append((node_id, data))
    
    print("\nNode type distribution:")
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type}: {count}")
    
    print(f"\nHallmark nodes found: {len(hallmark_nodes)}")
    for node_id, data in hallmark_nodes:
        print(f"  {node_id}: {data.get('name')}")
    
    # Check if hallmark IDs match expected
    expected_hallmark_ids = [f"HALLMARK:{h}" for h in hallmarks]
    print(f"\nExpected hallmark IDs: {expected_hallmark_ids}")
    
    actual_hallmark_ids = [node_id for node_id, _ in hallmark_nodes]
    print(f"Actual hallmark IDs: {actual_hallmark_ids}")
    
    # Check node attributes
    if hallmark_nodes:
        print("\nFirst hallmark node full data:")
        node_id, data = hallmark_nodes[0]
        for key, value in data.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_hallmarks())