#!/usr/bin/env python3
"""
Detailed debugging of KG construction to find why edges are missing
"""

import yaml
import logging
import asyncio
from src.kg_construction import BiologicalKGPipeline

# Setup very detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Also log from specific modules
logging.getLogger('src.kg_construction.kg_builder').setLevel(logging.DEBUG)
logging.getLogger('src.kg_construction.bio_entity_extractor').setLevel(logging.DEBUG)


async def debug_kg_construction():
    """Debug KG construction in detail"""
    
    # Load config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize pipeline
    pipeline = BiologicalKGPipeline(config['knowledge_graph'])
    
    # Test text with obvious genes
    text = "TP53 and EGFR are important cancer genes. VEGF promotes angiogenesis."
    hallmarks = ['resisting_cell_death', 'inducing_angiogenesis']
    
    print(f"\nInput text: {text}")
    print(f"Hallmarks: {hallmarks}")
    print("="*80)
    
    # Step 1: Entity extraction
    print("\n1. ENTITY EXTRACTION:")
    entities = pipeline.entity_extractor.extract_entities(text)
    print(f"Found {len(entities)} entities:")
    for i, e in enumerate(entities):
        print(f"  Entity {i}: {e.text} ({e.entity_type}) - confidence: {e.confidence:.3f}")
        print(f"    IDs: {e.normalized_ids}")
    
    if len(entities) == 0:
        print("\nPROBLEM: No entities extracted! Check ScispaCy installation.")
        return
    
    # Step 2: Create entity nodes
    print("\n2. ENTITY NODES:")
    entity_nodes = pipeline.kg_builder._create_entity_nodes(entities)
    for node in entity_nodes:
        print(f"  Node: {node.node_id} - {node.name} ({node.node_type})")
    
    # Step 3: Try to fetch relationships
    print("\n3. FETCHING RELATIONSHIPS:")
    edges = await pipeline.kg_builder._fetch_relationships(entity_nodes)
    print(f"Got {len(edges)} edges from databases")
    for edge in edges:
        print(f"  Edge: {edge.source} -> {edge.target} ({edge.edge_type})")
    
    # Step 4: Build complete KG
    print("\n4. BUILDING COMPLETE KG:")
    kg = await pipeline.kg_builder.build_knowledge_graph(entities, hallmarks)
    
    print(f"\nFinal KG: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
    
    # List all nodes
    print("\nNodes in graph:")
    for node, data in kg.nodes(data=True):
        print(f"  {node}: {data}")
    
    # List all edges
    print("\nEdges in graph:")
    for u, v, data in kg.edges(data=True):
        print(f"  {u} -> {v}: {data}")
    
    # Step 5: Check specific API calls
    print("\n5. TESTING STRING API DIRECTLY:")
    if entity_nodes:
        string_edges = await pipeline.kg_builder._fetch_string_interactions(entity_nodes)
        print(f"STRING returned {len(string_edges)} edges")
    
    # Step 6: Check pathway fetching
    print("\n6. TESTING PATHWAY FETCHING:")
    pathway_nodes = await pipeline.kg_builder._fetch_pathway_nodes(entity_nodes)
    print(f"Got {len(pathway_nodes)} pathway nodes")
    for pnode in pathway_nodes:
        print(f"  Pathway: {pnode.node_id} - {pnode.name}")


def test_simple_graph():
    """Test creating a simple graph directly"""
    print("\n\nTESTING SIMPLE GRAPH CREATION:")
    print("="*80)
    
    import networkx as nx
    
    g = nx.MultiDiGraph()
    g.add_node("GENE:TP53", type="gene", name="TP53")
    g.add_node("GENE:EGFR", type="gene", name="EGFR")
    g.add_edge("GENE:TP53", "GENE:EGFR", type="interacts", confidence=0.8)
    
    print(f"Simple graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print("This should have 2 nodes and 1 edge. If not, NetworkX is broken.")


if __name__ == "__main__":
    # Run async function
    asyncio.run(debug_kg_construction())
    
    # Test simple graph
    test_simple_graph()