#!/usr/bin/env python3
"""
Debug script for Knowledge Graph Pipeline

Tests entity extraction and KG construction on sample texts.
"""

import yaml
import logging
from src.kg_construction import BiologicalKGPipeline, BioEntityExtractor

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_entity_extraction():
    """Test entity extraction on sample biomedical texts."""
    print("\n" + "="*60)
    print("Testing Entity Extraction")
    print("="*60 + "\n")
    
    # Load config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize entity extractor
    extractor = BioEntityExtractor(config['knowledge_graph']['entity_extraction'])
    
    # Test texts
    test_texts = [
        "p53 mutation leads to apoptosis resistance in cancer cells.",
        "EGFR overexpression promotes sustained proliferative signaling through the MAPK pathway.",
        "VEGF expression induces angiogenesis by activating endothelial cell migration.",
        "The tumor suppressor gene BRCA1 is frequently mutated in breast cancer.",
        "Inhibition of CDK4/6 blocks cell cycle progression in RB-positive tumors."
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nText {i+1}: {text}")
        print("-" * 40)
        
        # Extract entities
        entities = extractor.extract_entities(text)
        
        if entities:
            print(f"Found {len(entities)} entities:")
            for entity in entities:
                print(f"  - {entity.text} ({entity.entity_type})")
                print(f"    Position: {entity.start}-{entity.end}")
                print(f"    Confidence: {entity.confidence:.3f}")
                print(f"    IDs: {entity.normalized_ids}")
        else:
            print("  No entities found!")
            
            # Try to debug
            import spacy
            nlp = extractor.nlp
            doc = nlp(text)
            
            print(f"\n  Debug - SpaCy found {len(doc.ents)} entities:")
            for ent in doc.ents:
                print(f"    - {ent.text} ({ent.label_})")


def test_kg_construction():
    """Test full KG construction pipeline."""
    print("\n" + "="*60)
    print("Testing Knowledge Graph Construction")
    print("="*60 + "\n")
    
    # Load config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize pipeline
    pipeline = BiologicalKGPipeline(config['knowledge_graph'])
    
    # Test text
    text = "p53 mutation leads to apoptosis resistance in cancer cells. VEGF promotes angiogenesis."
    hallmarks = ['resisting_cell_death', 'inducing_angiogenesis']
    
    print(f"Text: {text}")
    print(f"Hallmarks: {hallmarks}")
    print("-" * 40)
    
    # Process
    output = pipeline.process_text(text, hallmarks)
    
    print(f"\nEntities: {len(output.entities)}")
    for entity in output.entities:
        print(f"  - {entity.text} ({entity.entity_type})")
    
    print(f"\nKnowledge Graph:")
    print(f"  Nodes: {output.knowledge_graph.number_of_nodes()}")
    print(f"  Edges: {output.knowledge_graph.number_of_edges()}")
    
    # List nodes
    if output.knowledge_graph.number_of_nodes() > 0:
        print("\n  Node details:")
        for node in list(output.knowledge_graph.nodes(data=True))[:10]:
            print(f"    - {node[0]}: {node[1].get('node_type', 'unknown')} - {node[1].get('name', 'N/A')}")
    
    # List edges
    if output.knowledge_graph.number_of_edges() > 0:
        print("\n  Edge details:")
        for edge in list(output.knowledge_graph.edges(data=True))[:10]:
            print(f"    - {edge[0]} -> {edge[1]} ({edge[2].get('edge_type', 'unknown')})")
    else:
        print("\n  No edges found - this is the problem!")
        
    # Get statistics
    stats = pipeline.get_statistics(output.knowledge_graph)
    print(f"\nStatistics: {stats}")


def check_spacy_model():
    """Check if ScispaCy model is properly installed."""
    print("\n" + "="*60)
    print("Checking ScispaCy Installation")
    print("="*60 + "\n")
    
    try:
        import spacy
        import scispacy
        
        print("ScispaCy package: OK")
        
        # Try to load the model
        try:
            nlp = spacy.load("en_core_sci_lg")
            print("en_core_sci_lg model: OK")
            
            # Test on simple text
            doc = nlp("p53 is a tumor suppressor gene.")
            print(f"\nTest parse found {len(doc.ents)} entities:")
            for ent in doc.ents:
                print(f"  - {ent.text} ({ent.label_})")
                
        except Exception as e:
            print(f"Error loading en_core_sci_lg: {e}")
            print("\nTry installing with:")
            print("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz")
            
    except ImportError as e:
        print(f"ScispaCy not installed: {e}")


def test_api_connections():
    """Test connections to biological databases."""
    print("\n" + "="*60)
    print("Testing Database Connections")
    print("="*60 + "\n")
    
    import requests
    
    # Test APIs
    apis = {
        "KEGG": "https://rest.kegg.jp/info/kegg",
        "STRING": "https://string-db.org/api/json/version",
        "Reactome": "https://reactome.org/ContentService/data/database/version"
    }
    
    for name, url in apis.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"{name}: OK (Status {response.status_code})")
            else:
                print(f"{name}: Error (Status {response.status_code})")
        except Exception as e:
            print(f"{name}: Connection failed - {e}")


if __name__ == "__main__":
    # Run all tests
    check_spacy_model()
    test_entity_extraction()
    test_kg_construction()
    test_api_connections()