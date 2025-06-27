"""
Test script for the Biological Knowledge Graph Pipeline

This script tests the entity extraction and knowledge graph construction
components to ensure they work correctly.
"""

import yaml
import logging
from src.kg_construction import BiologicalKGPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_pipeline():
    """Test the biological knowledge graph pipeline"""
    
    # Load configuration
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize pipeline
    logger.info("Initializing Biological KG Pipeline...")
    pipeline = BiologicalKGPipeline(config['knowledge_graph'])
    
    # Test sentences with cancer hallmark context
    test_sentences = [
        "p53 mutation leads to apoptosis resistance in cancer cells.",
        "EGFR overexpression promotes sustained proliferative signaling through the MAPK pathway.",
        "VEGF expression induces angiogenesis by activating endothelial cell migration.",
        "Loss of PTEN results in PI3K-Akt pathway activation and cell survival.",
        "BRCA1 deficiency causes genomic instability and DNA repair defects."
    ]
    
    hallmarks = [
        ['resisting_cell_death'],
        ['sustaining_proliferative_signaling'],
        ['inducing_angiogenesis'],
        ['evading_growth_suppressors', 'resisting_cell_death'],
        ['genomic_instability']
    ]
    
    # Process each sentence
    for i, (sentence, hallmark_list) in enumerate(zip(test_sentences, hallmarks)):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing sentence {i+1}: {sentence}")
        logger.info(f"Associated hallmarks: {hallmark_list}")
        
        try:
            # Process through pipeline
            output = pipeline.process_text(sentence, hallmark_list)
            
            # Display results
            logger.info(f"\nExtracted Entities ({len(output.entities)}):")
            for entity in output.entities:
                logger.info(f"  - {entity.text} ({entity.entity_type})")
                logger.info(f"    IDs: {entity.normalized_ids}")
                logger.info(f"    Confidence: {entity.confidence:.3f}")
            
            # Knowledge graph statistics
            kg_stats = pipeline.get_statistics(output.knowledge_graph)
            logger.info(f"\nKnowledge Graph Statistics:")
            logger.info(f"  - Nodes: {kg_stats['num_nodes']}")
            logger.info(f"  - Edges: {kg_stats['num_edges']}")
            logger.info(f"  - Node types: {kg_stats['node_types']}")
            logger.info(f"  - Edge types: {kg_stats['edge_types']}")
            logger.info(f"  - Average degree: {kg_stats['avg_degree']:.2f}")
            
            # Test graph feature preparation
            graph_features = pipeline.prepare_graph_features(output.knowledge_graph)
            logger.info(f"\nGraph Features:")
            logger.info(f"  - Node features shape: {graph_features['node_features'].shape}")
            logger.info(f"  - Edge index shape: {graph_features['edge_index'].shape}")
            logger.info(f"  - Number of edge types: {len(graph_features['edge_type_to_idx'])}")
            
        except Exception as e:
            logger.error(f"Error processing sentence: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*60}")
    logger.info("Pipeline test completed!")


def test_entity_extraction_only():
    """Test just the entity extraction component"""
    from src.kg_construction import BioEntityExtractor
    
    # Load configuration
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize entity extractor
    logger.info("Testing entity extraction...")
    extractor = BioEntityExtractor(config['knowledge_graph']['entity_extraction'])
    
    # Test sentence
    test_text = "The tumor suppressor p53 regulates apoptosis through BCL2 and BAX proteins in response to DNA damage."
    
    # Extract entities
    entities = extractor.extract_entities(test_text)
    
    logger.info(f"Extracted {len(entities)} entities from: {test_text}")
    for entity in entities:
        logger.info(f"  - {entity.text} ({entity.entity_type}) at positions {entity.start}-{entity.end}")
        logger.info(f"    Normalized IDs: {entity.normalized_ids}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--entities-only":
        test_entity_extraction_only()
    else:
        test_pipeline()