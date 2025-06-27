"""
Test the improved entity extractor with HoC dataset samples
"""

import asyncio
import logging
from src.kg_construction.bio_entity_extractor_improved import BioEntityExtractor
from src.kg_construction.kg_builder import BiologicalKGBuilder
from datasets import load_dataset
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_improved_extractor():
    """Test improved entity extractor with HoC samples"""
    
    # Load config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create improved extractor
    extractor = BioEntityExtractor(config['knowledge_graph']['entity_extraction'])
    
    # Create KG builder with fixed APIs
    kg_builder = BiologicalKGBuilder(config['knowledge_graph'])
    
    # Test samples from HoC dataset
    print("Loading HoC dataset...")
    dataset = load_dataset("qanastek/HoC", split="train", trust_remote_code=True)
    
    # Test specific problematic samples
    test_samples = [
        # Sample with Cu (should be CHEMICAL)
        "Cu exposure caused oxidative stress in fish liver cells",
        # Sample from actual HoC dataset
        dataset[0]['text'][:200] if len(dataset) > 0 else "",
        dataset[1]['text'][:200] if len(dataset) > 1 else "",
        # Additional test cases
        "TP53 mutations and BRCA1 deficiency lead to genomic instability",
        "The interaction between copper and zinc affects cellular metabolism",
        "Fish exposed to heavy metals showed increased apoptosis"
    ]
    
    print("\n" + "="*80)
    print("TESTING IMPROVED ENTITY EXTRACTION")
    print("="*80)
    
    for i, text in enumerate(test_samples):
        if not text:
            continue
            
        print(f"\nSample {i+1}: {text[:100]}...")
        print("-"*60)
        
        # Extract entities
        entities = extractor.extract_entities(text)
        
        # Categorize by type
        entity_types = {}
        gene_protein_entities = []
        
        for entity in entities:
            ent_type = entity.entity_type
            if ent_type not in entity_types:
                entity_types[ent_type] = []
            entity_types[ent_type].append(entity)
            
            if ent_type in ['GENE', 'PROTEIN']:
                gene_protein_entities.append(entity)
        
        # Show results
        print(f"Total entities extracted: {len(entities)}")
        for ent_type, ents in entity_types.items():
            print(f"  {ent_type}: {len(ents)}")
            for ent in ents[:3]:  # Show first 3 of each type
                print(f"    - '{ent.text}' (confidence: {ent.confidence:.2f})")
        
        # Test KG construction if we have gene/protein entities
        if gene_protein_entities:
            print(f"\nBuilding KG with {len(gene_protein_entities)} gene/protein entities...")
            
            # Extract gene names
            gene_names = []
            for entity in gene_protein_entities:
                gene_names.append(entity.text)
                # Also add normalized forms if available
                for db, gene_id in entity.normalized_ids.items():
                    if db in ['HGNC', 'SYMBOL']:
                        gene_names.append(gene_id)
            
            # Build KG
            kg = await kg_builder.build_knowledge_graph(
                entities=entities,
                hallmarks=['evading_growth_suppressors']
            )
            
            print(f"KG built: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
            
            # Count edge types
            edge_types = {}
            for u, v, data in kg.edges(data=True):
                edge_type = data.get('edge_type', 'unknown')
                database = data.get('database', 'N/A')
                key = f"{edge_type} ({database})"
                edge_types[key] = edge_types.get(key, 0) + 1
            
            if edge_types:
                print("Edge types:")
                for edge_type, count in edge_types.items():
                    print(f"  - {edge_type}: {count}")
        else:
            print("\nNo gene/protein entities found - cannot build meaningful KG")
        
        print("-"*60)

    # Test specific entity type classification
    print("\n" + "="*80)
    print("TESTING SPECIFIC ENTITY CLASSIFICATION")
    print("="*80)
    
    test_entities = [
        "Cu", "copper", "Zn", "zinc", "Fe", "iron",  # Should be CHEMICAL
        "fish", "mouse", "human", "yeast",  # Should be ORGANISM
        "TP53", "BRCA1", "p53", "EGFR",  # Should be GENE
        "apoptosis", "proliferation", "angiogenesis",  # Should be PROCESS
        "breast cancer", "melanoma", "tumor",  # Should be DISEASE
    ]
    
    for text in test_entities:
        entities = extractor.extract_entities(f"The {text} is important")
        if entities:
            entity = entities[0]
            print(f"'{text}' -> Type: {entity.entity_type}")
        else:
            print(f"'{text}' -> Not extracted")

if __name__ == "__main__":
    asyncio.run(test_improved_extractor())