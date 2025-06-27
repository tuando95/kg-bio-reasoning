#!/usr/bin/env python3
"""
Test ScispaCy entity extraction to diagnose the issue
"""

import spacy
import scispacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector

def test_scispacy():
    print("Testing ScispaCy entity extraction...")
    
    # Load the model
    try:
        nlp = spacy.load("en_core_sci_lg")
        print("✓ Model loaded successfully")
    except:
        print("✗ Failed to load en_core_sci_lg")
        print("Install with: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz")
        return
    
    # Add pipes
    try:
        nlp.add_pipe("abbreviation_detector")
        print("✓ Abbreviation detector added")
    except:
        print("✗ Failed to add abbreviation detector")
    
    try:
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        print("✓ UMLS entity linker added")
    except Exception as e:
        print(f"✗ Failed to add UMLS linker: {e}")
        print("This might be the issue - UMLS linker requires additional setup")
    
    # Test texts
    test_texts = [
        "p53 mutation leads to apoptosis resistance in cancer cells.",
        "EGFR overexpression promotes sustained proliferative signaling through the MAPK pathway.",
        "VEGF expression induces angiogenesis by activating endothelial cell migration.",
        "The tumor suppressor gene BRCA1 is frequently mutated in breast cancer.",
        "Inhibition of CDK4/6 blocks cell cycle progression in RB-positive tumors."
    ]
    
    print("\nTesting entity extraction on sample texts:")
    print("-" * 60)
    
    for text in test_texts:
        print(f"\nText: {text}")
        doc = nlp(text)
        
        print(f"Entities found: {len(doc.ents)}")
        for ent in doc.ents:
            print(f"  - {ent.text} ({ent.label_})")
            if hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                print(f"    UMLS: {ent._.kb_ents[0][0]} (confidence: {ent._.kb_ents[0][1]:.3f})")


def test_simple_extraction():
    """Test without UMLS linker"""
    print("\n\nTesting without UMLS linker:")
    print("-" * 60)
    
    nlp = spacy.load("en_core_sci_lg")
    
    test_text = "p53 and BRCA1 are tumor suppressor genes. EGFR and VEGF promote cancer growth."
    doc = nlp(test_text)
    
    print(f"Text: {test_text}")
    print(f"Entities: {len(doc.ents)}")
    for ent in doc.ents:
        print(f"  - {ent.text} ({ent.label_}) [{ent.start_char}:{ent.end_char}]")


if __name__ == "__main__":
    test_scispacy()
    test_simple_extraction()