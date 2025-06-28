"""
Debug script to understand why associations are not being found
"""

import pickle
import json
from pathlib import Path
import numpy as np
from collections import Counter

def debug_associations():
    """Debug why associations are not being found."""
    
    # Load a few samples to inspect
    cache_dir = Path("cache/kg_preprocessed/train")
    index_path = cache_dir / "index.json"
    
    if not index_path.exists():
        print(f"Index file not found at {index_path}")
        return
    
    with open(index_path, 'r') as f:
        cache_index = json.load(f)
    
    # Load first 10 samples
    hallmark_counts = Counter()
    pathway_counts = Counter()
    gene_counts = Counter()
    samples_with_kg = 0
    samples_with_pathways = 0
    samples_with_genes = 0
    
    for i, (sample_id, cache_info) in enumerate(list(cache_index.items())[:100]):
        cache_file = cache_dir / cache_info['file']
        
        with open(cache_file, 'rb') as f:
            sample = pickle.load(f)
        
        # Check labels
        labels = sample.get('labels', [])
        if isinstance(labels, list):
            labels = np.array(labels)
        
        active_hallmarks = np.where(labels == 1)[0].tolist()
        for h in active_hallmarks:
            hallmark_counts[h] += 1
        
        # Check KG
        kg = sample.get('knowledge_graph')
        if kg:
            samples_with_kg += 1
            
            # Count node types
            has_pathways = False
            has_genes = False
            
            if isinstance(kg, dict) and 'nodes' in kg:
                # Serialized format
                for node, attrs in kg['nodes']:
                    node_type = attrs.get('node_type', 'unknown')
                    if node_type == 'pathway':
                        has_pathways = True
                        pathway_counts[attrs.get('name', 'unknown')] += 1
                    elif node_type in ['gene', 'protein']:
                        has_genes = True
                        gene_counts[attrs.get('name', 'unknown')] += 1
            
            if has_pathways:
                samples_with_pathways += 1
            if has_genes:
                samples_with_genes += 1
        
        if i == 0:
            # Print first sample details
            print(f"\nFirst sample details:")
            print(f"Labels: {labels}")
            print(f"Active hallmarks: {active_hallmarks}")
            print(f"Has KG: {kg is not None}")
            if kg and isinstance(kg, dict):
                print(f"KG structure: {kg.keys()}")
                if 'nodes' in kg:
                    print(f"Number of nodes: {len(kg['nodes'])}")
                    # Print first few nodes
                    for j, (node, attrs) in enumerate(kg['nodes'][:5]):
                        print(f"  Node {j}: type={attrs.get('node_type')}, name={attrs.get('name')}")
    
    print(f"\nAnalyzed {i+1} samples")
    print(f"Samples with KG: {samples_with_kg}")
    print(f"Samples with pathways: {samples_with_pathways}")
    print(f"Samples with genes: {samples_with_genes}")
    
    print(f"\nHallmark distribution:")
    for h, count in sorted(hallmark_counts.items()):
        print(f"  Hallmark {h}: {count} samples")
    
    print(f"\nTop pathways:")
    for pathway, count in pathway_counts.most_common(10):
        print(f"  {pathway}: {count}")
    
    print(f"\nTop genes:")
    for gene, count in gene_counts.most_common(10):
        print(f"  {gene}: {count}")

if __name__ == "__main__":
    debug_associations()