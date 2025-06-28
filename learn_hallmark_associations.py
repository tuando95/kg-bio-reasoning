"""
Learn Hallmark-Pathway-Gene Associations from Cached KG Data

This script analyzes the cached knowledge graphs and training labels to:
1. Extract biological associations for each hallmark
2. Learn importance weights based on co-occurrence and predictive power
3. Save learned associations for use in mechanistic interpretability
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
import networkx as nx
from collections import defaultdict, Counter
import logging
from typing import Dict, List, Set, Tuple
import yaml
from tqdm import tqdm
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HallmarkAssociationLearner:
    """Learn hallmark-pathway-gene associations from cached KG data."""
    
    def __init__(self, config_path: str = 'configs/default_config.yaml'):
        """Initialize learner."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.hallmark_names = {
            0: "Evading growth suppressors",
            1: "Tumor promoting inflammation",
            2: "Enabling replicative immortality",
            3: "Cellular energetics",
            4: "Resisting cell death",
            5: "Activating invasion and metastasis",
            6: "Genomic instability and mutation",
            7: "None",
            8: "Inducing angiogenesis",
            9: "Sustaining proliferative signaling",
            10: "Avoiding immune destruction"
        }
        
        # Initialize association storage
        self.hallmark_pathway_counts = defaultdict(lambda: defaultdict(int))
        self.hallmark_gene_counts = defaultdict(lambda: defaultdict(int))
        self.hallmark_entity_counts = defaultdict(lambda: defaultdict(int))
        
        # Background frequencies
        self.total_pathway_counts = defaultdict(int)
        self.total_gene_counts = defaultdict(int)
        self.total_entity_counts = defaultdict(int)
        
        # Sample counts
        self.hallmark_sample_counts = defaultdict(int)
        self.total_samples = 0
    
    def load_cached_dataset(self, split: str = 'train') -> List[Dict]:
        """Load cached dataset with KGs."""
        cache_dir = Path(self.config['data']['cache_dir'])
        cache_file = cache_dir / f"cached_{split}_dataset.pkl"
        
        logger.info(f"Loading cached {split} dataset from {cache_file}")
        
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        return cached_data
    
    def extract_associations(self, cached_data: List[Dict]):
        """Extract associations from cached KG data."""
        logger.info(f"Extracting associations from {len(cached_data)} samples")
        
        for sample_idx, sample in enumerate(tqdm(cached_data, desc="Processing samples")):
            self.total_samples += 1
            
            # Get labels for this sample
            labels = sample.get('labels', np.array([]))
            active_hallmarks = np.where(labels == 1)[0].tolist()
            
            # Update hallmark sample counts
            for hallmark_id in active_hallmarks:
                self.hallmark_sample_counts[hallmark_id] += 1
            
            # Get knowledge graph
            kg = sample.get('knowledge_graph', None)
            if kg is None or len(kg.nodes()) == 0:
                continue
            
            # Extract biological entities from KG
            pathways, genes, entities = self._extract_kg_entities(kg)
            
            # Update background counts
            for pathway_id, pathway_name in pathways.items():
                self.total_pathway_counts[pathway_id] += 1
            for gene in genes:
                self.total_gene_counts[gene] += 1
            for entity_type, entity_set in entities.items():
                for entity in entity_set:
                    self.total_entity_counts[f"{entity_type}:{entity}"] += 1
            
            # Update hallmark-specific counts
            for hallmark_id in active_hallmarks:
                if hallmark_id == 7:  # Skip "None"
                    continue
                
                # Count pathways
                for pathway_id, pathway_name in pathways.items():
                    self.hallmark_pathway_counts[hallmark_id][f"{pathway_id}:{pathway_name}"] += 1
                
                # Count genes
                for gene in genes:
                    self.hallmark_gene_counts[hallmark_id][gene] += 1
                
                # Count other entities
                for entity_type, entity_set in entities.items():
                    for entity in entity_set:
                        self.hallmark_entity_counts[hallmark_id][f"{entity_type}:{entity}"] += 1
        
        logger.info(f"Processed {self.total_samples} samples")
        logger.info(f"Found {len(self.total_pathway_counts)} unique pathways")
        logger.info(f"Found {len(self.total_gene_counts)} unique genes")
    
    def _extract_kg_entities(self, kg: nx.MultiDiGraph) -> Tuple[Dict, Set, Dict]:
        """Extract pathways, genes, and other entities from KG."""
        pathways = {}
        genes = set()
        entities = defaultdict(set)
        
        for node in kg.nodes():
            node_data = kg.nodes[node]
            node_type = node_data.get('node_type', 'unknown')
            
            if node_type == 'pathway':
                pathway_id = node_data.get('properties', {}).get('pathway_id', '')
                pathway_name = node_data.get('name', '')
                if pathway_id:
                    pathways[pathway_id] = pathway_name
            
            elif node_type in ['gene', 'protein']:
                gene_name = node_data.get('name', '').upper()
                if gene_name and len(gene_name) > 1:  # Filter out single letters
                    genes.add(gene_name)
                    entities['gene_protein'].add(gene_name)
            
            elif node_type == 'disease':
                disease_name = node_data.get('name', '')
                if disease_name:
                    entities['disease'].add(disease_name)
            
            elif node_type == 'chemical':
                chemical_name = node_data.get('name', '')
                if chemical_name:
                    entities['chemical'].add(chemical_name)
            
            elif node_type == 'biological_process':
                process_name = node_data.get('name', '')
                if process_name:
                    entities['biological_process'].add(process_name)
        
        return pathways, genes, entities
    
    def calculate_association_scores(self) -> Dict:
        """Calculate association scores using statistical measures."""
        logger.info("Calculating association scores")
        
        associations = {
            'pathways': defaultdict(dict),
            'genes': defaultdict(dict),
            'entities': defaultdict(dict)
        }
        
        # Calculate pathway associations
        for hallmark_id in range(11):
            if hallmark_id == 7:  # Skip "None"
                continue
            
            hallmark_name = self.hallmark_names[hallmark_id]
            associations['pathways'][hallmark_id] = self._calculate_pathway_scores(hallmark_id)
            associations['genes'][hallmark_id] = self._calculate_gene_scores(hallmark_id)
            associations['entities'][hallmark_id] = self._calculate_entity_scores(hallmark_id)
            
            logger.info(f"{hallmark_name}: {len(associations['pathways'][hallmark_id])} pathways, "
                       f"{len(associations['genes'][hallmark_id])} genes")
        
        return associations
    
    def _calculate_pathway_scores(self, hallmark_id: int) -> Dict:
        """Calculate pathway association scores for a hallmark."""
        pathway_scores = {}
        hallmark_samples = self.hallmark_sample_counts[hallmark_id]
        
        if hallmark_samples == 0:
            return pathway_scores
        
        for pathway, count in self.hallmark_pathway_counts[hallmark_id].items():
            pathway_id = pathway.split(':')[0]
            pathway_name = ':'.join(pathway.split(':')[1:])
            
            # Calculate metrics
            # 1. Support: How often this pathway appears with this hallmark
            support = count / hallmark_samples
            
            # 2. Confidence: P(hallmark | pathway)
            total_pathway_occurrences = self.total_pathway_counts.get(pathway_id, 1)
            confidence = count / total_pathway_occurrences
            
            # 3. Lift: How much more likely is this association than random
            hallmark_prob = hallmark_samples / self.total_samples
            pathway_prob = total_pathway_occurrences / self.total_samples
            expected_count = hallmark_prob * pathway_prob * self.total_samples
            lift = count / max(expected_count, 1)
            
            # 4. Chi-square test for independence
            contingency_table = np.array([
                [count, hallmark_samples - count],
                [total_pathway_occurrences - count, 
                 self.total_samples - hallmark_samples - total_pathway_occurrences + count]
            ])
            
            try:
                chi2, p_value, _, _ = chi2_contingency(contingency_table)
                chi2_score = chi2 if p_value < 0.05 else 0
            except:
                chi2_score = 0
                p_value = 1.0
            
            # Combined score (weighted combination)
            combined_score = (
                0.3 * min(support, 1.0) +  # Cap at 1
                0.3 * min(confidence, 1.0) +  
                0.2 * min(lift / 10, 1.0) +  # Normalize lift
                0.2 * min(chi2_score / 100, 1.0)  # Normalize chi2
            )
            
            if combined_score > 0.1:  # Threshold for inclusion
                pathway_scores[pathway] = {
                    'id': pathway_id,
                    'name': pathway_name,
                    'count': count,
                    'support': float(support),
                    'confidence': float(confidence),
                    'lift': float(lift),
                    'chi2': float(chi2_score),
                    'p_value': float(p_value),
                    'score': float(combined_score)
                }
        
        # Sort by score and return top pathways
        sorted_pathways = sorted(pathway_scores.items(), 
                                key=lambda x: x[1]['score'], 
                                reverse=True)
        
        return {k: v for k, v in sorted_pathways[:20]}  # Top 20 pathways
    
    def _calculate_gene_scores(self, hallmark_id: int) -> Dict:
        """Calculate gene association scores for a hallmark."""
        gene_scores = {}
        hallmark_samples = self.hallmark_sample_counts[hallmark_id]
        
        if hallmark_samples == 0:
            return gene_scores
        
        for gene, count in self.hallmark_gene_counts[hallmark_id].items():
            # Similar calculations as pathways
            support = count / hallmark_samples
            total_gene_occurrences = self.total_gene_counts.get(gene, 1)
            confidence = count / total_gene_occurrences
            
            hallmark_prob = hallmark_samples / self.total_samples
            gene_prob = total_gene_occurrences / self.total_samples
            expected_count = hallmark_prob * gene_prob * self.total_samples
            lift = count / max(expected_count, 1)
            
            # Combined score
            combined_score = (
                0.4 * min(support, 1.0) +
                0.3 * min(confidence, 1.0) +
                0.3 * min(lift / 10, 1.0)
            )
            
            if combined_score > 0.1 and count >= 3:  # Minimum support
                gene_scores[gene] = {
                    'symbol': gene,
                    'count': count,
                    'support': float(support),
                    'confidence': float(confidence),
                    'lift': float(lift),
                    'score': float(combined_score)
                }
        
        # Sort and return top genes
        sorted_genes = sorted(gene_scores.items(), 
                             key=lambda x: x[1]['score'], 
                             reverse=True)
        
        return {k: v for k, v in sorted_genes[:30]}  # Top 30 genes
    
    def _calculate_entity_scores(self, hallmark_id: int) -> Dict:
        """Calculate other entity association scores."""
        entity_scores = defaultdict(list)
        hallmark_samples = self.hallmark_sample_counts[hallmark_id]
        
        if hallmark_samples == 0:
            return dict(entity_scores)
        
        for entity, count in self.hallmark_entity_counts[hallmark_id].items():
            entity_type, entity_name = entity.split(':', 1)
            
            support = count / hallmark_samples
            total_entity_occurrences = self.total_entity_counts.get(entity, 1)
            confidence = count / total_entity_occurrences
            
            if support > 0.05 and count >= 2:  # Minimum thresholds
                entity_scores[entity_type].append({
                    'name': entity_name,
                    'count': count,
                    'support': float(support),
                    'confidence': float(confidence)
                })
        
        # Sort each entity type by support
        for entity_type in entity_scores:
            entity_scores[entity_type] = sorted(
                entity_scores[entity_type], 
                key=lambda x: x['support'], 
                reverse=True
            )[:10]  # Top 10 per type
        
        return dict(entity_scores)
    
    def visualize_associations(self, associations: Dict, save_dir: Path):
        """Create visualizations of learned associations."""
        save_dir.mkdir(exist_ok=True)
        
        # 1. Pathway association heatmap
        self._plot_pathway_heatmap(associations['pathways'], save_dir)
        
        # 2. Gene association network
        self._plot_gene_network(associations['genes'], save_dir)
        
        # 3. Association statistics
        self._plot_association_stats(associations, save_dir)
    
    def _plot_pathway_heatmap(self, pathway_associations: Dict, save_dir: Path):
        """Plot heatmap of pathway associations."""
        # Prepare data for heatmap
        hallmarks = []
        pathways = set()
        
        for hallmark_id, pathways_dict in pathway_associations.items():
            if hallmark_id == 7:
                continue
            hallmarks.append(self.hallmark_names[hallmark_id])
            pathways.update([p['name'] for p in pathways_dict.values()])
        
        # Create matrix
        pathway_list = sorted(list(pathways))[:30]  # Top 30 pathways
        matrix = np.zeros((len(hallmarks), len(pathway_list)))
        
        for i, hallmark_id in enumerate([h for h in range(11) if h != 7]):
            for j, pathway_name in enumerate(pathway_list):
                # Find score for this pathway
                for pathway_data in pathway_associations[hallmark_id].values():
                    if pathway_data['name'] == pathway_name:
                        matrix[i, j] = pathway_data['score']
                        break
        
        # Plot
        plt.figure(figsize=(14, 10))
        sns.heatmap(matrix, 
                    xticklabels=pathway_list,
                    yticklabels=[h.split()[0] + "..." for h in hallmarks],
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Association Score'})
        
        plt.title('Learned Hallmark-Pathway Associations')
        plt.xlabel('Pathways')
        plt.ylabel('Cancer Hallmarks')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_dir / 'hallmark_pathway_associations.png', dpi=300)
        plt.close()
    
    def _plot_gene_network(self, gene_associations: Dict, save_dir: Path):
        """Plot network of gene associations."""
        # Create network graph
        G = nx.Graph()
        
        # Add hallmark nodes
        for hallmark_id in range(11):
            if hallmark_id == 7:
                continue
            G.add_node(f"H{hallmark_id}", 
                      node_type='hallmark',
                      label=self.hallmark_names[hallmark_id].split()[0])
        
        # Add gene nodes and edges
        all_genes = set()
        for hallmark_id, genes_dict in gene_associations.items():
            if hallmark_id == 7:
                continue
            
            for gene, gene_data in list(genes_dict.items())[:10]:  # Top 10 genes
                gene_symbol = gene_data['symbol']
                all_genes.add(gene_symbol)
                
                if not G.has_node(gene_symbol):
                    G.add_node(gene_symbol, node_type='gene')
                
                G.add_edge(f"H{hallmark_id}", gene_symbol, 
                          weight=gene_data['score'],
                          support=gene_data['support'])
        
        # Plot
        plt.figure(figsize=(16, 12))
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        hallmark_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'hallmark']
        gene_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'gene']
        
        nx.draw_networkx_nodes(G, pos, nodelist=hallmark_nodes, 
                              node_color='lightblue', node_size=2000, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes, 
                              node_color='lightgreen', node_size=1000, alpha=0.6)
        
        # Draw edges with width based on score
        edges = G.edges()
        weights = [G[u][v]['weight'] * 5 for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5)
        
        # Labels
        labels = {}
        for node in G.nodes():
            if node.startswith('H'):
                labels[node] = G.nodes[node]['label']
            else:
                labels[node] = node
        
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        plt.title('Learned Hallmark-Gene Association Network')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_dir / 'hallmark_gene_network.png', dpi=300)
        plt.close()
    
    def _plot_association_stats(self, associations: Dict, save_dir: Path):
        """Plot statistics about learned associations."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Number of pathways per hallmark
        ax = axes[0, 0]
        hallmark_names = []
        pathway_counts = []
        
        for h_id in range(11):
            if h_id == 7:
                continue
            hallmark_names.append(self.hallmark_names[h_id].split()[0] + "...")
            pathway_counts.append(len(associations['pathways'].get(h_id, {})))
        
        ax.bar(hallmark_names, pathway_counts)
        ax.set_xlabel('Hallmarks')
        ax.set_ylabel('Number of Associated Pathways')
        ax.set_title('Pathway Associations per Hallmark')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. Number of genes per hallmark
        ax = axes[0, 1]
        gene_counts = []
        
        for h_id in range(11):
            if h_id == 7:
                continue
            gene_counts.append(len(associations['genes'].get(h_id, {})))
        
        ax.bar(hallmark_names, gene_counts, color='green', alpha=0.7)
        ax.set_xlabel('Hallmarks')
        ax.set_ylabel('Number of Associated Genes')
        ax.set_title('Gene Associations per Hallmark')
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Sample distribution
        ax = axes[1, 0]
        sample_counts = [self.hallmark_sample_counts[h_id] for h_id in range(11) if h_id != 7]
        
        ax.bar(hallmark_names, sample_counts, color='orange', alpha=0.7)
        ax.set_xlabel('Hallmarks')
        ax.set_ylabel('Number of Training Samples')
        ax.set_title('Training Sample Distribution')
        ax.tick_params(axis='x', rotation=45)
        
        # 4. Top shared genes
        ax = axes[1, 1]
        gene_hallmark_counts = Counter()
        
        for h_id, genes_dict in associations['genes'].items():
            for gene in genes_dict:
                gene_hallmark_counts[gene] += 1
        
        shared_genes = [(gene, count) for gene, count in gene_hallmark_counts.items() if count > 1]
        shared_genes.sort(key=lambda x: x[1], reverse=True)
        
        if shared_genes:
            top_shared = shared_genes[:10]
            genes, counts = zip(*top_shared)
            ax.barh(genes, counts, color='purple', alpha=0.7)
            ax.set_xlabel('Number of Hallmarks')
            ax.set_ylabel('Gene Symbol')
            ax.set_title('Genes Shared Across Multiple Hallmarks')
        else:
            ax.text(0.5, 0.5, 'No shared genes found', ha='center', va='center')
            ax.set_title('Genes Shared Across Multiple Hallmarks')
        
        plt.suptitle('Learned Association Statistics', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'association_statistics.png', dpi=300)
        plt.close()
    
    def save_associations(self, associations: Dict, output_path: Path):
        """Save learned associations to file."""
        # Convert to JSON-serializable format
        json_associations = {
            'metadata': {
                'total_samples': self.total_samples,
                'hallmark_sample_counts': dict(self.hallmark_sample_counts),
                'creation_date': pd.Timestamp.now().isoformat()
            },
            'associations': associations
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_associations, f, indent=2)
        
        logger.info(f"Saved associations to {output_path}")
    
    def generate_report(self, associations: Dict, save_dir: Path):
        """Generate comprehensive report of learned associations."""
        report_path = save_dir / 'learned_associations_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("Learned Hallmark Associations Report\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total training samples analyzed: {self.total_samples}\n")
            f.write(f"Unique pathways found: {len(self.total_pathway_counts)}\n")
            f.write(f"Unique genes found: {len(self.total_gene_counts)}\n\n")
            
            for hallmark_id in range(11):
                if hallmark_id == 7:
                    continue
                
                hallmark_name = self.hallmark_names[hallmark_id]
                f.write(f"\n{hallmark_name.upper()}\n")
                f.write("-" * len(hallmark_name) + "\n")
                
                f.write(f"Training samples: {self.hallmark_sample_counts[hallmark_id]}\n\n")
                
                # Top pathways
                f.write("Top Associated Pathways:\n")
                pathways = associations['pathways'].get(hallmark_id, {})
                for i, (pathway_key, pathway_data) in enumerate(list(pathways.items())[:5]):
                    f.write(f"  {i+1}. {pathway_data['name']} ({pathway_data['id']})\n")
                    f.write(f"     Score: {pathway_data['score']:.3f}, "
                           f"Support: {pathway_data['support']:.3f}, "
                           f"Lift: {pathway_data['lift']:.2f}\n")
                
                # Top genes
                f.write("\nTop Associated Genes:\n")
                genes = associations['genes'].get(hallmark_id, {})
                gene_list = [g['symbol'] for g in list(genes.values())[:10]]
                f.write(f"  {', '.join(gene_list)}\n")
                
                # Other entities
                f.write("\nOther Key Entities:\n")
                entities = associations['entities'].get(hallmark_id, {})
                for entity_type, entity_list in entities.items():
                    if entity_list:
                        top_entities = [e['name'] for e in entity_list[:3]]
                        f.write(f"  {entity_type}: {', '.join(top_entities)}\n")
            
            # Cross-hallmark analysis
            f.write("\n\nCROSS-HALLMARK ANALYSIS\n")
            f.write("=" * 40 + "\n\n")
            
            # Shared pathways
            pathway_hallmark_map = defaultdict(list)
            for h_id, pathways in associations['pathways'].items():
                for pathway_key, pathway_data in pathways.items():
                    pathway_hallmark_map[pathway_data['name']].append(self.hallmark_names[h_id])
            
            f.write("Pathways Shared Across Multiple Hallmarks:\n")
            shared_pathways = [(p, h) for p, h in pathway_hallmark_map.items() if len(h) > 1]
            shared_pathways.sort(key=lambda x: len(x[1]), reverse=True)
            
            for pathway, hallmarks in shared_pathways[:10]:
                f.write(f"  - {pathway}:\n")
                f.write(f"    {', '.join(hallmarks)}\n")
            
            # Shared genes
            gene_hallmark_map = defaultdict(list)
            for h_id, genes in associations['genes'].items():
                for gene_key, gene_data in genes.items():
                    gene_hallmark_map[gene_data['symbol']].append(self.hallmark_names[h_id])
            
            f.write("\nGenes Shared Across Multiple Hallmarks:\n")
            shared_genes = [(g, h) for g, h in gene_hallmark_map.items() if len(h) > 1]
            shared_genes.sort(key=lambda x: len(x[1]), reverse=True)
            
            for gene, hallmarks in shared_genes[:10]:
                f.write(f"  - {gene}: {len(hallmarks)} hallmarks\n")
            
            # Key insights
            f.write("\n\nKEY INSIGHTS\n")
            f.write("=" * 40 + "\n\n")
            f.write("1. The associations are learned entirely from data, making them adaptive\n")
            f.write("2. Statistical measures (support, confidence, lift) ensure meaningful associations\n")
            f.write("3. Cross-hallmark shared entities reveal biological connections\n")
            f.write("4. These learned associations can improve mechanistic interpretability\n")
        
        logger.info(f"Report saved to {report_path}")


def main():
    """Main function to learn hallmark associations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Learn hallmark associations from cached KG data")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--output_dir', type=str, default='learned_associations',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize learner
    learner = HallmarkAssociationLearner(args.config)
    
    # Load cached dataset
    cached_data = learner.load_cached_dataset(args.split)
    
    # Extract associations
    learner.extract_associations(cached_data)
    
    # Calculate association scores
    associations = learner.calculate_association_scores()
    
    # Visualize associations
    learner.visualize_associations(associations, output_dir)
    
    # Save associations
    learner.save_associations(associations, output_dir / 'hallmark_associations.json')
    
    # Generate report
    learner.generate_report(associations, output_dir)
    
    logger.info(f"Association learning complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()