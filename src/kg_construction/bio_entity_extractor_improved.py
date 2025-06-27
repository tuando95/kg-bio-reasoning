"""
Improved Biological Entity Extraction and Normalization Module

This module handles:
1. Biomedical Named Entity Recognition using ScispaCy
2. Better entity type classification
3. Entity normalization to standardized database identifiers
4. Cross-database mapping for comprehensive biological context
"""

import logging
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import spacy
from spacy.tokens import Doc, Span
import scispacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
import requests
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


@dataclass
class BioEntity:
    """Represents a biological entity with normalized identifiers"""
    text: str
    start: int
    end: int
    entity_type: str
    normalized_ids: Dict[str, str]  # database -> identifier mapping
    confidence: float
    context: Optional[str] = None


class BioEntityExtractor:
    """
    Extracts and normalizes biological entities from text using ScispaCy
    and maps them to standardized database identifiers.
    """
    
    # Expanded chemical elements and compounds
    CHEMICAL_ELEMENTS = {
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 
        'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 
        'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Ag', 'Au', 'Hg', 'Pb', 'Pt', 'Pd'
    }
    
    # Common chemical compounds
    CHEMICAL_COMPOUNDS = {
        'h2o', 'co2', 'nacl', 'h2so4', 'hcl', 'naoh', 'cao', 'mgcl2', 'feso4',
        'glucose', 'sucrose', 'lactose', 'ethanol', 'methanol', 'acetone',
        'atp', 'adp', 'amp', 'gtp', 'gdp', 'nad', 'nadh', 'nadp', 'nadph',
        'dna', 'rna', 'mrna', 'trna', 'rrna', 'mirna', 'sirna'
    }
    
    # Organism patterns
    ORGANISM_PATTERNS = {
        'mouse', 'mice', 'rat', 'rats', 'human', 'humans', 'patient', 'patients',
        'fish', 'zebrafish', 'drosophila', 'yeast', 'bacteria', 'virus', 
        'e. coli', 'escherichia coli', 'saccharomyces cerevisiae',
        'caenorhabditis elegans', 'c. elegans', 'xenopus', 'arabidopsis'
    }
    
    def __init__(self, config: Dict):
        """
        Initialize the entity extractor with ScispaCy model and linkers.
        
        Args:
            config: Configuration dictionary with entity extraction settings
        """
        self.config = config
        self.entity_types = set(config.get('entity_types', 
                                          ['GENE', 'PROTEIN', 'CHEMICAL', 'DISEASE', 
                                           'ORGANISM', 'PROCESS', 'CELL_TYPE']))
        self.confidence_threshold = config.get('confidence_threshold', 0.3)
        
        # Load ScispaCy model
        logger.info("Loading ScispaCy model...")
        self.nlp = spacy.load("en_core_sci_lg")
        
        # Add abbreviation detector
        self.nlp.add_pipe("abbreviation_detector")
        
        # Add entity linker for UMLS (optional - may fail if KB not downloaded)
        try:
            self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, 
                                                         "linker_name": "umls"})
            logger.info("UMLS entity linker added successfully")
        except Exception as e:
            logger.warning(f"Could not add UMLS entity linker: {e}")
            logger.warning("Continuing without UMLS normalization")
        
        # Initialize database mappers
        self._init_database_mappers()
        
        # Cache for API calls
        self.mapping_cache = {}
        
    def _init_database_mappers(self):
        """Initialize mappings between different biological databases"""
        self.database_mappers = {
            'GENE': {
                'primary': 'HGNC',
                'alternate': ['NCBI_GENE', 'ENSEMBL', 'UNIPROT']
            },
            'PROTEIN': {
                'primary': 'UNIPROT',
                'alternate': ['PDB', 'PFAM', 'INTERPRO']
            },
            'CHEMICAL': {
                'primary': 'CHEBI',
                'alternate': ['PUBCHEM', 'DRUGBANK', 'CHEMBL']
            },
            'DISEASE': {
                'primary': 'DO',
                'alternate': ['MESH', 'OMIM', 'HPO']
            },
            'ORGANISM': {
                'primary': 'NCBI_TAXON',
                'alternate': ['MESH']
            },
            'CELL_TYPE': {
                'primary': 'CL',  # Cell Ontology
                'alternate': ['MESH', 'EFO']
            }
        }
        
    def extract_entities(self, text: str) -> List[BioEntity]:
        """
        Extract biological entities from text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of BioEntity objects with normalized identifiers
        """
        # Process text with ScispaCy
        doc = self.nlp(text)
        
        entities = []
        seen_spans = set()  # Track seen spans to avoid duplicates
        
        # Extract named entities
        for ent in doc.ents:
            span_key = (ent.start_char, ent.end_char)
            if span_key not in seen_spans:
                entity = self._process_entity(ent, text)
                if entity and entity.confidence >= self.confidence_threshold:
                    entities.append(entity)
                    seen_spans.add(span_key)
        
        # Additional pattern-based extraction for missed entities
        additional_entities = self._extract_pattern_based_entities(doc, text, seen_spans)
        entities.extend(additional_entities)
        
        # Handle abbreviations
        if hasattr(doc._, "abbreviations"):
            for abbrev in doc._.abbreviations:
                entity = self._process_abbreviation(abbrev, text)
                if entity:
                    entities.append(entity)
        
        # Merge overlapping entities
        entities = self._merge_overlapping_entities(entities)
        
        return entities
    
    def _extract_pattern_based_entities(self, doc, text: str, seen_spans: Set) -> List[BioEntity]:
        """Extract entities based on patterns that ScispaCy might miss"""
        additional_entities = []
        
        # Extract chemical elements
        for token in doc:
            if token.text in self.CHEMICAL_ELEMENTS and len(token.text) <= 2:
                span_key = (token.idx, token.idx + len(token.text))
                if span_key not in seen_spans:
                    entity = BioEntity(
                        text=token.text,
                        start=token.idx,
                        end=token.idx + len(token.text),
                        entity_type='CHEMICAL',
                        normalized_ids={'SYMBOL': token.text},
                        confidence=0.9,
                        context=text[max(0, token.idx-50):min(len(text), token.idx+50)]
                    )
                    additional_entities.append(entity)
                    seen_spans.add(span_key)
        
        # Extract organisms
        text_lower = text.lower()
        for organism in self.ORGANISM_PATTERNS:
            start = 0
            while True:
                pos = text_lower.find(organism, start)
                if pos == -1:
                    break
                end = pos + len(organism)
                span_key = (pos, end)
                if span_key not in seen_spans:
                    # Check word boundaries
                    if (pos == 0 or not text[pos-1].isalnum()) and \
                       (end == len(text) or not text[end].isalnum()):
                        entity = BioEntity(
                            text=text[pos:end],
                            start=pos,
                            end=end,
                            entity_type='ORGANISM',
                            normalized_ids={'NAME': organism},
                            confidence=0.8,
                            context=text[max(0, pos-50):min(len(text), end+50)]
                        )
                        additional_entities.append(entity)
                        seen_spans.add(span_key)
                start = pos + 1
        
        return additional_entities
    
    def _process_entity(self, ent: Span, full_text: str) -> Optional[BioEntity]:
        """Process a single entity and normalize it"""
        try:
            # Get entity type with improved mapping
            entity_type = self._determine_entity_type(ent)
            
            # Skip if not a relevant entity type
            if entity_type not in self.entity_types:
                return None
            
            # Get UMLS concepts if available
            umls_ids = []
            confidence = 0.5  # Default confidence
            
            if hasattr(ent._, "kb_ents") and ent._.kb_ents:
                # Get top UMLS concept
                top_concept = ent._.kb_ents[0]
                umls_ids.append(top_concept[0])
                confidence = top_concept[1]
            
            # Normalize entity text
            normalized_text = self._normalize_entity_text(ent.text, entity_type)
            
            # Get database mappings
            normalized_ids = self._get_database_mappings(
                normalized_text, entity_type, umls_ids
            )
            
            # Extract context
            context_start = max(0, ent.start_char - 50)
            context_end = min(len(full_text), ent.end_char + 50)
            context = full_text[context_start:context_end]
            
            return BioEntity(
                text=ent.text,
                start=ent.start_char,
                end=ent.end_char,
                entity_type=entity_type,
                normalized_ids=normalized_ids,
                confidence=confidence,
                context=context
            )
            
        except Exception as e:
            logger.warning(f"Error processing entity {ent.text}: {e}")
            return None
    
    def _determine_entity_type(self, ent: Span) -> str:
        """Determine entity type with improved logic"""
        # ALWAYS check for chemical elements first to avoid misclassification
        if ent.text.upper() in self.CHEMICAL_ELEMENTS:
            return 'CHEMICAL'
        
        # Check for organisms
        if ent.text.lower() in self.ORGANISM_PATTERNS:
            return 'ORGANISM'
            
        # Extended label mapping
        label_mapping = {
            'GENE_OR_GENE_PRODUCT': 'GENE',
            'PROTEIN': 'PROTEIN',
            'CHEMICAL': 'CHEMICAL',
            'SIMPLE_CHEMICAL': 'CHEMICAL',
            'AMINO_ACID': 'CHEMICAL',
            'DISEASE': 'DISEASE',
            'CANCER': 'DISEASE',
            'PATHOLOGICAL_FORMATION': 'DISEASE',
            'CELL_TYPE': 'CELL_TYPE',
            'CELL_LINE': 'CELL_TYPE',
            'CELL': 'CELL_TYPE',
            'TISSUE': 'TISSUE',
            'ORGAN': 'ORGAN',
            'ORGANISM': 'ORGANISM',
            'BIOLOGICAL_PROCESS': 'PROCESS',
            'MOLECULAR_FUNCTION': 'PROCESS',
            'DNA': 'GENE',
            'RNA': 'GENE',
            'ENTITY': None  # Will infer type
        }
        
        # Get mapped type
        mapped_type = label_mapping.get(ent.label_)
        
        # If no mapping or ENTITY, infer type from text
        if mapped_type is None:
            mapped_type = self._infer_entity_type(ent.text)
        
        return mapped_type
    
    def _normalize_entity_text(self, text: str, entity_type: str) -> str:
        """Normalize entity text based on entity type"""
        text = text.strip()
        
        if entity_type == 'GENE' or entity_type == 'PROTEIN':
            # Handle gene/protein variations
            # p53 -> TP53, P53 -> TP53
            if re.match(r'^[pP]\d+$', text):
                text = 'TP' + text[1:]
            
            # Remove hyphens in some cases
            if '-' in text and not text.startswith('HIF-'):
                text = text.replace('-', '')
            
            # Handle Greek letters
            greek_map = {
                'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
                'Alpha': 'α', 'Beta': 'β', 'Gamma': 'γ', 'Delta': 'δ'
            }
            for word, letter in greek_map.items():
                text = text.replace(word, letter)
        
        elif entity_type == 'CHEMICAL':
            # Normalize chemical names
            if text.upper() in self.CHEMICAL_ELEMENTS:
                text = text.upper()
            else:
                text = text.lower()
        
        return text
    
    def _get_database_mappings(self, normalized_text: str, entity_type: str, 
                              umls_ids: List[str]) -> Dict[str, str]:
        """
        Get database mappings for an entity.
        
        Returns dictionary mapping database names to identifiers.
        """
        # Check cache first
        cache_key = f"{normalized_text}_{entity_type}"
        if cache_key in self.mapping_cache:
            return self.mapping_cache[cache_key]
        
        mappings = {}
        
        # Add UMLS IDs if available
        if umls_ids:
            mappings['UMLS'] = umls_ids[0]
        
        # Get mappings based on entity type
        if entity_type == 'GENE':
            mappings.update(self._map_gene(normalized_text))
        elif entity_type == 'PROTEIN':
            mappings.update(self._map_protein(normalized_text))
        elif entity_type == 'CHEMICAL':
            mappings.update(self._map_chemical(normalized_text))
        elif entity_type == 'DISEASE':
            mappings.update(self._map_disease(normalized_text))
        elif entity_type == 'ORGANISM':
            mappings.update(self._map_organism(normalized_text))
        
        # Cache the result
        self.mapping_cache[cache_key] = mappings
        
        return mappings
    
    def _map_gene(self, gene_name: str) -> Dict[str, str]:
        """Map gene name to various database identifiers"""
        mappings = {}
        
        # Extended gene mappings
        gene_aliases = {
            'TP53': {'HGNC': '11998', 'NCBI_GENE': '7157', 'ENSEMBL': 'ENSG00000141510'},
            'EGFR': {'HGNC': '3236', 'NCBI_GENE': '1956', 'ENSEMBL': 'ENSG00000146648'},
            'KRAS': {'HGNC': '6407', 'NCBI_GENE': '3845', 'ENSEMBL': 'ENSG00000133703'},
            'BRCA1': {'HGNC': '1100', 'NCBI_GENE': '672', 'ENSEMBL': 'ENSG00000012048'},
            'BRCA2': {'HGNC': '1101', 'NCBI_GENE': '675', 'ENSEMBL': 'ENSG00000139618'},
            'MYC': {'HGNC': '7553', 'NCBI_GENE': '4609', 'ENSEMBL': 'ENSG00000136997'},
            'PTEN': {'HGNC': '9588', 'NCBI_GENE': '5728', 'ENSEMBL': 'ENSG00000171862'},
            'RB1': {'HGNC': '9884', 'NCBI_GENE': '5925', 'ENSEMBL': 'ENSG00000139687'},
            'APC': {'HGNC': '583', 'NCBI_GENE': '324', 'ENSEMBL': 'ENSG00000134982'},
            'PIK3CA': {'HGNC': '8975', 'NCBI_GENE': '5290', 'ENSEMBL': 'ENSG00000121879'},
            'VEGFA': {'HGNC': '12680', 'NCBI_GENE': '7422', 'ENSEMBL': 'ENSG00000112715'},
            'VEGF': {'HGNC': '12680', 'NCBI_GENE': '7422', 'ENSEMBL': 'ENSG00000112715'},
            'BCL2': {'HGNC': '990', 'NCBI_GENE': '596', 'ENSEMBL': 'ENSG00000171791'},
            'BRAF': {'HGNC': '1097', 'NCBI_GENE': '673', 'ENSEMBL': 'ENSG00000157764'},
            'AKT1': {'HGNC': '391', 'NCBI_GENE': '207', 'ENSEMBL': 'ENSG00000142208'},
            'ERBB2': {'HGNC': '3430', 'NCBI_GENE': '2064', 'ENSEMBL': 'ENSG00000141736'},
            'HER2': {'HGNC': '3430', 'NCBI_GENE': '2064', 'ENSEMBL': 'ENSG00000141736'},
            'NFKB1': {'HGNC': '7794', 'NCBI_GENE': '4790', 'ENSEMBL': 'ENSG00000109320'},
            'STAT3': {'HGNC': '11364', 'NCBI_GENE': '6774', 'ENSEMBL': 'ENSG00000168610'},
            'HIF1A': {'HGNC': '4910', 'NCBI_GENE': '3091', 'ENSEMBL': 'ENSG00000100644'},
            'HIF1α': {'HGNC': '4910', 'NCBI_GENE': '3091', 'ENSEMBL': 'ENSG00000100644'},
        }
        
        gene_upper = gene_name.upper()
        if gene_upper in gene_aliases:
            mappings.update(gene_aliases[gene_upper])
        
        return mappings
    
    def _map_protein(self, protein_name: str) -> Dict[str, str]:
        """Map protein name to database identifiers"""
        mappings = {}
        
        # Common protein mappings
        protein_aliases = {
            'P53': {'UNIPROT': 'P04637', 'PDB': '1TUP'},
            'TP53': {'UNIPROT': 'P04637', 'PDB': '1TUP'},
            'EGFR': {'UNIPROT': 'P00533', 'PDB': '1IVO'},
            'KRAS': {'UNIPROT': 'P01116', 'PDB': '4OBE'},
            'AKT1': {'UNIPROT': 'P31749', 'PDB': '1UNQ'},
            'VEGFA': {'UNIPROT': 'P15692', 'PDB': '1VPF'},
            'BCL2': {'UNIPROT': 'P10415', 'PDB': '2W3L'},
            'BRAF': {'UNIPROT': 'P15056', 'PDB': '3OG7'},
        }
        
        if protein_name.upper() in protein_aliases:
            mappings.update(protein_aliases[protein_name.upper()])
        
        return mappings
    
    def _map_chemical(self, chemical_name: str) -> Dict[str, str]:
        """Map chemical/drug name to database identifiers"""
        mappings = {}
        
        # Extended chemical/drug mappings
        chemical_aliases = {
            # Drugs
            'tamoxifen': {'CHEBI': '41774', 'PUBCHEM': '2733526', 'DRUGBANK': 'DB00675'},
            'doxorubicin': {'CHEBI': '28748', 'PUBCHEM': '31703', 'DRUGBANK': 'DB00997'},
            'cisplatin': {'CHEBI': '27899', 'PUBCHEM': '441203', 'DRUGBANK': 'DB00515'},
            'paclitaxel': {'CHEBI': '45863', 'PUBCHEM': '36314', 'DRUGBANK': 'DB01229'},
            
            # Elements
            'copper': {'CHEBI': '29036', 'PUBCHEM': '23978'},
            'cu': {'CHEBI': '29036', 'PUBCHEM': '23978'},
            'iron': {'CHEBI': '18248', 'PUBCHEM': '23925'},
            'fe': {'CHEBI': '18248', 'PUBCHEM': '23925'},
            'zinc': {'CHEBI': '29105', 'PUBCHEM': '23994'},
            'zn': {'CHEBI': '29105', 'PUBCHEM': '23994'},
            
            # Common compounds
            'glucose': {'CHEBI': '17234', 'PUBCHEM': '5793'},
            'atp': {'CHEBI': '15422', 'PUBCHEM': '5957'},
            'nadh': {'CHEBI': '16908', 'PUBCHEM': '439153'},
        }
        
        chemical_lower = chemical_name.lower()
        if chemical_lower in chemical_aliases:
            mappings.update(chemical_aliases[chemical_lower])
        
        return mappings
    
    def _map_disease(self, disease_name: str) -> Dict[str, str]:
        """Map disease name to database identifiers"""
        mappings = {}
        
        # Common disease mappings
        disease_aliases = {
            'breast cancer': {'DO': 'DOID:1612', 'MESH': 'D001943', 'OMIM': '114480'},
            'lung cancer': {'DO': 'DOID:1324', 'MESH': 'D008175', 'OMIM': '211980'},
            'colorectal cancer': {'DO': 'DOID:9256', 'MESH': 'D015179', 'OMIM': '114500'},
            'melanoma': {'DO': 'DOID:1909', 'MESH': 'D008545', 'OMIM': '155600'},
            'leukemia': {'DO': 'DOID:1240', 'MESH': 'D007938', 'OMIM': '601626'},
            'lymphoma': {'DO': 'DOID:0060058', 'MESH': 'D008223'},
            'glioblastoma': {'DO': 'DOID:3068', 'MESH': 'D005909'},
            'prostate cancer': {'DO': 'DOID:10283', 'MESH': 'D011471'},
            'ovarian cancer': {'DO': 'DOID:2394', 'MESH': 'D010051'},
            'pancreatic cancer': {'DO': 'DOID:1793', 'MESH': 'D010190'},
        }
        
        if disease_name.lower() in disease_aliases:
            mappings.update(disease_aliases[disease_name.lower()])
        
        return mappings
    
    def _map_organism(self, organism_name: str) -> Dict[str, str]:
        """Map organism name to database identifiers"""
        mappings = {}
        
        organism_aliases = {
            'human': {'NCBI_TAXON': '9606', 'MESH': 'D006801'},
            'mouse': {'NCBI_TAXON': '10090', 'MESH': 'D051379'},
            'rat': {'NCBI_TAXON': '10116', 'MESH': 'D051381'},
            'zebrafish': {'NCBI_TAXON': '7955', 'MESH': 'D015027'},
            'drosophila': {'NCBI_TAXON': '7227', 'MESH': 'D004330'},
            'yeast': {'NCBI_TAXON': '4932', 'MESH': 'D012441'},
            'e. coli': {'NCBI_TAXON': '562', 'MESH': 'D004926'},
            'c. elegans': {'NCBI_TAXON': '6239', 'MESH': 'D017173'},
        }
        
        if organism_name.lower() in organism_aliases:
            mappings.update(organism_aliases[organism_name.lower()])
        
        return mappings
    
    def _process_abbreviation(self, abbrev, full_text: str) -> Optional[BioEntity]:
        """Process abbreviations detected by ScispaCy"""
        # Get the long form
        long_form = abbrev._.long_form
        
        # Process the long form as an entity
        if long_form:
            # Create a temporary span for the long form
            temp_ent = self._create_temp_span(long_form, full_text)
            if temp_ent:
                entity = self._process_entity(temp_ent, full_text)
                if entity:
                    # Update with abbreviation info
                    entity.text = f"{abbrev.text} ({long_form.text})"
                    return entity
        
        return None
    
    def _create_temp_span(self, span, full_text: str):
        """Create a temporary span object for processing"""
        # This is a simplified implementation
        # In production, would properly create a Span object
        return span
    
    def _merge_overlapping_entities(self, entities: List[BioEntity]) -> List[BioEntity]:
        """Merge overlapping entities, keeping the most specific one"""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda e: (e.start, -e.end))
        
        merged = []
        current = entities[0]
        
        for entity in entities[1:]:
            # Check for overlap
            if entity.start < current.end:
                # Keep the one with higher confidence or more specific type
                # Chemical elements should have high priority to avoid misclassification
                type_priority = {'CHEMICAL': 6, 'GENE': 5, 'PROTEIN': 4, 
                               'DISEASE': 3, 'ORGANISM': 3, 'CELL_TYPE': 2, 'PROCESS': 1}
                
                current_priority = type_priority.get(current.entity_type, 0)
                entity_priority = type_priority.get(entity.entity_type, 0)
                
                if entity_priority > current_priority or \
                   (entity_priority == current_priority and entity.confidence > current.confidence):
                    current = entity
            else:
                merged.append(current)
                current = entity
        
        merged.append(current)
        return merged
    
    def _infer_entity_type(self, text: str) -> str:
        """Infer entity type from text patterns with improved logic"""
        text_upper = text.upper()
        text_lower = text.lower()
        
        # Check for chemical elements first
        if text_upper in self.CHEMICAL_ELEMENTS:
            return 'CHEMICAL'
        
        # Check for chemical compounds
        if text_lower in self.CHEMICAL_COMPOUNDS:
            return 'CHEMICAL'
        
        # Check for organisms
        if text_lower in self.ORGANISM_PATTERNS:
            return 'ORGANISM'
        
        # Known gene/protein patterns
        gene_patterns = [
            'TP53', 'P53', 'EGFR', 'VEGF', 'VEGFA', 'BRCA1', 'BRCA2', 'KRAS', 
            'PIK3CA', 'AKT1', 'MYC', 'PTEN', 'RB1', 'APC', 'CDK4', 'CDK6',
            'MAPK', 'ERK', 'RAF', 'RAS', 'HER2', 'ERBB2', 'BCL2', 'BAX',
            'CDKN1A', 'CDKN2A', 'MDM2', 'ATM', 'CHEK2', 'MLH1', 'MSH2',
            'NFKB', 'STAT3', 'HIF1A', 'TGFB1', 'TNF', 'IL6', 'CD274', 'PDCD1'
        ]
        
        # Disease patterns
        disease_patterns = ['cancer', 'tumor', 'tumour', 'carcinoma', 'melanoma', 'leukemia', 
                          'lymphoma', 'sarcoma', 'adenoma', 'neoplasm', 'metastasis',
                          'malignancy', 'oncogene', 'syndrome']
        
        # Process patterns
        process_patterns = ['apoptosis', 'proliferation', 'angiogenesis', 'metastasis',
                          'migration', 'invasion', 'adhesion', 'signaling', 'pathway',
                          'expression', 'mutation', 'activation', 'inhibition',
                          'phosphorylation', 'methylation', 'acetylation', 'ubiquitination',
                          'transcription', 'translation', 'replication', 'repair',
                          'differentiation', 'development', 'growth', 'death',
                          'response', 'regulation', 'metabolism', 'synthesis']
        
        # Chemical patterns
        chemical_patterns = ['drug', 'compound', 'inhibitor', 'antibody', 'molecule',
                           'acid', 'base', 'salt', 'ion', 'radical', 'substrate',
                           'product', 'reagent', 'catalyst', 'enzyme']
        
        # Cell type patterns
        cell_patterns = ['cell', 'cells', 'lymphocyte', 'macrophage', 'neutrophil',
                        'fibroblast', 'epithelial', 'endothelial', 'stem cell',
                        't cell', 'b cell', 'nk cell', 'dendritic cell']
        
        # Check patterns with priority
        if any(gene in text_upper for gene in gene_patterns):
            return 'GENE'
        elif any(disease in text_lower for disease in disease_patterns):
            return 'DISEASE'
        elif any(cell in text_lower for cell in cell_patterns):
            return 'CELL_TYPE'
        elif any(chem in text_lower for chem in chemical_patterns):
            return 'CHEMICAL'
        elif any(proc in text_lower for proc in process_patterns):
            return 'PROCESS'
        elif text_upper.endswith('IN') or text_upper.endswith('ASE') or text_upper.endswith('OR'):
            return 'PROTEIN'  # Common protein name endings
        elif re.match(r'^[A-Z]+\d+[A-Z]*$', text_upper):  # Like CD4, BCL2, etc.
            return 'GENE'
        elif text.isupper() and len(text) > 2:
            return 'GENE'  # Likely gene acronym
        else:
            # Default based on capitalization
            if text[0].isupper() and not text.islower():
                return 'GENE'
            return 'PROCESS'  # Default for other entities