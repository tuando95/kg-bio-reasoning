"""
Biological Entity Extraction and Normalization Module

This module handles:
1. Biomedical Named Entity Recognition using ScispaCy
2. Entity normalization to standardized database identifiers
3. Cross-database mapping for comprehensive biological context
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
    
    def __init__(self, config: Dict):
        """
        Initialize the entity extractor with ScispaCy model and linkers.
        
        Args:
            config: Configuration dictionary with entity extraction settings
        """
        self.config = config
        self.entity_types = set(config.get('entity_types', 
                                          ['GENE', 'PROTEIN', 'CHEMICAL', 'DISEASE']))
        self.confidence_threshold = config.get('confidence_threshold', 0.3)  # Lower threshold for more entities
        
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
        
        # Extract named entities
        for ent in doc.ents:
            if self._is_relevant_entity(ent):
                entity = self._process_entity(ent, text)
                if entity and entity.confidence >= self.confidence_threshold:
                    entities.append(entity)
        
        # Handle abbreviations
        if hasattr(doc._, "abbreviations"):
            for abbrev in doc._.abbreviations:
                entity = self._process_abbreviation(abbrev, text)
                if entity:
                    entities.append(entity)
        
        # Merge overlapping entities
        entities = self._merge_overlapping_entities(entities)
        
        return entities
    
    def _is_relevant_entity(self, ent: Span) -> bool:
        """Check if entity type is relevant for our use case"""
        # Map ScispaCy labels to our entity types
        label_mapping = {
            'GENE_OR_GENE_PRODUCT': 'GENE',
            'PROTEIN': 'PROTEIN',
            'CHEMICAL': 'CHEMICAL',
            'DISEASE': 'DISEASE',
            'CANCER': 'DISEASE',
            'CELL_TYPE': 'CELL_TYPE',
            'CELL_LINE': 'CELL_LINE',
            'DNA': 'GENE',
            'RNA': 'GENE',
            'AMINO_ACID': 'CHEMICAL',
            'SIMPLE_CHEMICAL': 'CHEMICAL',
            'CELL': 'CELL_TYPE',
            'TISSUE': 'TISSUE',
            'ORGAN': 'ORGAN',
            'ORGANISM': 'ORGANISM',
            'BIOLOGICAL_PROCESS': 'PROCESS',
            'MOLECULAR_FUNCTION': 'FUNCTION',
            'PATHOLOGICAL_FORMATION': 'DISEASE',
            'ENTITY': 'GENE'  # Default mapping for generic entities
        }
        
        mapped_type = label_mapping.get(ent.label_, ent.label_)
        
        # For generic ENTITY labels, try to infer type from text
        if ent.label_ == 'ENTITY' and mapped_type not in self.entity_types:
            mapped_type = self._infer_entity_type(ent.text)
        
        return mapped_type in self.entity_types
    
    def _process_entity(self, ent: Span, full_text: str) -> Optional[BioEntity]:
        """Process a single entity and normalize it"""
        try:
            # Get entity type
            label_mapping = {
                'GENE_OR_GENE_PRODUCT': 'GENE',
                'PROTEIN': 'PROTEIN', 
                'CHEMICAL': 'CHEMICAL',
                'DISEASE': 'DISEASE',
                'CANCER': 'DISEASE',
                'ENTITY': 'GENE'  # Default for generic entities
            }
            entity_type = label_mapping.get(ent.label_, ent.label_)
            
            # For generic ENTITY labels, try to infer type from text
            if ent.label_ == 'ENTITY':
                entity_type = self._infer_entity_type(ent.text)
            
            # Get UMLS concepts if available
            umls_ids = []
            confidence = 0.5  # Default confidence
            
            if hasattr(ent._, "kb_ents") and ent._.kb_ents:
                # Get top UMLS concept
                top_concept = ent._.kb_ents[0]
                umls_ids.append(top_concept[0])
                confidence = top_concept[1]
            
            # Normalize entity text
            normalized_text = self._normalize_entity_text(ent.text)
            
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
    
    def _normalize_entity_text(self, text: str) -> str:
        """Normalize entity text for better matching"""
        # Common normalizations for biological entities
        text = text.strip()
        
        # Handle gene/protein variations
        # p53 -> TP53, P53 -> TP53
        if re.match(r'^[pP]\d+$', text):
            text = 'TP' + text[1:]
        
        # Handle Greek letters
        greek_map = {
            'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
            'Alpha': 'α', 'Beta': 'β', 'Gamma': 'γ', 'Delta': 'δ'
        }
        for word, letter in greek_map.items():
            text = text.replace(word, letter)
        
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
        
        # Cache the result
        self.mapping_cache[cache_key] = mappings
        
        return mappings
    
    def _map_gene(self, gene_name: str) -> Dict[str, str]:
        """Map gene name to various database identifiers"""
        mappings = {}
        
        # Common gene mappings
        # This is a simplified version - in production, would query actual APIs
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
            'PIK3CA': {'HGNC': '8975', 'NCBI_GENE': '5290', 'ENSEMBL': 'ENSG00000121879'}
        }
        
        if gene_name.upper() in gene_aliases:
            mappings.update(gene_aliases[gene_name.upper()])
        
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
            'AKT1': {'UNIPROT': 'P31749', 'PDB': '1UNQ'}
        }
        
        if protein_name.upper() in protein_aliases:
            mappings.update(protein_aliases[protein_name.upper()])
        
        return mappings
    
    def _map_chemical(self, chemical_name: str) -> Dict[str, str]:
        """Map chemical/drug name to database identifiers"""
        mappings = {}
        
        # Common chemical/drug mappings
        chemical_aliases = {
            'tamoxifen': {'CHEBI': '41774', 'PUBCHEM': '2733526', 'DRUGBANK': 'DB00675'},
            'doxorubicin': {'CHEBI': '28748', 'PUBCHEM': '31703', 'DRUGBANK': 'DB00997'},
            'cisplatin': {'CHEBI': '27899', 'PUBCHEM': '441203', 'DRUGBANK': 'DB00515'},
            'paclitaxel': {'CHEBI': '45863', 'PUBCHEM': '36314', 'DRUGBANK': 'DB01229'}
        }
        
        if chemical_name.lower() in chemical_aliases:
            mappings.update(chemical_aliases[chemical_name.lower()])
        
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
            'leukemia': {'DO': 'DOID:1240', 'MESH': 'D007938', 'OMIM': '601626'}
        }
        
        if disease_name.lower() in disease_aliases:
            mappings.update(disease_aliases[disease_name.lower()])
        
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
                # Keep the one with higher confidence or longer span
                if entity.confidence > current.confidence or \
                   (entity.end - entity.start) > (current.end - current.start):
                    current = entity
            else:
                merged.append(current)
                current = entity
        
        merged.append(current)
        return merged
    
    def _infer_entity_type(self, text: str) -> str:
        """Infer entity type from text patterns"""
        text_upper = text.upper()
        
        # Known gene/protein patterns
        gene_patterns = [
            'TP53', 'P53', 'EGFR', 'VEGF', 'VEGFA', 'BRCA1', 'BRCA2', 'KRAS', 
            'PIK3CA', 'AKT1', 'MYC', 'PTEN', 'RB1', 'APC', 'CDK4', 'CDK6',
            'MAPK', 'ERK', 'RAF', 'RAS', 'HER2', 'ERBB2', 'BCL2', 'BAX',
            'CDKN1A', 'CDKN2A', 'MDM2', 'ATM', 'CHEK2', 'MLH1', 'MSH2'
        ]
        
        # Disease patterns
        disease_patterns = ['cancer', 'tumor', 'carcinoma', 'melanoma', 'leukemia', 
                          'lymphoma', 'sarcoma', 'adenoma', 'neoplasm']
        
        # Process patterns
        process_patterns = ['apoptosis', 'proliferation', 'angiogenesis', 'metastasis',
                          'migration', 'invasion', 'adhesion', 'signaling', 'pathway',
                          'expression', 'mutation', 'activation', 'inhibition']
        
        # Chemical patterns
        chemical_patterns = ['drug', 'compound', 'inhibitor', 'antibody', 'molecule']
        
        # Check patterns
        if any(gene in text_upper for gene in gene_patterns):
            return 'GENE'
        elif any(disease in text.lower() for disease in disease_patterns):
            return 'DISEASE'
        elif any(proc in text.lower() for proc in process_patterns):
            return 'PROCESS'
        elif any(chem in text.lower() for chem in chemical_patterns):
            return 'CHEMICAL'
        elif text_upper.endswith('IN') or text_upper.endswith('ASE') or text_upper.endswith('OR'):
            return 'PROTEIN'  # Common protein name endings
        else:
            # Default to GENE for entities that look like gene names
            if text.isupper() or (text[0].isupper() and any(c.isdigit() for c in text)):
                return 'GENE'
            return 'PROCESS'  # Default for other entities