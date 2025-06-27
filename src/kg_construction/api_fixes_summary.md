# Biological Knowledge Graph API Integration Fixes

## Summary of Issues Fixed

### 1. STRING API Fixes

#### Original Issues:
- **Incorrect parameter encoding**: Used `'%0d'.join()` which creates URL-encoded newlines
- **Wrong identifier format**: STRING expects space-separated identifiers
- **Poor gene name mapping**: Simple p53 → TP53 mapping insufficient
- **Response parsing errors**: Expected JSON format but got TSV

#### Fixes Applied:
```python
# BEFORE:
params = {
    'identifiers': '%0d'.join(gene_names[:10]),  # Wrong!
    ...
}

# AFTER:
params = {
    'identifiers': ' '.join(gene_names),  # Space-separated
    ...
}
```

- Changed URL to use TSV endpoint: `https://string-db.org/api/tsv/network`
- Added comprehensive gene name normalization mapping
- Fixed response parsing to handle TSV format correctly
- Improved node matching logic

### 2. KEGG API Fixes

#### Original Issues:
- **Missing gene ID conversion**: No mapping from gene names to KEGG IDs
- **Wrong API endpoint**: Used configured endpoint instead of REST API
- **No rate limiting**: KEGG requires max 3 requests/second

#### Fixes Applied:
```python
# Added gene name to KEGG ID mapping
gene_to_kegg_id = {
    'TP53': '7157',
    'EGFR': '1956',
    'KRAS': '3845',
    # ... more mappings
}

# Added rate limiting
self.kegg_delay = 0.35  # ~3 calls per second
```

- Fixed endpoint: `https://rest.kegg.jp/get/hsa:{gene_id}`
- Added comprehensive gene-to-KEGG-ID mapping
- Implemented rate limiting to respect API limits

### 3. Reactome API Fixes

#### Original Issues:
- **Using deprecated API**: Old RESTful API was deprecated in 2019
- **Wrong endpoint**: `/identifiers/projection` doesn't exist
- **Incorrect request format**: Wrong content type and data format

#### Fixes Applied:
```python
# BEFORE (deprecated):
url = f"{api_endpoint}/identifiers/projection"

# AFTER (ContentService):
url = "https://reactome.org/ContentService/data/mapping"
# Then fetch pathways:
pathway_url = f"https://reactome.org/ContentService/data/pathways/low/entity/{id}/9606"
```

- Updated to use new ContentService API
- Implemented two-step process: mapping then pathway fetching
- Fixed request headers and data format

### 4. Entity Normalization Improvements

#### Original Issues:
- **Oversimplified normalization**: Only handled p53 → TP53
- **No comprehensive mapping**: Missing common gene aliases
- **Duplicate entities**: Same gene with different names created multiple nodes

#### Fixes Applied:
```python
# Added comprehensive gene name mapping
GENE_NAME_MAPPINGS = {
    'P53': 'TP53',
    'HER2': 'ERBB2',
    'BCL-2': 'BCL2',
    'PD-L1': 'CD274',
    # ... 40+ mappings
}
```

- Added comprehensive gene alias mapping
- Implemented duplicate detection
- Improved entity ID generation

### 5. Error Handling and Logging

#### Original Issues:
- **Silent failures**: API errors not properly logged
- **No debugging info**: Insufficient logging for troubleshooting
- **Missing fallbacks**: No graceful degradation

#### Fixes Applied:
- Added detailed logging at INFO and DEBUG levels
- Added try-catch blocks with specific error messages
- Implemented cache to reduce API calls
- Added fallback edges when APIs fail

## Usage Example

```python
from kg_construction.kg_builder_fixed import BiologicalKGBuilder

# Configure with all databases
config = {
    'databases': [
        {
            'name': 'STRING',
            'enabled': True,
            'api_endpoint': 'https://string-db.org/api',
            'confidence_threshold': 400
        },
        {
            'name': 'KEGG',
            'enabled': True,
            'api_endpoint': 'https://rest.kegg.jp'
        },
        {
            'name': 'Reactome',
            'enabled': True,
            'api_endpoint': 'https://reactome.org/ContentService'
        }
    ],
    'graph_construction': {
        'max_hops': 1,
        'max_neighbors': 100,
        'include_pathways': True
    }
}

# Build knowledge graph
builder = BiologicalKGBuilder(config)
kg = await builder.build_knowledge_graph(entities, hallmarks)
```

## Testing

Run the test script to verify all APIs are working:

```bash
python test_api_integration.py
```

Expected output:
- STRING: Should find protein-protein interactions
- KEGG: Should find pathway memberships
- Reactome: Should find pathway associations
- Combined: Should create a comprehensive knowledge graph

## Key Improvements

1. **Reliability**: APIs now return actual biological relationships
2. **Accuracy**: Proper gene name normalization ensures correct matches
3. **Performance**: Caching reduces redundant API calls
4. **Debugging**: Comprehensive logging helps troubleshoot issues
5. **Robustness**: Fallback mechanisms ensure graph construction continues even if some APIs fail