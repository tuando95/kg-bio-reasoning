"""
Test Reactome API endpoints directly to find the correct one
"""

import asyncio
import aiohttp
import json

async def test_reactome_endpoints():
    """Test various Reactome API endpoints"""
    
    test_genes = ['TP53', 'BRCA1', 'EGFR']
    
    endpoints = [
        # Try different endpoint variations
        {
            'name': 'Entity Query',
            'method': 'GET',
            'url': 'https://reactome.org/ContentService/data/query/enhanced/{identifier}',
            'test_id': 'TP53'
        },
        {
            'name': 'Search Query',
            'method': 'GET', 
            'url': 'https://reactome.org/ContentService/search/query?query={identifier}&species=Homo sapiens',
            'test_id': 'TP53'
        },
        {
            'name': 'Mapping POST',
            'method': 'POST',
            'url': 'https://reactome.org/ContentService/data/mapping',
            'data': 'TP53\nBRCA1',
            'headers': {'Content-Type': 'text/plain', 'Accept': 'application/json'}
        },
        {
            'name': 'Identifiers Projection',
            'method': 'POST',
            'url': 'https://reactome.org/ContentService/data/identifiers/projection',
            'data': 'TP53\nBRCA1',
            'headers': {'Content-Type': 'text/plain', 'Accept': 'application/json'}
        },
        {
            'name': 'Enhanced Mapping',
            'method': 'POST',
            'url': 'https://reactome.org/ContentService/data/mapping/ENSEMBL',
            'data': 'TP53\nBRCA1',
            'headers': {'Content-Type': 'text/plain', 'Accept': 'application/json'}
        }
    ]
    
    print("Testing Reactome API endpoints...")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            print(f"\nTesting: {endpoint['name']}")
            print(f"URL: {endpoint['url']}")
            
            try:
                if endpoint['method'] == 'GET':
                    url = endpoint['url'].replace('{identifier}', endpoint.get('test_id', 'TP53'))
                    async with session.get(url) as response:
                        print(f"Status: {response.status}")
                        if response.status == 200:
                            data = await response.json()
                            print(f"Success! Response keys: {list(data.keys())[:5]}")
                            if isinstance(data, list) and len(data) > 0:
                                print(f"First item keys: {list(data[0].keys())[:5]}")
                        else:
                            error_text = await response.text()
                            print(f"Error: {error_text[:100]}")
                            
                elif endpoint['method'] == 'POST':
                    async with session.post(
                        endpoint['url'], 
                        data=endpoint['data'],
                        headers=endpoint.get('headers', {})
                    ) as response:
                        print(f"Status: {response.status}")
                        if response.status == 200:
                            data = await response.json()
                            print(f"Success! Response type: {type(data)}")
                            if isinstance(data, list) and len(data) > 0:
                                print(f"First item: {json.dumps(data[0], indent=2)[:200]}")
                        else:
                            error_text = await response.text()
                            print(f"Error: {error_text[:100]}")
                            
            except Exception as e:
                print(f"Exception: {str(e)}")
    
    # Test specific working endpoint for pathways
    print("\n" + "=" * 60)
    print("Testing pathway retrieval for found entities...")
    
    async with aiohttp.ClientSession() as session:
        # First search for TP53
        search_url = "https://reactome.org/ContentService/search/query?query=TP53&species=Homo sapiens"
        
        async with session.get(search_url) as response:
            if response.status == 200:
                search_data = await response.json()
                
                # Find the first protein/gene result
                for result in search_data.get('results', []):
                    if result.get('typeName') in ['Protein', 'Gene', 'EntityWithAccessionedSequence']:
                        stId = result.get('stId')
                        print(f"\nFound TP53 entity: {stId}")
                        
                        # Get pathways for this entity
                        pathways_url = f"https://reactome.org/ContentService/data/pathways/low/entity/{stId}?species=9606"
                        
                        async with session.get(pathways_url) as pathway_response:
                            print(f"Pathways URL: {pathways_url}")
                            print(f"Pathways Status: {pathway_response.status}")
                            
                            if pathway_response.status == 200:
                                pathways = await pathway_response.json()
                                print(f"Found {len(pathways)} pathways for TP53")
                                if pathways:
                                    print(f"First pathway: {pathways[0].get('displayName')}")
                        break

if __name__ == "__main__":
    asyncio.run(test_reactome_endpoints())