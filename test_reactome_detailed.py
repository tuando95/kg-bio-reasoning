"""
Detailed Reactome API debugging to understand response structure
"""

import asyncio
import aiohttp
import json

async def test_reactome_detailed():
    """Test Reactome API with detailed response inspection"""
    
    print("Detailed Reactome API Testing")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Search and inspect full response structure
        print("\n1. Search for TP53 with full response inspection:")
        search_url = "https://reactome.org/ContentService/search/query?query=TP53&species=Homo sapiens"
        
        async with session.get(search_url) as response:
            if response.status == 200:
                data = await response.json()
                
                # Inspect the structure
                print(f"Response keys: {list(data.keys())}")
                print(f"Number of results: {len(data.get('results', []))}")
                
                # Look at first result in detail
                if data.get('results'):
                    first_result = data['results'][0]
                    print(f"\nFirst result structure:")
                    print(f"  All keys: {list(first_result.keys())}")
                    
                    # Print all key-value pairs
                    for key, value in first_result.items():
                        if isinstance(value, (str, int, float, bool, type(None))):
                            print(f"  {key}: {value}")
                        elif isinstance(value, list):
                            print(f"  {key}: [list with {len(value)} items]")
                            if value and len(value) > 0:
                                print(f"    First item: {value[0]}")
                        elif isinstance(value, dict):
                            print(f"  {key}: [dict with keys: {list(value.keys())}]")
                    
                    # Look for anything TP53-related in name fields
                    print("\nSearching for TP53-related entries:")
                    for i, result in enumerate(data['results'][:5]):
                        for key, value in result.items():
                            if isinstance(value, str) and 'TP53' in value.upper():
                                print(f"  Result {i}, field '{key}': {value}")
                            elif key in ['stId', 'dbId', 'identifier', 'id', 'displayName', 'name']:
                                print(f"  Result {i}, {key}: {value}")
        
        # Test 2: Try different search parameters
        print("\n\n2. Testing different search approaches:")
        
        # Try searching with scope parameter
        search_urls = [
            ("Search with scope=Pathways", "https://reactome.org/ContentService/search/query?query=TP53&species=Homo sapiens&types=Pathway"),
            ("Search with scope=Proteins", "https://reactome.org/ContentService/search/query?query=TP53&species=Homo sapiens&types=Protein"),
            ("Search without species", "https://reactome.org/ContentService/search/query?query=TP53"),
            ("Search for P04637 (UniProt)", "https://reactome.org/ContentService/search/query?query=P04637")
        ]
        
        for desc, url in search_urls:
            print(f"\n  {desc}:")
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get('results', [])
                    print(f"    Found {len(results)} results")
                    
                    # Look for useful results
                    for result in results[:3]:
                        type_name = result.get('typeName', 'Unknown')
                        # Check different possible name fields
                        name = None
                        for name_field in ['name', 'displayName', 'identifier', 'primaryIdentifier']:
                            if name_field in result and result[name_field]:
                                name = result[name_field]
                                break
                        
                        # Check for IDs
                        stId = result.get('stId') or result.get('stableIdentifier')
                        dbId = result.get('dbId') or result.get('databaseIdentifier')
                        
                        print(f"      Type: {type_name}, Name: {name}, stId: {stId}, dbId: {dbId}")
        
        # Test 3: Try the interactors endpoint
        print("\n\n3. Testing interactors endpoint:")
        interactors_url = "https://reactome.org/ContentService/interactors/static/molecules/P04637/details"
        
        async with session.get(interactors_url) as response:
            print(f"  Status: {response.status}")
            if response.status == 200:
                data = await response.json()
                print(f"  Response type: {type(data)}")
                if isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())}")
        
        # Test 4: Test data/query with UniProt ID
        print("\n\n4. Testing data/query with UniProt ID P04637:")
        query_url = "https://reactome.org/ContentService/data/query/P04637"
        
        async with session.get(query_url) as response:
            print(f"  Status: {response.status}")
            if response.status == 200:
                data = await response.json()
                if isinstance(data, list):
                    print(f"  Found {len(data)} results")
                    for item in data[:3]:
                        print(f"    {item.get('displayName')} ({item.get('className')})")
                elif isinstance(data, dict):
                    print(f"  Single result: {data.get('displayName')} ({data.get('className')})")

if __name__ == "__main__":
    asyncio.run(test_reactome_detailed())