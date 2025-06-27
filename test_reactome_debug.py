"""
Debug Reactome API to find the exact working endpoint
"""

import asyncio
import aiohttp
import json

async def debug_reactome():
    """Debug Reactome API with different approaches"""
    
    print("Debugging Reactome API...")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Basic search
        print("\n1. Testing basic search for TP53:")
        search_url = "https://reactome.org/ContentService/search/query?query=TP53&species=Homo sapiens"
        
        async with session.get(search_url) as response:
            print(f"Status: {response.status}")
            if response.status == 200:
                data = await response.json()
                print(f"Found {len(data.get('results', []))} results")
                
                # Show first few results
                for i, result in enumerate(data.get('results', [])[:5]):
                    print(f"\nResult {i+1}:")
                    print(f"  Type: {result.get('typeName')}")
                    print(f"  Name: {result.get('name')}")
                    print(f"  stId: {result.get('stId')}")
                    print(f"  dbId: {result.get('dbId')}")
                    print(f"  Species: {result.get('species', [{}])[0].get('displayName', 'N/A') if result.get('species') else 'N/A'}")
        
        # Test 2: Get entity details for a known TP53 entity
        print("\n\n2. Testing entity query for R-HSA-69541 (TP53 entity):")
        entity_url = "https://reactome.org/ContentService/data/query/R-HSA-69541"
        
        async with session.get(entity_url) as response:
            print(f"Status: {response.status}")
            if response.status == 200:
                data = await response.json()
                print(f"Entity type: {data.get('className')}")
                print(f"Display name: {data.get('displayName')}")
        
        # Test 3: Get pathways for known entity
        print("\n\n3. Testing pathway retrieval for TP53 (using dbId):")
        # First, get a valid dbId from search
        search_url = "https://reactome.org/ContentService/search/query?query=TP53&species=Homo sapiens"
        
        async with session.get(search_url) as response:
            if response.status == 200:
                data = await response.json()
                
                # Find a protein/gene result
                tp53_id = None
                for result in data.get('results', []):
                    if result.get('typeName') in ['ReferenceGeneProduct', 'Protein', 'EntityWithAccessionedSequence']:
                        if 'TP53' in result.get('name', ''):
                            tp53_id = result.get('stId') or str(result.get('dbId'))
                            print(f"Found TP53 entity: {tp53_id} ({result.get('typeName')})")
                            break
                
                if tp53_id:
                    # Try different pathway endpoints
                    endpoints = [
                        f"https://reactome.org/ContentService/data/pathways/low/entity/{tp53_id}?species=9606",
                        f"https://reactome.org/ContentService/data/pathways/low/entity/{tp53_id}",
                        f"https://reactome.org/ContentService/data/entity/{tp53_id}/containedEvents"
                    ]
                    
                    for endpoint in endpoints:
                        print(f"\n  Trying: {endpoint}")
                        async with session.get(endpoint) as response:
                            print(f"  Status: {response.status}")
                            if response.status == 200:
                                data = await response.json()
                                if isinstance(data, list):
                                    print(f"  Success! Found {len(data)} pathways")
                                    if data:
                                        print(f"  First pathway: {data[0].get('displayName', 'N/A')}")
                                        break
                                else:
                                    print(f"  Response type: {type(data)}")
        
        # Test 4: Alternative approach - use the interactors endpoint
        print("\n\n4. Testing alternative approach - query endpoint:")
        query_url = "https://reactome.org/ContentService/data/query/TP53"
        
        async with session.get(query_url) as response:
            print(f"Status: {response.status}")
            if response.status == 200:
                data = await response.json()
                print(f"Response type: {type(data)}")
                if isinstance(data, dict):
                    print(f"Class: {data.get('className')}")
                    print(f"Display name: {data.get('displayName')}")
                    print(f"stId: {data.get('stId')}")
        
        # Test 5: Try the entities/findByIds endpoint
        print("\n\n5. Testing entities/findByIds endpoint:")
        url = "https://reactome.org/ContentService/data/entities/findByIds"
        headers = {'Content-Type': 'text/plain', 'Accept': 'application/json'}
        data = "TP53\nBRCA1"
        
        async with session.post(url, data=data, headers=headers) as response:
            print(f"Status: {response.status}")
            if response.status == 200:
                result = await response.json()
                print(f"Found {len(result)} results")
                for key, value in result.items():
                    print(f"  {key}: {value.get('displayName', 'N/A')} ({value.get('className', 'N/A')})")

if __name__ == "__main__":
    asyncio.run(debug_reactome())