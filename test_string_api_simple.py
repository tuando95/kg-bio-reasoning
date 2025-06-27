#!/usr/bin/env python3
"""
Simple test of STRING API with correct formatting
"""

import asyncio
import aiohttp
import json


async def test_string_api():
    """Test STRING API with different identifier formats"""
    
    test_cases = [
        {
            "name": "Gene names without prefix",
            "identifiers": "TP53%0dEGFR%0dVEGFA",
            "species": 9606
        },
        {
            "name": "With species prefix", 
            "identifiers": "9606.TP53%0d9606.EGFR%0d9606.VEGFA",
            "species": 9606
        },
        {
            "name": "Mixed case",
            "identifiers": "tp53%0dEgfr%0dVEGFA", 
            "species": 9606
        }
    ]
    
    url = "https://string-db.org/api/json/network"
    
    async with aiohttp.ClientSession() as session:
        for test in test_cases:
            print(f"\nTesting: {test['name']}")
            print(f"Identifiers: {test['identifiers']}")
            
            params = {
                'identifiers': test['identifiers'],
                'species': test['species'],
                'required_score': 400,
                'caller_identity': 'biokg_test'
            }
            
            try:
                async with session.get(url, params=params) as response:
                    print(f"Status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"Success! Got {len(data)} interactions")
                        if data:
                            first = data[0]
                            print(f"Example: {first.get('preferredName_A', 'N/A')} <-> {first.get('preferredName_B', 'N/A')}")
                    else:
                        text = await response.text()
                        print(f"Error response: {text[:200]}")
                        
            except Exception as e:
                print(f"Exception: {e}")
            
            print("-" * 50)


if __name__ == "__main__":
    asyncio.run(test_string_api())