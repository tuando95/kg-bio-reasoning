"""
Test a simple working Reactome integration
"""

import asyncio
import aiohttp
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def get_reactome_pathways_simple(gene_name: str):
    """Get Reactome pathways for a gene using the simplest working approach"""
    pathways = []
    
    async with aiohttp.ClientSession() as session:
        # Step 1: Search for the gene
        search_url = f"https://reactome.org/ContentService/search/query?query={gene_name}&species=Homo sapiens"
        
        async with session.get(search_url) as response:
            if response.status == 200:
                data = await response.json()
                
                # Find protein entries
                for result in data.get('results', []):
                    if result.get('typeName') == 'Protein' and 'entries' in result:
                        for entry in result['entries']:
                            # Check if human
                            if 'Homo sapiens' in entry.get('species', []):
                                stId = entry.get('stId')
                                
                                if stId:
                                    # Get pathways
                                    pathways_url = f"https://reactome.org/ContentService/data/pathways/low/entity/{stId}?species=9606"
                                    
                                    async with session.get(pathways_url) as pathway_response:
                                        if pathway_response.status == 200:
                                            pathway_data = await pathway_response.json()
                                            
                                            for pathway in pathway_data[:5]:
                                                pathways.append({
                                                    'gene': gene_name,
                                                    'stId': pathway.get('stId'),
                                                    'name': pathway.get('displayName')
                                                })
                                            
                                            return pathways  # Return after first successful entity
    
    return pathways


async def test_simple():
    """Test the simple approach"""
    print("Testing Simple Reactome Integration")
    print("=" * 60)
    
    genes = ['TP53', 'BRCA1', 'EGFR']
    
    for gene in genes:
        print(f"\nGetting pathways for {gene}:")
        pathways = await get_reactome_pathways_simple(gene)
        
        if pathways:
            print(f"  Found {len(pathways)} pathways:")
            for p in pathways[:3]:
                print(f"    {p['stId']}: {p['name']}")
        else:
            print("  No pathways found")
        
        await asyncio.sleep(0.2)


if __name__ == "__main__":
    asyncio.run(test_simple())