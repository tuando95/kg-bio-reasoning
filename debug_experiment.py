"""
Debug script to check experiment results structure
"""

import json
import pandas as pd
from pathlib import Path

# Check if there are any results files
experiment_dir = Path("experiments")
if experiment_dir.exists():
    print("Found experiment directory")
    
    # List all result files
    result_files = list(experiment_dir.glob("**/all_results.csv"))
    json_files = list(experiment_dir.glob("**/*_results.json"))
    
    print(f"\nFound {len(result_files)} CSV result files:")
    for f in result_files:
        print(f"  - {f}")
        
    print(f"\nFound {len(json_files)} JSON result files:")
    for f in json_files:
        print(f"  - {f}")
    
    # Check the structure of results
    if result_files:
        print("\nChecking CSV structure:")
        for csv_file in result_files[:1]:  # Check first file
            df = pd.read_csv(csv_file)
            print(f"\nFile: {csv_file}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Shape: {df.shape}")
            if not df.empty:
                print("\nFirst row:")
                print(df.iloc[0].to_dict())
    
    if json_files:
        print("\nChecking JSON structure:")
        for json_file in json_files[:1]:  # Check first file
            with open(json_file, 'r') as f:
                data = json.load(f)
            print(f"\nFile: {json_file}")
            print(f"Keys: {list(data.keys())}")
            # Print a subset of the data
            for key in ['experiment_name', 'f1_macro', 'f1_micro', 'status']:
                if key in data:
                    print(f"  {key}: {data[key]}")