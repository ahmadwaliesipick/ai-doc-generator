import os
import argparse
import pandas as pd
from extract_js import extract_functions as extract_js
from extract_php import extract_functions as extract_php

def process_directory(directory, language):
    all_results = []
    
    extensions = {
        'js': ['.js', '.mjs', '.cjs'],
        'php': ['.php']
    }
    
    extract_fn = extract_js if language == 'js' else extract_php
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions[language]):
                file_path = os.path.join(root, file)
                try:
                    results = extract_fn(file_path)
                    for r in results:
                        r['file'] = file_path
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract function-comment pairs from a directory.")
    parser.add_argument("directory", help="Directory to process")
    parser.add_argument("language", choices=['js', 'php'], help="Programming language (js or php)")
    parser.add_argument("--output", default="data/raw/extracted_pairs.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    results = process_directory(args.directory, args.language)
    df = pd.DataFrame(results)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Extracted {len(df)} pairs. Results saved to {args.output}")
