#!/usr/bin/env python
"""
Script to download required datasets for the RAG-DINO-LLaVA system.
"""

import os
import sys
import argparse
import zipfile
import shutil
from huggingface_hub import hf_hub_download

def download_dermnet_dataset(output_dir="data"):
    """Download DermNet dataset"""
    print("Downloading DermNet dataset...")
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # You need to host this file somewhere (e.g., HuggingFace, GitHub, or Kaggle)
        # This is a placeholder - replace with actual download logic
        print("Please download the selective DermNet dataset from: https://www.kaggle.com/datasets/aranyasaha/selective-dermnet-for-llm")
        print(f"And extract it to: {os.path.join(output_dir, 'dermnet')}")
        return True
    except Exception as e:
        print(f"Failed to download DermNet dataset: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download datasets for RAG-DINO-LLaVA')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Directory to save datasets')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download datasets
    dermnet_success = download_dermnet_dataset(args.output_dir)
    
    # Print summary
    print("\nDownload Summary:")
    print(f"DermNet dataset: {'✓' if dermnet_success else '✗'}")
    
    if not all([dermnet_success]):
        print("\nSome downloads failed. Please check the error messages and try again.")
        sys.exit(1)
    
    print("\nAll datasets downloaded successfully.")

if __name__ == "__main__":
    main()
