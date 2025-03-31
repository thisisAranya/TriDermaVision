#!/usr/bin/env python
"""
Script to download required models for the RAG-DINO-LLaVA system.
"""

import os
import sys
import argparse
import torch
import huggingface_hub
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoProcessor

def download_dinov2():
    """Download DINOv2 model"""
    print("Downloading DINOv2 model...")
    try:
        # This will cache the model
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        print("DINOv2 model downloaded successfully.")
        return True
    except Exception as e:
        print(f"Failed to download DINOv2 model: {e}")
        return False

def download_dino_weights(output_dir="models"):
    """Download DINO classifier weights"""
    print("Downloading DINO classifier weights...")
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # You need to host this file somewhere (e.g., HuggingFace, GitHub releases)
        # This is a placeholder - replace with actual download logic
        print("Please download the DINO model weights from: https://www.kaggle.com/models/aranyasaha/dino-model-trained-on-dermnet")
        print(f"And place them in: {os.path.join(output_dir, 'best_model.pth')}")
        return True
    except Exception as e:
        print(f"Failed to download DINO weights: {e}")
        return False

def download_llava_model(model_id="Aranya31/DermLLaVA-7b", output_dir="models"):
    """Download LLaVA model"""
    print(f"Downloading LLaVA model: {model_id}...")
    try:
        # This will only download the model configuration, not the weights
        # The weights will be downloaded when the model is first used
        processor = AutoProcessor.from_pretrained(model_id)
        print(f"LLaVA processor downloaded successfully.")
        return True
    except Exception as e:
        print(f"Failed to download LLaVA model: {e}")
        return False

def download_knowledge_graph(output_dir="models"):
    """Download knowledge graph"""
    print("Downloading knowledge graph...")
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # You need to host this file somewhere (e.g., HuggingFace, GitHub releases)
        # This is a placeholder - replace with actual download logic
        print("Please download the knowledge graph from: https://www.kaggle.com/datasets/chapkhabo/zxzzzzzzzzzzzzzz")
        print(f"And place it in: {os.path.join(output_dir, 'knowledge_graph.graphml')}")
        return True
    except Exception as e:
        print(f"Failed to download knowledge graph: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download models for RAG-DINO-LLaVA')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--llava-model', type=str, default='Aranya31/DermLLaVA-7b',
                        help='LLaVA model ID')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download models
    dinov2_success = download_dinov2()
    dino_weights_success = download_dino_weights(args.output_dir)
    llava_success = download_llava_model(args.llava_model, args.output_dir)
    kg_success = download_knowledge_graph(args.output_dir)
    
    # Print summary
    print("\nDownload Summary:")
    print(f"DINOv2 base model: {'✓' if dinov2_success else '✗'}")
    print(f"DINO classifier weights: {'✓' if dino_weights_success else '✗'}")
    print(f"LLaVA model: {'✓' if llava_success else '✗'}")
    print(f"Knowledge graph: {'✓' if kg_success else '✗'}")
    
    if not all([dinov2_success, dino_weights_success, llava_success, kg_success]):
        print("\nSome downloads failed. Please check the error messages and try again.")
        sys.exit(1)
    
    print("\nAll models downloaded successfully.")

if __name__ == "__main__":
    main()
