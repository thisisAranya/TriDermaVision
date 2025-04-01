#!/usr/bin/env python
"""
Example inference script for the RAG-DINO-LLaVA system.
"""

import os
import sys
import argparse
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_system import rag_query
from src.dino_classifier import load_dino_model, predict_skin_disease
from src.llava_pipeline import load_llava_model, query_llava
from src.utils import get_device, load_image, get_skin_disease_labels

def main():
    parser = argparse.ArgumentParser(description='Run inference with RAG-DINO-LLaVA')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the skin disease image')
    parser.add_argument('--query', type=str, default="What is the treatment for this skin condition?",
                        help='Query about the skin condition')
    parser.add_argument('--dino-model', type=str, default="models/best_model.pth",
                        help='Path to DINO model weights')
    parser.add_argument('--llava-model', type=str, default="Aranya31/DermLLaVA-7b",
                        help='LLaVA model ID')
    parser.add_argument('--graph', type=str, default="models/knowledge_graph.graphml",
                        help='Path to knowledge graph')
    parser.add_argument('--mode', type=str, choices=['full', 'classify', 'llava'], default='full',
                        help='Inference mode: full RAG system, classification only, or LLaVA only')
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        sys.exit(1)
        
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    if args.mode == 'classify':
        # Load DINO model and classify
        print("Loading DINO model...")
        dino_model = load_dino_model(args.dino_model)
        dino_model.to(device)
        
        # Predict disease
        print("Classifying image...")
        probability, predicted_disease = predict_skin_disease(dino_model, args.image, device)
        
        # Print results
        print("\nClassification Results:")
        print(f"Predicted Disease: {predicted_disease}")
        print(f"Confidence: {probability:.2%}")
        
    elif args.mode == 'llava':
        # Load LLaVA model
        print("Loading LLaVA model...")
        _, _, qa_pipeline = load_llava_model(args.llava_model, device_map=device)
        
        # Query LLaVA
        print("Querying LLaVA model...")
        response = query_llava(qa_pipeline, args.image, args.query)
        
        # Print results
        print("\nLLaVA Response:")
        print(response)
        
    else:  # full RAG system
        # Query the RAG system
        print("Running full RAG system...")
        response = rag_query(
            args.query, 
            args.image, 
            args.dino_model, 
            args.llava_model, 
            args.graph
        )
        
        # Print results
        print("\nRAG System Response:")
        print(response)

if __name__ == "__main__":
    main()
