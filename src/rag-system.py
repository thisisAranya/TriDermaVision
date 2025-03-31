import os
import torch
from sentence_transformers import util

from .dino_classifier import load_dino_model, predict_skin_disease
from .llava_pipeline import load_llava_model, query_llava
from .knowledge_graph import load_knowledge_graph, encode_graph_entities, get_entity_relations

class RagSystem:
    """
    Retrieval-Augmented Generation system combining DINO, LLaVA, and a knowledge graph.
    """
    def __init__(self, 
                 dino_model_path="models/best_model.pth",
                 llava_model_id="Aranya31/DermLLaVA-7b",
                 graph_path="models/knowledge_graph.graphml",
                 device=None):
        """
        Initialize the RAG system.
        
        Args:
            dino_model_path (str): Path to DINO model weights
            llava_model_id (str): LLaVA model ID or path
            graph_path (str): Path to knowledge graph
            device (str): Device to use ('cpu', 'cuda', or None for auto)
        """
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load DINO model
        self.dino_model = load_dino_model(dino_model_path)
        self.dino_model.to(self.device)
        
        # Load LLaVA model
        _, _, self.qa_pipeline = load_llava_model(llava_model_id, device_map=self.device)
        
        # Load knowledge graph
        self.graph = load_knowledge_graph(graph_path)
        
        # Initialize retrieval model and encode entities
        self.retrieval_model, self.entity_embeddings = encode_graph_entities(self.graph)
    
    def query(self, query, image_path):
        """
        Process a query with the RAG system.
        
        Args:
            query (str): User query
            image_path (str): Path to the image
            
        Returns:
            str: Generated response
        """
        # Check if image exists
        if not os.path.exists(image_path):
            return "Error: Image file not found."
            
        # 1. First identify the disease
        probability, predicted_disease = predict_skin_disease(
            self.dino_model, image_path, device=self.device
        )
        
        # If confidence is low, ask LLaVA for disease name
        if probability < 0.8:
            disease_prompt = "What is the name of the skin condition in this image?"
            disease_name = query_llava(self.qa_pipeline, image_path, disease_prompt, max_new_tokens=64)
        else:
            disease_name = predicted_disease
            
        # 2. Encode query for retrieval
        query_embedding = self.retrieval_model.encode(query)
        
        # 3. Find the best matching entity from the KG
        best_match = max(self.entity_embeddings.items(), 
                         key=lambda item: util.cos_sim(query_embedding, item[1]))
        best_entity = best_match[0]
        
        # 4. Get relations for the entity
        relation_text = get_entity_relations(self.graph, best_entity)
        
        # 5. Construct the prompt for LLaVA
        prompt = (
            f"Using knowledge about {best_entity} and its relations ({relation_text}), "
            f"answer the following question about {disease_name}: {query}"
        )
        
        # 6. Query LLaVA
        response = query_llava(self.qa_pipeline, image_path, prompt)
        
        return response

# Function wrapper for simpler usage
def rag_query(query, image_path, 
              dino_model_path="models/best_model.pth",
              llava_model_id="Aranya31/DermLLaVA-7b",
              graph_path="models/knowledge_graph.graphml"):
    """
    Simple function to query the RAG system.
    
    Args:
        query (str): User query
        image_path (str): Path to the image
        dino_model_path (str): Path to DINO model weights
        llava_model_id (str): LLaVA model ID or path
        graph_path (str): Path to knowledge graph
        
    Returns:
        str: Generated response
    """
    rag = RagSystem(dino_model_path, llava_model_id, graph_path)
    return rag.query(query, image_path)
