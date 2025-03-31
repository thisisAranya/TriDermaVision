import networkx as nx
from sentence_transformers import SentenceTransformer
import os

def load_knowledge_graph(graph_path):
    """
    Load a knowledge graph from a GraphML file.
    
    Args:
        graph_path (str): Path to the GraphML file
        
    Returns:
        nx.Graph: Loaded graph
    """
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Knowledge graph file not found at {graph_path}")
    
    return nx.read_graphml(graph_path)

def encode_graph_entities(graph, model_name="all-MiniLM-L6-v2"):
    """
    Encode graph entities using a sentence transformer.
    
    Args:
        graph (nx.Graph): Knowledge graph
        model_name (str): Name of the sentence transformer model
        
    Returns:
        dict: Dictionary mapping entity names to embeddings
    """
    # Load retrieval model
    retrieval_model = SentenceTransformer(model_name)
    
    # Encode all nodes
    entity_embeddings = {node: retrieval_model.encode(node) for node in graph.nodes}
    
    return retrieval_model, entity_embeddings

def get_entity_relations(graph, entity):
    """
    Get relations for a given entity in the graph.
    
    Args:
        graph (nx.Graph): Knowledge graph
        entity (str): Entity to get relations for
        
    Returns:
        str: Text describing the entity's relations
    """
    if entity in graph:
        related_entities = [(target, data.get('relation', 'related_to')) 
                            for target, data in graph[entity].items()]
        relation_text = " ".join([f"{entity} -({relation})-> {target}" 
                                 for target, relation in related_entities])
        return relation_text
    else:
        return "No known relations."
