import os
import torch
from PIL import Image

def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_device():
    """
    Get the available device.
    
    Returns:
        str: 'cuda' if available, otherwise 'cpu'
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(image_path):
    """
    Load an image from path.
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        PIL.Image: Loaded image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    return Image.open(image_path).convert('RGB')
    
def get_skin_disease_labels():
    """
    Get the list of skin disease labels.
    
    Returns:
        list: List of disease labels
    """
    return [
        'actinic keratosis',
        'basal cell carcinoma',
        'dermatitis',
        'lichen planus',
        'melanoma',
        'psoriasis',
        'rosacea',
        'seborrheic keratosis'
    ]
