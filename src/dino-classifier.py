import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class DinoVisionTransformerClassifier(nn.Module):
    """
    Vision Transformer Classifier based on DINOv2 for skin disease classification.
    """
    def __init__(self, num_classes=8):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x

def load_dino_model(model_path):
    """
    Load a pre-trained DINO vision transformer model.
    
    Args:
        model_path (str): Path to the model weights
        
    Returns:
        model: Loaded model
    """
    model = DinoVisionTransformerClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    """
    Preprocess an image for the DINO model.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    preprocess = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
    ])
    return preprocess(image).unsqueeze(0)

def predict_skin_disease(model, image_path, device='cpu'):
    """
    Predict skin disease from an image.
    
    Args:
        model: DINO classifier model
        image_path (str): Path to the image
        device (str): Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        tuple: (probability, predicted_disease)
    """
    # List of skin disease labels
    disease_labels = [
        'actinic keratosis',
        'basal cell carcinoma',
        'dermatitis',
        'lichen planus',
        'melanoma',
        'psoriasis',
        'rosacea',
        'seborrheic keratosis'
    ]
    
    image = Image.open(image_path)
    input_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = F.softmax(output, dim=1)
    predicted_index = torch.argmax(probabilities, dim=1).item()

    predicted_probability = probabilities[0, predicted_index].item()
    predicted_disease = disease_labels[predicted_index]

    return predicted_probability, predicted_disease
