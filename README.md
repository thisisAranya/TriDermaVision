# TriDermaVision

<p align="center">
  <img src="https://via.placeholder.com/800x200?text=TriDermaVision" alt="TriDermaVision Logo" width="800"/>
</p>

## Advanced Multimodal AI for Dermatological Analysis

TriDermaVision is a comprehensive multimodal system that combines three powerful AI technologies for accurate dermatological image analysis and medical recommendations:

- **Retrieval-Augmented Generation (RAG)**: Enhanced responses using structured dermatological knowledge
- **Vision Transformers (DINOv2)**: State-of-the-art image classification for skin conditions
- **Large Language and Vision Assistant (LLaVA)**: Specialized vision-language model fine-tuned for dermatology

This repository provides an end-to-end implementation for analyzing skin conditions, delivering accurate classifications, and providing medically relevant information based on visual input.

---

## Features

- **DINOv2 Image Classification**: Classifies skin conditions into 8 categories
- **LLaVA Model Integration**: Vision-language model fine-tuned for dermatological analysis
- **Knowledge Graph**: Structured medical knowledge for enhanced responses
- **Retrieval-Augmented Generation**: Combines vision, language, and knowledge graph data

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/rag-dino-llava.git
cd rag-dino-llava
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models and datasets:
```bash
python scripts/download_models.py
python scripts/download_datasets.py
```

## Usage

### Basic Inference

```python
from src.rag_system import rag_query
from PIL import Image

# Load an image
image_path = "path/to/your/skin/image.jpg"
query = "What is the treatment of the disease?"

# Get AI response
response = rag_query(query, image_path)
print("Answer:", response)
```

### Available Skin Conditions

The model can classify and analyze the following skin conditions:
- Actinic Keratosis
- Basal Cell Carcinoma
- Dermatitis
- Lichen Planus
- Melanoma
- Psoriasis
- Rosacea
- Seborrheic Keratosis

## How It Works

1. **Image Classification**: The DINOv2 vision transformer first analyzes and classifies the skin image
2. **Knowledge Retrieval**: Based on the classification, relevant information is retrieved from a dermatological knowledge graph
3. **LLaVA Processing**: The multimodal LLaVA model integrates the image, classification, and knowledge to generate comprehensive answers
4. **Response Generation**: The system produces detailed medical information and recommendations

## Project Structure

- `src/`: Core source code modules
- `scripts/`: Utility scripts for setup and inference
- `data/`: Storage location for datasets
- `models/`: Storage location for model weights

## Citations and References

If you use this code, please cite the following works:

```
@article{DINOv2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Bojanowski, Piotr and Materzyńska, Joanna and Joulin, Armand and Misra, Ishan},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}

@article{LLaVA,
  title={Visual Instruction Tuning},
  author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
  journal={arXiv preprint arXiv:2304.08485},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
