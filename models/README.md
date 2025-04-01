# Models Directory

This directory is used to store model weights and data for the RAG-DINO-LLaVA system.

## Required Models

### 1. DINO Vision Transformer Model

The DINO model is used for classifying skin conditions.

#### How to Obtain

1. Download the model weights from [Kaggle](https://www.kaggle.com/models/aranyasaha/dino-model-trained-on-dermnet)
2. Place the `best_model.pth` file in this directory

### 2. Knowledge Graph

The knowledge graph contains structured medical information for the RAG system.

#### How to Obtain

1. Download the knowledge graph from [Kaggle](https://www.kaggle.com/datasets/chapkhabo/zxzzzzzzzzzzzzzz)
2. Place the `knowledge_graph.graphml` file in this directory

### 3. LLaVA Model

The LLaVA model will be automatically downloaded from Hugging Face when first used.

## Directory Structure

Once all models are downloaded, this directory should have the following structure:

```
models/
├── README.md (this file)
├── best_model.pth (from DINO model)
└── knowledge_graph.graphml (from knowledge graph dataset)
```

## Model Usage

- `best_model.pth`: Used by the DinoVisionTransformerClassifier for skin disease classification
- `knowledge_graph.graphml`: Used by the RAG system to enhance responses with medical knowledge
- LLaVA model: Downloaded and cached by the Hugging Face Transformers library
