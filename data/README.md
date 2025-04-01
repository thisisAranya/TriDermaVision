# Data Directory

This directory is used to store datasets for the RAG-DINO-LLaVA system.

## Required Datasets

### DermNet Dataset

The system uses a selective subset of the DermNet dataset for dermatological image analysis.

#### How to Obtain

1. Download the selective DermNet dataset from [Kaggle](https://www.kaggle.com/datasets/aranyasaha/selective-dermnet-for-llm)
2. Extract the dataset to this directory

#### Expected Structure

```
data/
└── dermnet/
    ├── train_merged_selective_resized/
    │   ├── known_1/
    │   ├── known_2/
    │   ├── ...
    ├── test_merged_selective_resized/
    │   ├── known_1/
    │   ├── known_2/
    │   ├── ...
```

## Directory Structure

Once all datasets are downloaded, this directory should have the following structure:

```
data/
├── README.md (this file)
└── dermnet/ (from selective-dermnet-for-llm dataset)
```

## Data Usage

The datasets are used for:
- Testing the DINO model's classification capabilities
- Providing example images for the LLaVA model
- Demonstrating the RAG system's functionality
