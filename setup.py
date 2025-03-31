from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rag-dino-llava",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A multimodal system for dermatological analysis combining RAG, DINO, and LLaVA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rag-dino-llava",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "pillow>=9.0.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "networkx>=2.6.0",
        "rdflib>=6.0.0",
        "sentence-transformers>=2.2.0",
        "huggingface-hub>=0.13.0",
        "datasets>=2.10.0",
        "bitsandbytes>=0.40.0",
        "peft>=0.4.0",
    ],
)
