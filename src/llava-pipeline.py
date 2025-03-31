import torch
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration, LlavaProcessor
from transformers import pipeline

def load_llava_model(model_id="Aranya31/DermLLaVA-7b", device_map="auto", use_4bit=True):
    """
    Load LLaVA model and processor.
    
    Args:
        model_id (str): Model ID or path
        device_map (str): Device mapping strategy
        use_4bit (bool): Whether to use 4-bit quantization
        
    Returns:
        tuple: (processor, model, pipeline)
    """
    # Load the processor
    processor = LlavaProcessor.from_pretrained(model_id)
    
    if use_4bit:
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load the model with 4-bit quantization
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map
        )
    else:
        # Load model without quantization
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device_map
        )
    
    # Create pipeline
    qa_pipeline = pipeline(
        "image-text-to-text", 
        model=model, 
        max_new_tokens=512, 
        processor=processor
    )
    
    return processor, model, qa_pipeline

def create_llava_prompt(image_path, text_prompt):
    """
    Create a prompt for the LLaVA model.
    
    Args:
        image_path (str): Path to the image
        text_prompt (str): Text prompt
        
    Returns:
        list: Formatted messages for the model
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    return messages

def query_llava(pipeline, image_path, text_prompt, max_new_tokens=512):
    """
    Query the LLaVA model.
    
    Args:
        pipeline: HuggingFace pipeline
        image_path (str): Path to the image
        text_prompt (str): Text prompt
        max_new_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: Model response
    """
    messages = create_llava_prompt(image_path, text_prompt)
    
    # Call pipeline
    response = pipeline(text=messages, max_new_tokens=max_new_tokens)
    
    # Extract generated text
    return response[0]['generated_text'][1]['content']
