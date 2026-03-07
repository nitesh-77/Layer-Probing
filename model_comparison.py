#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from typing import List, Dict

def generate_completion(
    model_name: str,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate a completion for the given prompt using the specified model.
    
    Args:
        model_name: HuggingFace model name
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Top-p sampling parameter
        
    Returns:
        Generated text completion
    """
    print(f"\nLoading model {model_name}...")

    # Set up device - prioritize CUDA, then MPS, finally CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for GPU acceleration")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("No GPU acceleration available, using CPU")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 to save memory
        device_map=device  # Use specific device instead of "auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # If no EOS token, use a common choice
            tokenizer.pad_token = '</s>'
            
    # Use prompt as-is without any formatting
    formatted_prompt = prompt

    # Tokenize the prompt with padding and attention mask
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True
    ).to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,  # Use sampling if temperature > 0
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new content (the completion)
    completion = generated_text[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
    
    # Clean up resources
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return completion.strip(), generated_text

def compare_models(
    prompt: str,
    models: List[str] = ["Qwen/Qwen2-7B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"],
    max_new_tokens: int = 100,
    temperature: float = 0.7,
) -> Dict[str, str]:
    """
    Compare completions from multiple models for the same prompt.
    
    Args:
        prompt: Input text prompt
        models: List of model names to compare
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Dictionary mapping model names to their completions
    """
    results = {}
    
    for model_name in models:
        print(f"\nGenerating completion with {model_name}...")
        completion, full_text = generate_completion(
            model_name=model_name,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        results[model_name] = {
            "completion": completion,
            "full_text": full_text
        }
        
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compare completions from multiple language models")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt for text generation")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0 = deterministic)")
    args = parser.parse_args()
    
    # Get prompt from user if not provided as argument
    prompt = args.prompt
    if not prompt:
        prompt = input("Enter your prompt: ")
    
    # Define models to compare
    models = [
        "Qwen/Qwen2-7B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    ]
    
    # Compare models
    results = compare_models(
        prompt=prompt,
        models=models,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Display results
    print("\n" + "="*80)
    print("COMPLETION COMPARISON")
    print("="*80)
    
    for model_name, result in results.items():
        print(f"\n[{model_name}]")
        print("-" * 40)
        print(f"Prompt: {prompt}")
        print(f"Completion: {result['completion']}")
        print("\nFull text:")
        print("-" * 40)
        print(result['full_text'])
        print("="*80)

if __name__ == "__main__":
    main() 