import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import math

class ResidualStreamHook:
    """Hook to extract residual streams from transformer model layers."""

    def __init__(self):
        self.residual_stream = {}

    def __call__(self, module, input_tensors, output_tensors):
        # Store the output of the module
        # This captures the residual stream after this layer
        self.residual_stream[id(module)] = output_tensors[0].detach().clone()


def extract_residual_stream(
    model_name: str = "Qwen/Qwen2-7B-Instruct",
    prompt: str = "Tell me about quantum computing",
    max_layers: Optional[int] = None
) -> Tuple[Dict[int, torch.Tensor], List[str], Optional[torch.Tensor], Optional[AutoTokenizer]]:
    """
    Extract the residual stream from a transformer model.

    Args:
        model_name: HuggingFace model name
        prompt: Input text prompt
        max_layers: Maximum number of layers to extract from (None for all)

    Returns:
        Tuple of (residual_stream_dict, layer_names, logprobs, tokenizer)
    """
    print(f"Loading model {model_name}...")

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

    # Debug: Print model architecture information
    print(f"Model architecture: {model.__class__.__name__}")
    
    # Get number of layers - different models might store this differently
    if hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
        print(f"Model config reports {num_layers} hidden layers")
    elif hasattr(model.config, 'n_layer'):
        num_layers = model.config.n_layer
        print(f"Model config reports {num_layers} layers (n_layer)")
    else:
        # Attempt to infer from model structure
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
            print(f"Inferred {num_layers} layers from model.model.layers")
        else:
            print("WARNING: Could not determine number of layers from config. Defaulting to 32.")
            num_layers = 32
            
    if max_layers is not None:
        num_layers = min(num_layers, max_layers)

    print(f"Model loaded. Will extract residual stream from {num_layers} layers...")

    # Register hooks for each transformer layer
    hooks = []
    hook_handles = []
    layer_names = []

    # Debug: Look at model structure
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        print(f"Actual number of layers in model.model.layers: {len(model.model.layers)}")
        layer_access_path = 'model.model.layers'
    else:
        print("WARNING: Could not find model.model.layers.")
        # Attempt to find alternative paths for different model architectures
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            print(f"Found alternative structure: model.transformer.h with {len(model.transformer.h)} layers")
            layer_access_path = 'model.transformer.h'
        else:
            raise ValueError("Could not determine the correct layer access path for this model")
    
    # Add hooks to all transformer layers
    for i in range(num_layers):
        hook = ResidualStreamHook()
        hooks.append(hook)
        
        # Access the model's transformer blocks using the determined path
        if layer_access_path == 'model.model.layers':
            if i < len(model.model.layers):
                transformer_layer = model.model.layers[i]
            else:
                print(f"WARNING: Layer index {i} is out of bounds. Model only has {len(model.model.layers)} layers.")
                continue
        elif layer_access_path == 'model.transformer.h':
            transformer_layer = model.transformer.h[i]
            
        # Store the name of the layer
        layer_names.append(f"Layer {i}")
        
        # Register the hook to capture output
        handle = transformer_layer.register_forward_hook(hook)
        hook_handles.append(handle)
        print(f"Added hook for layer {i}")

    # Process the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Forward pass
    with torch.no_grad():
        # Run model forward to get logits
        outputs = model.forward(inputs.input_ids)
        
        # Get the logits for just the last token
        logits = outputs.logits[:, -1, :]
        
        # Convert to log probabilities
        logprobs = torch.log_softmax(logits, dim=-1)
        
        # Run generate to extract residual streams via hooks
        _ = model.generate(
            inputs.input_ids,
            max_new_tokens=1,
            return_dict_in_generate=True,
        )

    # Collect residual streams from all hooks
    residual_streams = {}
    print(f"Collected hooks from {len(hooks)} layers")
    for i, hook in enumerate(hooks):
        print(f"Hook {i} captured data from {len(hook.residual_stream)} modules")
        for module_id, tensor in hook.residual_stream.items():
            residual_streams[i] = tensor
            print(f"  - Layer {i} tensor shape: {tensor.shape}")

    print(f"Total layers with residual streams: {len(residual_streams)}")
    
    # Remove hooks
    for handle in hook_handles:
        handle.remove()

    print("Residual stream extraction complete.")
    return residual_streams, layer_names, logprobs, tokenizer


def plot_token_logprobs(
    logprobs_dict: Dict[str, torch.Tensor],
    tokenizers_dict: Dict[str, AutoTokenizer],
    prompt: str,
) -> None:
    """
    Plot the logprobs of tokens as a bar graph for multiple models.

    Args:
        logprobs_dict: Dictionary mapping model names to their logprob tensors
        tokenizers_dict: Dictionary mapping model names to their tokenizers
        prompt: The input prompt used
    """
    plt.figure(figsize=(15, 10))
    
    # Truncate prompt if it's too long (show first 50 chars + "...")
    truncated_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    num_models = len(logprobs_dict)
    
    for idx, (model_name, logprobs) in enumerate(logprobs_dict.items()):
        if logprobs is None:
            print(f"Warning: No logprobs available for model {model_name}")
            continue
            
        tokenizer = tokenizers_dict[model_name]
        token_logprobs = logprobs[0].cpu().numpy()
        token_probs = np.exp(token_logprobs)
        
        # Get the top N tokens with highest probabilities
        N = 50  # Number of top tokens to show
        top_indices = np.argsort(token_probs)[-N:][::-1]  # Sort in descending order
        top_probs = token_probs[top_indices]
        
        # Get token texts for the top predictions
        top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
        
        ax = plt.subplot(num_models, 1, idx + 1)
        bars = plt.bar(range(len(top_probs)), top_probs, 
                color=colors[idx % len(colors)], alpha=0.7, width=0.8)
        
        # Add token text and probability for top tokens
        for i, (prob, token) in enumerate(zip(top_probs, top_tokens)):
            if i < 5:  # Show text for top 5 tokens
                label = f"{token} ({prob:.3f})"
                plt.text(i, prob, label, ha='center', va='bottom', fontsize=8, rotation=45)
        
        cumulative_prob = np.sum(top_probs)
        
        plt.title(f"{model_name}\nPrompt: \"{truncated_prompt}\"\n(Top {N} tokens, cumulative prob: {cumulative_prob:.4f})", fontsize=12)
        plt.ylabel('Probability', fontsize=10)
        if idx == len(logprobs_dict) - 1:
            plt.xlabel('Top Tokens (ranked by probability)', fontsize=10)
        
        plt.ylim(0, max(top_probs) * 1.2)  # Increased y-limit to accommodate token text
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.xticks(range(0, len(top_probs), 5))
    
    plt.tight_layout()
    plt.savefig('token_logprobs_distribution.png', dpi=300)
    plt.close()
    print("Plot saved as 'token_logprobs_distribution.png'")


def plot_residual_stream_entropy(
    residual_streams_dict: Dict[str, Dict[int, torch.Tensor]],
    layer_names_dict: Dict[str, List[str]],
    logprobs_dict: Dict[str, torch.Tensor],
    prompt: str,
    last_token_only: bool = False,
) -> None:
    """
    Plot the entropy of residual streams for multiple models as a line graph.
    Entropy is calculated as the Frobenius norm of q*q_T (outer product)
    across all tokens in the residual stream.

    Args:
        residual_streams_dict: Dictionary mapping model names to their residual stream dictionaries
        layer_names_dict: Dictionary mapping model names to their layer names lists
        logprobs_dict: Dictionary mapping model names to their logprob tensors
        prompt: The input prompt used
        last_token_only: If True, only calculate entropy for the last token
    """
    plt.figure(figsize=(12, 7))
    
    # Truncate prompt if it's too long (show first 50 chars + "...")
    truncated_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'x']
    
    max_layers = 0  # Track max number of layers for x-axis
    
    for idx, (model_name, residual_streams) in enumerate(residual_streams_dict.items()):
        layer_names = layer_names_dict[model_name]
        entropies = []

        # Calculate the entropy for each layer's residual stream
        for i in range(len(layer_names)):
            if i in residual_streams:
                tensor = residual_streams[i].cpu()
                batch_size, seq_len, hidden_dim = tensor.shape
                
                for batch_idx in range(batch_size):
                    if last_token_only:
                        q = tensor[batch_idx, -1:, :]
                    else:
                        q = tensor[batch_idx]
                    
                    q = q.to(torch.float64)
                    q = torch.cov(q)
                    x = q.norm(p='fro').item()
                    entropies.append(np.log(x))
            else:
                entropies.append(0)
        
        # Calculate entropy of logprobs
        if model_name in logprobs_dict:
            logprobs = logprobs_dict[model_name][0].cpu()  # Shape: [vocab_size]
            probs = torch.exp(logprobs)
            # Calculate entropy: -sum(p * log(p))
            logprob_entropy = -torch.sum(probs * logprobs).item()
            entropies.append(logprob_entropy)
        
        max_layers = max(max_layers, len(entropies))
        
        # Plot as a line with markers
        x_points = list(range(len(entropies) - 1)) + ['logits']
        plt.plot(range(len(entropies)), entropies, label=model_name, 
                 color=colors[idx % len(colors)], marker=markers[idx % len(markers)], 
                 linewidth=2, markersize=6)

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Log Entropy', fontsize=12)
    title_prefix = 'Last Token' if last_token_only else 'All Tokens'
    plt.title(f'{title_prefix} Residual Stream Entropy by Layer\nPrompt: "{truncated_prompt}"', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Customize x-axis ticks to show "logits" for the last point
    x_ticks = list(range(max_layers - 1)) + ['logits']
    plt.xticks(range(max_layers), x_ticks, rotation=45)
    
    # Add some padding to the x-axis
    plt.xlim(-0.5, max_layers - 0.5)
    
    plt.tight_layout()

    # Save the plot
    suffix = 'last_token' if last_token_only else 'all_tokens'
    filename = f'residual_stream_entropy_{suffix}.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved as '{filename}'")


def main():
    # Define the prompt
    prompt = """the dog the dog the dog the dog the dog the dog the dog the dog the dog the dog the dog the dog the dog the dog the dog the dog """
    
    # Define models to compare
    models = [
        "Qwen/Qwen2-7B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    ]
    
    # Dictionary to store residual streams, layer names, logprobs, and tokenizers
    residual_streams_dict = {}
    layer_names_dict = {}
    logprobs_dict = {}
    tokenizers_dict = {}
    
    # Extract residual streams for each model
    for model_name in models:
        print(f"\nProcessing model: {model_name}")
        residual_streams, layer_names, logprobs, tokenizer = extract_residual_stream(
            model_name=model_name,
            prompt=prompt,
            max_layers=None
        )
        
        residual_streams_dict[model_name] = residual_streams
        layer_names_dict[model_name] = layer_names
        logprobs_dict[model_name] = logprobs
        tokenizers_dict[model_name] = tokenizer

    # Plot the comparison of residual stream entropies (all tokens)
    plot_residual_stream_entropy(residual_streams_dict, layer_names_dict, logprobs_dict, prompt, last_token_only=False)
    
    # Plot the comparison of residual stream entropies (last token only)
    plot_residual_stream_entropy(residual_streams_dict, layer_names_dict, logprobs_dict, prompt, last_token_only=True)
    
    # Plot the token logprobs distribution
    plot_token_logprobs(logprobs_dict, tokenizers_dict, prompt)

    print(f"\nCompleted analysis for prompt: '{prompt}'")


if __name__ == "__main__":
    main()
