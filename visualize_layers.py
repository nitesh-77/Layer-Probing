import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging  # Add logging import
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from information_level_identifier import ResidualStreamMLP, ResidualStreamHook, GSM8KDataset
from datasets import load_dataset

def visualize_layer_performances_from_files(probe_dir, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", num_examples=1000, max_new_tokens=1):
    """
    Visualize how well each layer's MLP probe predicts the next token from individually saved probe files.
    
    Args:
        probe_dir: Directory containing the saved MLP probes with format layer_mlp_probes_layer{layer_num}.pt
        model_name: Name of the model to load
        num_examples: Number of GSM8K examples to use for evaluation
        max_new_tokens: Maximum number of new tokens to generate for each example
    """
    
    # Load tokenizer and model
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Fix padding token - for attention mask warning
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use quantization with the same settings
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    # Load GSM8K dataset
    print(f"Loading GSM8K dataset...")
    gsm8k_test = load_dataset("gsm8k", "main")["test"]
    
    # Select a subset of examples
    examples = []
    sample_indices = torch.randperm(len(gsm8k_test))[:num_examples].tolist()
    
    for idx in sample_indices:
        example = gsm8k_test[idx]
        question = example["question"]
        examples.append(question)
    
    print(f"Selected {len(examples)} examples from GSM8K test set")
    for i, example in enumerate(examples):
        print(f"  Example {i+1}: {example[:80]}...")
    
    # Determine the model's dtype for consistency
    model_dtype = next(model.parameters()).dtype
    print(f"Model dtype: {model_dtype}")
    
    # Find all layer probe files
    probe_files = []
    for file in os.listdir(probe_dir):
        if file.startswith("layer_mlp_probes_layer") and file.endswith(".pt"):
            layer_num = int(file.split("layer")[-1].split(".pt")[0])
            probe_files.append((layer_num, os.path.join(probe_dir, file)))
    
    probe_files.sort()  # Sort by layer number
    print(f"Found {len(probe_files)} layer probe files")
    
    # Initialize hooks
    hooks = []
    hook_handles = []
    for i, layer in enumerate(model.model.layers):
        hook = ResidualStreamHook()
        handle = layer.register_forward_hook(hook)
        hooks.append((hook, handle))
    
    # Precompute all example residual streams and targets to avoid redundant computation
    print("Pre-processing examples...")
    example_data = []
    
    for example_idx, example in enumerate(examples):
        print(f"  Processing example {example_idx+1}/{len(examples)}: {example[:30]}...")
        # Format as instruction for GSM8K
        prompt = f"<|im_start|>user\nSolve this math problem step by step:\n{example}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize - keep it simple
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate one token to capture residual streams and get target
        with torch.no_grad():
            # Forward pass to generate token and capture residual streams
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id  # Explicitly set pad_token_id to suppress warnings
            )
            
            # Get the generated token (the target)
            target_token = outputs.sequences[0, -1].unsqueeze(0)
            
            # Get the model's final logits for this token
            # Extract the final logits from the last generation step
            final_logits = outputs.scores[0]  # Get logits for the first generated token
            
            # Collect residual streams for this example
            example_residual_streams = {}
            for i, (hook, _) in enumerate(hooks):
                for module_id, tensor in hook.residual_stream.items():
                    # Take the last position's hidden state
                    last_position = tensor[:, -1, :].detach()
                    example_residual_streams[i] = last_position
                hook.residual_stream = {}  # Clear the hook
            
            example_data.append({
                'residual_streams': example_residual_streams,
                'target_token': target_token,
                'final_logits': final_logits  # Store the model's final logits
            })
    
    # Clean up hooks after all examples
    for _, handle in hooks:
        handle.remove()
    
    # Store metrics per layer
    layer_metrics = {}
    
    # Process one layer probe at a time
    for probe_idx, (layer_idx, probe_path) in enumerate(probe_files):
        print(f"Processing layer {layer_idx} probe ({probe_idx+1}/{len(probe_files)})...")
        
        try:
            # Load the state dict first to check dtype
            state_dict = torch.load(probe_path, map_location="cpu")
            first_key = list(state_dict.keys())[0]
            saved_dtype = state_dict[first_key].dtype
            print(f"  Probe saved with dtype: {saved_dtype}")
            
            # Create the MLP probe
            probe = ResidualStreamMLP(
                model.config.hidden_size,
                model.config.vocab_size
            ).to(device=model.device, dtype=saved_dtype)
            
            # Load weights
            probe.load_state_dict(state_dict)
            probe.eval()
            
            # Process all examples with this probe
            layer_results = []
            
            for example_idx, example_datum in enumerate(example_data):
                # Get residual stream for this layer
                if layer_idx in example_datum['residual_streams']:
                    # Get hidden state and target
                    hidden = example_datum['residual_streams'][layer_idx]
                    target = example_datum['target_token']
                    final_logits = example_datum['final_logits']
                    
                    # Ensure correct dtype and format - use model.device instead of probe.device
                    hidden = hidden.to(device=model.device, dtype=saved_dtype)
                    target = target.to(device=model.device, dtype=torch.long)
                    final_logits = final_logits.to(device=model.device)
                    
                    # Add batch dimension if needed
                    if len(hidden.shape) == 1:
                        hidden = hidden.unsqueeze(0)
                        
                    # Add sequence length dimension if needed
                    if len(hidden.shape) == 2:
                        hidden = hidden.unsqueeze(1)
                        
                    # Make prediction with the probe
                    with torch.no_grad():
                        logits = probe(hidden).squeeze(1)  # Remove sequence dimension
                        
                        # Calculate accuracy (comparison to final layer prediction instead of ground truth)
                        pred_token = torch.argmax(logits, dim=-1)
                        final_pred_token = torch.argmax(final_logits, dim=-1)
                        correct = (pred_token == final_pred_token).float().mean().item()
                        
                        # Calculate cross-entropy loss (comparison to ground truth)
                        ce_loss = torch.nn.functional.cross_entropy(
                            logits, 
                            target
                        ).item()
                        
                        # Calculate KL divergence loss (comparison to final layer)
                        log_probs_predicted = torch.log_softmax(logits, dim=-1)
                        log_probs_target = torch.log_softmax(final_logits, dim=-1)
                        probs_target = torch.softmax(final_logits, dim=-1)
                        
                        kl_loss = torch.nn.functional.kl_div(
                            log_probs_predicted,
                            probs_target,
                            reduction='batchmean',
                            log_target=False
                        ).item()
                        
                        layer_results.append({
                            'accuracy': correct,
                            'ce_loss': ce_loss,
                            'kl_loss': kl_loss
                        })
            
            # Calculate average metrics
            if layer_results:
                avg_accuracy = sum(r['accuracy'] for r in layer_results) / len(layer_results)
                avg_ce_loss = sum(r['ce_loss'] for r in layer_results) / len(layer_results)
                avg_kl_loss = sum(r['kl_loss'] for r in layer_results) / len(layer_results)
                
                layer_metrics[layer_idx] = {
                    'accuracy': avg_accuracy,
                    'ce_loss': avg_ce_loss,
                    'kl_loss': avg_kl_loss
                }
                
                print(f"  Layer {layer_idx}: Accuracy = {avg_accuracy:.4f}, CE Loss = {avg_ce_loss:.4f}, KL Loss = {avg_kl_loss:.4f}")
            else:
                print(f"  Layer {layer_idx}: No valid examples processed")
                layer_metrics[layer_idx] = {
                    'accuracy': 0.0,
                    'ce_loss': float('inf'),
                    'kl_loss': float('inf')
                }
        
        except Exception as e:
            print(f"  Error processing layer {layer_idx}: {str(e)}")
            layer_metrics[layer_idx] = {
                'accuracy': 0.0,
                'ce_loss': float('inf'),
                'kl_loss': float('inf')
            }
        
        # Clean up
        if 'probe' in locals():
            del probe
        if 'state_dict' in locals():
            del state_dict
        torch.cuda.empty_cache()
    
    # Prepare for plotting
    sorted_layers = sorted(layer_metrics.keys())
    accuracies = [layer_metrics[layer]['accuracy'] for layer in sorted_layers]
    ce_losses = [layer_metrics[layer]['ce_loss'] for layer in sorted_layers]
    kl_losses = [layer_metrics[layer]['kl_loss'] for layer in sorted_layers]
    
    # Plotting
    plt.figure(figsize=(15, 15))
    
    # Plot accuracies
    plt.subplot(3, 1, 1)
    plt.plot(sorted_layers, accuracies, marker='o')
    plt.title('Average Accuracy (vs Final Layer Prediction) by Layer', fontsize=14)
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True)
    plt.ylim(0, 1.05)
    
    # Plot cross-entropy losses (against ground truth)
    plt.subplot(3, 1, 2)
    plt.plot(sorted_layers, ce_losses, marker='o', color='r')
    plt.title('Cross-Entropy Loss (vs Ground Truth) by Layer', fontsize=14)
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('CE Loss', fontsize=12)
    plt.grid(True)
    
    # Plot KL divergence losses (against final layer)
    plt.subplot(3, 1, 3)
    plt.plot(sorted_layers, kl_losses, marker='o', color='g')
    plt.title('KL Divergence Loss (vs Final Layer) by Layer', fontsize=14)
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('KL Loss', fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("r1_layer_information_analysis_with_kl.png", dpi=300)
    plt.close()
    print("Visualization saved as 'layer_information_analysis_with_kl.png'")
    
    return accuracies, ce_losses, kl_losses

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize layer-wise probe performances")
    parser.add_argument("--probe-dir", type=str, default="./", help="Directory containing the probe files")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", help="Model to use")
    parser.add_argument("--num-examples", type=int, default=1000, help="Number of GSM8K examples to use")
    parser.add_argument("--output", type=str, default="layer_information_analysis.png", help="Output image filename")
    parser.add_argument("--max-new-tokens", type=int, default=1, help="Maximum number of new tokens for each example")
    args = parser.parse_args()
    
    print(f"Using model: {args.model}")
    print(f"Using {args.num_examples} examples from GSM8K")
    
    # Run visualization
    accs, ce_losses, kl_losses = visualize_layer_performances_from_files(
        args.probe_dir,
        model_name=args.model,
        num_examples=args.num_examples,
        max_new_tokens=args.max_new_tokens
    )
    
    # Print out layer metrics for easier analysis
    print("\nLayer metrics:")
    for i, (acc, ce, kl) in enumerate(zip(accs, ce_losses, kl_losses)):
        print(f"Layer {i}: Acc={acc:.4f}, CE Loss={ce:.4f}, KL Loss={kl:.4f}")
    
if __name__ == "__main__":
    main() 