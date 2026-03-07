import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from datasets import load_dataset
import wandb
import numpy as np
from typing import Dict, List, Optional
import os
import matplotlib.pyplot as plt
import tqdm
from residual_stream_viz import ResidualStreamHook, extract_residual_stream
import psutil
import argparse
import time

# Define an MLP probe to predict logprobs from residual streams
class ResidualStreamMLP(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size, dropout_rate=0.1, bottleneck_size=1024):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, vocab_size)
        )
    
    def forward(self, hidden_states):
        # hidden_states shape: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Reshape to process all positions - use reshape instead of view for non-contiguous tensors
        reshaped = hidden_states.reshape(-1, hidden_size)
        
        # Apply MLP
        logits = self.mlp(reshaped)
        
        # Reshape back to [batch_size, seq_len, vocab_size]
        return logits.reshape(batch_size, seq_len, -1)

class GSM8KDataset(Dataset):
    def __init__(self, split="train", max_length=128):
        self.dataset = load_dataset("gsm8k", "main")[split]
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item["question"]
        answer = item["answer"]
        
        # Format as Qwen2 instruction format
        prompt = f"<|im_start|>user\nSolve this math problem step by step:\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        
        # Tokenize
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "labels": encoded["input_ids"][0].clone()
        }

class InformationLevelIdentifier:
    def __init__(
        self, 
        model_name="Qwen/Qwen2-7B-Instruct", 
        batch_size=128,
        learning_rate=5e-4,
        num_epochs=1,
        log_steps=100,
        save_path="./layer_mlp_probes.pt",
        gradient_accumulation_steps=4,
        device=None,
        max_layers=None  # Add option to limit number of layers
    ):
        """Initialize the information level identifier."""
        # Set device
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize wandb
        
        # Config
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.log_steps = log_steps
        self.save_path = save_path
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_step = 0
        self.max_layers = max_layers  # Store max_layers
        
        # Load model and tokenizer in low-precision
        # Use quantization to reduce memory usage
        print("Loading model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Initialize model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()  # Set to evaluation mode
        
        # Save some memory by sharing embedding weights if possible
        if hasattr(self.model.config, "tie_word_embeddings") and self.model.config.tie_word_embeddings:
            print("Word embeddings are tied - this saves memory")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize hooks and MLP probes
        self.hooks = []
        self.setup_hooks()
        self.mlp_probes = self.setup_mlp_probes()
    
    def setup_hooks(self):
        # Add hooks to capture residual streams
        num_layers = self.model.config.num_hidden_layers
        
        if self.max_layers is not None:
            print(f"Limiting analysis to first {self.max_layers} layers out of {num_layers}")
            num_layers = min(num_layers, self.max_layers)
        else:
            print(f"Processing all {num_layers} layers")
            
        print("Setting up hooks...")
        for i in range(num_layers):
            # Only process every 2nd layer if memory is a concern
            # if i % 2 != 0:  # Uncomment this to process only every 2nd layer
            #    continue
                
            # Create a new hook for each layer
            hook = ResidualStreamHook()
            
            # For Qwen2 models, hook into the layers this way
            if "qwen" in self.model_name.lower():
                module = self.model.model.layers[i]
            else:
                # Fallback for other models (e.g., Llama, Mistral)
                module = self.model.model.layers[i]
                
            # Register the hook with the module
            handle = module.register_forward_hook(hook)
            self.hooks.append((hook, handle))
            
            if i % 10 == 0:
                print(f"  Added hook for layer {i}")
                
        print(f"Total hooks added: {len(self.hooks)}")
    
    def setup_mlp_probes(self):
        # Initialize one MLP per layer to predict logprobs from residual streams
        num_layers = len(self.hooks)
        hidden_size = self.model.config.hidden_size
        vocab_size = self.model.config.vocab_size
        
        print(f"Setting up {num_layers} MLP probes with hidden_size={hidden_size}, vocab_size={vocab_size}")
        
        # Use dtype consistent with model
        dtype = torch.bfloat16 if hasattr(self.model, "dtype") and self.model.dtype == torch.bfloat16 else torch.float32
        
        # Create a single probe to reduce memory footprint - we'll train sequentially
        # We'll store trained probes separately, only keeping one in memory at a time
        mlp_probe = ResidualStreamMLP(hidden_size, vocab_size, bottleneck_size=1024).to(device=self.model.device, dtype=dtype)
        
        print(f"MLP probe created directly on device: {self.model.device} with dtype: {dtype}")
        
        return mlp_probe
    
    def compute_layer_losses(self, residual_streams, shifted_labels):
        """Compute losses for each layer and clear tensors after use to save memory"""
        layer_losses = []
        layer_accuracies = []
        
        # More concise logging - just summary information
        if len(residual_streams) > 0:
            print(f"Processing {len(residual_streams)} layers...")
        
        for i in range(len(self.hooks)):
            if i in residual_streams:
                # Get residual stream tensor
                tensor = residual_streams[i]
                
                # Only use residual states for tokens with a valid next token
                valid_residual = tensor[:, :-1, :]
                
                # Free original tensor immediately
                del tensor
                
                # Make sure tensor has same dtype as the MLP to avoid errors
                dtype = next(self.mlp_probes.parameters()).dtype
                valid_residual = valid_residual.to(dtype=dtype)
                
                # Process in smaller chunks if tensor is large
                if valid_residual.shape[1] * valid_residual.shape[0] > 10000:  # Threshold based on sequence length * batch size
                    chunk_size = max(1, valid_residual.shape[1] // 4)  # Process by chunks if needed
                    all_logits = []
                    
                    for chunk_start in range(0, valid_residual.shape[1], chunk_size):
                        chunk_end = min(chunk_start + chunk_size, valid_residual.shape[1])
                        chunk = valid_residual[:, chunk_start:chunk_end, :]
                        chunk_logits = self.mlp_probes(chunk)
                        all_logits.append(chunk_logits)
                        # Clear chunk immediately
                        del chunk
                        torch.cuda.empty_cache()
                    
                    predicted_logits = torch.cat(all_logits, dim=1)
                    del all_logits
                    torch.cuda.empty_cache()
                else:
                    # Predict logits using MLP probe for this layer
                    predicted_logits = self.mlp_probes(valid_residual)
                
                # Free valid_residual immediately
                del valid_residual
                torch.cuda.empty_cache()
                
                # Calculate cross-entropy loss
                loss_fct = torch.nn.CrossEntropyLoss()
                
                # Reshape for loss calculation
                reshaped_logits = predicted_logits.view(-1, predicted_logits.size(-1))
                reshaped_labels = shifted_labels.view(-1)
                
                layer_loss = loss_fct(reshaped_logits, reshaped_labels)
                
                # Calculate accuracy
                with torch.no_grad():  # No need to track gradients for accuracy
                    predicted_tokens = torch.argmax(predicted_logits, dim=-1)
                    correct_predictions = (predicted_tokens == shifted_labels)
                    accuracy = correct_predictions.float().mean().item()
                
                layer_losses.append((i, layer_loss))
                layer_accuracies.append((i, accuracy))
                
                # Explicitly delete tensors to free memory
                del predicted_logits, predicted_tokens, correct_predictions, reshaped_logits, reshaped_labels
                torch.cuda.empty_cache()  # Clear CUDA cache
        
        return layer_losses, layer_accuracies
    
    def train(self):
        def log_memory_usage(tag, force=False):
            # Only log memory at specific points or when forced
            if not hasattr(self, "_last_memory_log"):
                self._last_memory_log = 0
            
            # Log at most every 10 batches unless forced
            current_time = time.time()
            if force or (current_time - self._last_memory_log) > 10:
                process = psutil.Process(os.getpid())
                cpu_mem = process.memory_info().rss / (1024 * 1024)
                gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                print(f"[{tag}] CPU: {cpu_mem:.1f} MB, GPU: {gpu_mem:.1f} MB / {gpu_reserved:.1f} MB")
                self._last_memory_log = current_time
        
        import time
        import os
        import wandb
        log_memory_usage("Setup", force=True)
        
        # Prepare datasets
        train_dataset = GSM8KDataset(split="train")
        eval_dataset = GSM8KDataset(split="test")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True
        )
        
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            drop_last=False
        )
        
        log_memory_usage("After data loaders", force=True)
        
        # Process layers sequentially to save memory
        num_layers = len(self.hooks)
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # Get model configuration for creating new probes
        hidden_size = self.model.config.hidden_size
        vocab_size = self.model.config.vocab_size
        device = self.model.device
        dtype = next(self.model.parameters()).dtype
        
        all_layer_metrics = {}
        
        for layer_idx in range(num_layers):
            print(f"\n--- Training probe for layer {layer_idx} ---")
            
            # Reinitialize a fresh MLP probe for each layer to start from random weights
            self.mlp_probes = ResidualStreamMLP(hidden_size, vocab_size, bottleneck_size=1024).to(device=device, dtype=dtype)
            print(f"Initialized new probe for layer {layer_idx} with random weights")
            
            # Initialize a new wandb run for this layer
            run = wandb.init(
                project="layer_probes_base",
                name=f"layer_{layer_idx}_probe",
                group="layer_probes",
                tags=[f"layer_{layer_idx}", self.model.config.model_type],
                config={
                    "layer": layer_idx,
                    "model": self.model.config.model_type,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "hidden_size": self.model.config.hidden_size,
                    "vocab_size": self.model.config.vocab_size,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "num_epochs": self.num_epochs,
                },
                reinit=True  # Allow creating multiple runs
            )
            
            # Create optimizer for this layer's probe
            optimizer = torch.optim.AdamW(
                self.mlp_probes.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01
            )
            
            layer_metrics = {
                'train_loss': [],
                'train_accuracy': [],
                'eval_loss': [],
                'eval_accuracy': []
            }
            
            # Training loop
            for epoch in range(self.num_epochs):
                print(f"  Layer {layer_idx} - Epoch {epoch+1}/{self.num_epochs}")
                self.mlp_probes.train()
                
                # Training
                train_progress = tqdm.tqdm(train_loader, desc=f"  Training L{layer_idx} E{epoch+1}")
                optimizer.zero_grad()  # Zero gradients before accumulation starts
                accumulation_count = 0
                epoch_loss = 0
                epoch_accuracy = 0
                batch_count = 0
                
                # Track global batch number for this layer
                global_batch = 0
                
                for batch_idx, batch in enumerate(train_progress):
                    log_memory_usage(f"L{layer_idx} B{batch_idx}")
                    global_batch = epoch * len(train_loader) + batch_idx
                    
                    # Move batch to device
                    batch = {k: v.to(self.model.device) for k, v in batch.items()}
                    
                    try:
                        # Forward pass through model to get final layer logits
                        with torch.no_grad():
                            outputs = self.model(**batch)
                            # Get the final layer logits directly from the model output
                            # These will be our target distribution
                            final_layer_logits = outputs.logits[:, :-1, :]  # Remove last position with no target

                        # Get labels for calculating accuracy metrics
                        labels = batch["labels"]
                        shifted_labels = labels[:, 1:].contiguous()  # We'll still use these for accuracy calculation
                        
                        residual_stream = None
                        hook, _ = self.hooks[layer_idx]
                        for module_id, tensor in hook.residual_stream.items():
                            residual_stream = tensor.clone()
                            hook.residual_stream = {}  # Clear the hook's storage
                        
                        if residual_stream is None:
                            print(f"Warning: No residual stream found for layer {layer_idx}")
                            continue
                            
                        # Only use residual states for tokens with a valid next token
                        valid_residual = residual_stream[:, :-1, :]
                        
                        # Free original tensor immediately
                        del residual_stream
                        
                        # Make sure tensor has same dtype as the MLP
                        dtype = next(self.mlp_probes.parameters()).dtype
                        valid_residual = valid_residual.to(dtype=dtype)
                        
                        # Predict logits using MLP probe
                        predicted_logits = self.mlp_probes(valid_residual)
                        del valid_residual
                        
                        # Convert logits to log probabilities
                        log_probs_predicted = torch.log_softmax(predicted_logits, dim=-1)
                        log_probs_target = torch.log_softmax(final_layer_logits, dim=-1)
                        
                        # Calculate KL divergence loss
                        # KL(P||Q) = sum(P * (log(P) - log(Q)))
                        probs_target = torch.softmax(final_layer_logits, dim=-1)
                        kl_loss = torch.nn.functional.kl_div(
                            log_probs_predicted.view(-1, log_probs_predicted.size(-1)),
                            probs_target.view(-1, probs_target.size(-1)),
                            reduction='batchmean',
                            log_target=False
                        )
                        
                        # Calculate accuracy for monitoring (comparing to ground truth)
                        predicted_tokens = predicted_logits.argmax(dim=-1)
                        mask = (shifted_labels != -100)  # Only count non-padding tokens
                        correct = ((predicted_tokens == shifted_labels) & mask).sum().item()
                        total = mask.sum().item()
                        accuracy = correct / total if total > 0 else 0
                        
                        # Scale the loss by the accumulation steps
                        scaled_loss = kl_loss / self.gradient_accumulation_steps
                        
                        # Backward pass
                        scaled_loss.backward()
                        
                        # Update metrics
                        epoch_loss += kl_loss.item()
                        epoch_accuracy += accuracy
                        batch_count += 1
                        
                        # Log current batch info
                        train_progress.set_postfix(kl_loss=kl_loss.item(), acc=accuracy)
                        
                        # Log to wandb every 5 batches
                        if batch_idx % 5 == 0:
                            wandb.log({
                                "batch_kl_loss": kl_loss.item(),
                                "batch_train_accuracy": accuracy,
                                "learning_rate": optimizer.param_groups[0]['lr'],
                                "batch": global_batch,
                                "epoch": epoch
                            })
                        
                        # Accumulate gradients
                        accumulation_count += 1
                        
                        # Only update weights after accumulating enough gradients
                        if accumulation_count == self.gradient_accumulation_steps:
                            optimizer.step()
                            optimizer.zero_grad()
                            accumulation_count = 0
                            
                        # Clear memory
                        del predicted_logits, kl_loss, final_layer_logits, log_probs_predicted, log_probs_target, probs_target
                        torch.cuda.empty_cache()
                    
                    except Exception as e:
                        print(f"Error processing batch {batch_idx} for layer {layer_idx}: {e}")
                        continue
                
                # Average metrics
                if batch_count > 0:
                    epoch_loss /= batch_count
                    epoch_accuracy /= batch_count
                    layer_metrics['train_loss'].append(epoch_loss)
                    layer_metrics['train_accuracy'].append(epoch_accuracy)
                    
                    print(f"  Layer {layer_idx} - Epoch {epoch+1} - Avg KL Loss: {epoch_loss:.4f}, Avg Acc: {epoch_accuracy:.4f}")
                    
                    # Log to wandb for this specific layer run
                    wandb.log({
                        "epoch_kl_loss": epoch_loss,
                        "epoch_train_accuracy": epoch_accuracy,
                        "epoch": epoch,
                        "epoch_completed": epoch + 1
                    })
            
            # Save the trained probe for this layer
            layer_path = self.save_path.replace('.pt', f'_layer{layer_idx}.pt')
            torch.save(self.mlp_probes.state_dict(), layer_path)
            print(f"Saved probe for layer {layer_idx} to {layer_path}")
            
            # Store metrics for this layer
            all_layer_metrics[f"layer_{layer_idx}"] = layer_metrics
            
            # Finish this layer's wandb run before moving to the next
            wandb.finish()
            
            # For the final all-layers model, we could potentially load the best probe for each layer
            torch.cuda.empty_cache()
        
        print("Sequential layer training completed!")
        
        # Return combined results from all layers
        return all_layer_metrics

def visualize_layer_performances(probe_path, model_name="Qwen/Qwen2-7B-Instruct", examples=None):
    """
    Visualize how well each layer's MLP probe predicts the next token.
    
    Args:
        probe_path: Path to the saved MLP probes
        model_name: Name of the model to load
        examples: List of example prompts to test. If None, default examples are used.
    """
    if examples is None:
        examples = [
            """<|im_start|>user
Explain quantum computing in simple terms<|im_end|>
<|im_start|>assistant
Okay, so I need to explain quantum computing in simple terms. Hmm, where do I start?"""
        ]
    
    # Load tokenizer and model
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token to EOS if it's not already defined
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
    
    # Load saved probes
    num_layers = len(model.model.layers)
    mlp_probes = torch.nn.ModuleList([
        ResidualStreamMLP(model.config.hidden_size, model.config.vocab_size)
        for _ in range(num_layers)
    ]).to(model.device)
    
    mlp_probes.load_state_dict(torch.load(probe_path))
    mlp_probes.eval()
    
    # Initialize hooks
    hooks = []
    hook_handles = []
    for i, layer in enumerate(model.model.layers):
        hook = ResidualStreamHook()
        handle = layer.register_forward_hook(hook)
        hooks.append((hook, handle))
        
    # Lists to store metrics
    all_accuracies = []
    all_losses = []
    
    # Process each example
    for example in examples:
        # Format as instruction
        prompt = f"<|im_start|>user\n{example}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate one token to capture residual streams and get target
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Get the generated token (the target)
            target_token = outputs.sequences[0, -1].unsqueeze(0).unsqueeze(0)
            
            # Collect residual streams
            residual_streams = {}
            for i, (hook, _) in enumerate(hooks):
                for module_id, tensor in hook.residual_stream.items():
                    residual_streams[i] = tensor
            
            # Calculate accuracy for each layer
            layer_accuracies = []
            layer_losses = []
            
            for i in range(num_layers):
                if i in residual_streams:
                    # Get the residual stream at the last position (just before prediction)
                    tensor = residual_streams[i][:, -1, :].unsqueeze(1)
                    
                    # Predict using MLP probe
                    predicted_logits = mlp_probes[i](tensor).squeeze(1)
                    
                    # Get predicted token
                    predicted_token = torch.argmax(predicted_logits, dim=-1)
                    
                    # Calculate accuracy (1 if correct, 0 if not)
                    accuracy = (predicted_token == target_token.squeeze()).float().item()
                    
                    # Calculate loss
                    loss = torch.nn.functional.cross_entropy(
                        predicted_logits, 
                        target_token.squeeze()
                    ).item()
                    
                    layer_accuracies.append(accuracy)
                    layer_losses.append(loss)
            
            all_accuracies.append(layer_accuracies)
            all_losses.append(layer_losses)
    
    # Clean up hooks
    for _, handle in hooks:
        handle.remove()
    
    # Calculate average metrics across examples
    avg_accuracies = np.mean(all_accuracies, axis=0)
    avg_losses = np.mean(all_losses, axis=0)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot accuracies
    plt.subplot(2, 1, 1)
    plt.plot(range(len(avg_accuracies)), avg_accuracies, marker='o', linestyle='-', linewidth=2)
    plt.title(f"Layer-wise Prediction Accuracy", fontsize=14)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    
    # Plot losses
    plt.subplot(2, 1, 2)
    plt.plot(range(len(avg_losses)), avg_losses, marker='o', linestyle='-', linewidth=2, color='r')
    plt.title(f"Layer-wise Prediction Loss", fontsize=14)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("layer_information_analysis.png", dpi=300)
    plt.close()
    print("Visualization saved as 'layer_information_analysis.png'")
    
    return avg_accuracies, avg_losses

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train layer-wise probes to identify information levels in a model")
    parser.add_argument("--max-layers", type=int, default=None, help="Limit the number of layers to process (helps with memory issues)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training (default: 2)")
    parser.add_argument("--gradient-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-7B-Instruct", help="Model to use")
    parser.add_argument("--logging-steps", type=int, default=100, help="Log every N steps")
    args = parser.parse_args()
    
    print(f"Running with max_layers={args.max_layers}, batch_size={args.batch_size}")
    
    # Run training
    identifier = InformationLevelIdentifier(
        model_name=args.model,
        batch_size=args.batch_size,
        learning_rate=5e-4,
        num_epochs=1,
        log_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accum,
        save_path="./layer_mlp_probes.pt",
        max_layers=args.max_layers
    )
    
    try:
        identifier.train()
        # Run visualization after training
        visualize_layer_performances("./layer_mlp_probes.pt")
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current state...")
        torch.save(identifier.mlp_probes.state_dict(), "./layer_mlp_probes_interrupted.pt")
        print("Saved model state to ./layer_mlp_probes_interrupted.pt")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 