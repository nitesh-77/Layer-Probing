# Layer-Probing

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://spdx.org/licenses/MIT.html)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

**Mechanistic Interpretability of Information Evolution in Large Language Models**

Layer-Probing is a research project for analyzing the internal representations of Large Language Models (LLMs). By training MLP probes on the residual streams of transformer layers, this project tracks how predictive information about the next token emerges and stabilizes across layers — evaluated on mathematical reasoning tasks from the GSM8K benchmark.

##  Why This Project Exists

Traditional LLM evaluation relies heavily on final-layer outputs. **Layer-Probing** measures how well each layer's MLP probe matches the final layer's prediction (via accuracy and KL divergence), and how well it matches the ground truth token (via CE loss).

This project provides tools to analyze and visualize:
1. **Information Bottlenecks:** Identifying at which layer a model's internal representations converge toward its final prediction, using MLP probes trained on residual stream activations.
2. **Model Comparisons:** Comparing both the raw text outputs and the internal residual stream dynamics between a base instruction-tuned model (Qwen2) and a reasoning-distilled model (DeepSeek-R1).
3. **Residual Stream Dynamics:** Measuring how the entropy and variance of activations evolve across layers as information flows through the network.


## 📊 Key Findings & Results

### 1. Early Information Convergence
When probing `DeepSeek-R1-Distill-Qwen-7B` on GSM8K data, both KL Divergence (~1.6) and CE Loss (~2.4) drop sharply after layer 0 and stabilize by layer 1–2, showing the model aligns with its final output very early. Probe accuracy peaks at ~0.77 around layers 21–23, with a notable CE/KL spike at the final layer suggesting the last layer actively reshapes the representation.

### 2. Residual Stream Entropy (Qwen2 vs. DeepSeek-R1)
DeepSeek-R1 plateaus at a significantly higher entropy (~13) compared to Qwen2 (~10) across all middle layers. The contrast is sharpest in last-token dynamics — DeepSeek-R1 commits to a high-entropy representation by layer 12, while Qwen2 increases slowly and linearly throughout, never plateauing.

### 3. Logprob Distributions
For the prompt `"the dog the dog the dog..."`, DeepSeek-R1 is highly confident with a top-50 cumulative probability of **0.9736**, while Qwen2 is far more uncertain at **0.5303**, with probability spread across many candidate tokens.

## Key Features
- **Sequential Probe Training:** Trains individual MLP probes for every transformer layer using memory-optimized sequential processing to prevent Out-Of-Memory (OOM) errors.
- **KL Divergence & CE Loss Metrics:** Measures how closely each layer's hidden state aligns with the final output distribution using KL divergence, and against ground truth using cross-entropy loss.
- **Residual Stream Entropy Analysis:** Calculates layer-wise log entropy via the Frobenius norm of the covariance matrix, enabling comparison of information dynamics across models.
- **GSM8K Integration:** Benchmarked on math reasoning tasks to track where next-token predictions converge across layers.

## Getting Started

### Prerequisites
* Python 3.10+
* CUDA-enabled GPU (Highly recommended for probe training)

### Installation
```bash
git clone https://github.com/nitesh-77/Layer-Probing.git
cd Layer-Probing
pip install -r requirements.txt
```

## Usage

### 1. Train Probes
Train diagnostic MLP probes on the GSM8K dataset to measure when internal states begin to align with the final output:
```bash
python information_level_identifier.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --batch-size 64 --gradient-accum 4 --logging-steps 100
```
> **Note:** Use `--max-layers N` to limit training to the first N layers if you run into memory issues.

### 2. Compare Model Outputs
Observe differences in generated completions between Qwen2 and DeepSeek-R1:
```bash
python model_comparison.py --prompt "If 3x + 5 = 20, what is x?" --max_tokens 200 --temperature 0.7
```
> **Note:** If `--prompt` is omitted, the script will ask for input interactively.

### 3. Visualize Layer Alignment
Generate plots for layer-wise accuracy, CE loss, and KL divergence:
```bash
python visualize_layers.py --probe-dir "./" --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --num-examples 1000
```

### 4. Analyze Residual Stream Entropy
Compare information complexity across model architectures:
```bash
python residual_stream_viz.py
```
> **Note:** The prompt and models are configured directly in `residual_stream_viz.py`. Edit the `main()` function to change them.



## License

MIT
