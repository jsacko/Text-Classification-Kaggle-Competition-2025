# 🛡️ AI Content Moderator: Hybrid LLM & Encoder Ensemble
### Jigsaw Agile Community Rules Classification (Kaggle)

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow?style=for-the-badge)](https://huggingface.co/)
[![vLLM](https://img.shields.io/badge/Inference-vLLM-blue?style=for-the-badge)](https://github.com/vllm-project/vllm)

## 📖 Overview
This project implements a **hybrid NLP architecture** to predict rule violations in Reddit comments. To uphold community norms across diverse subreddits, I engineered an ensemble combining the reasoning capabilities of **Instruction-Tuned LLMs (Qwen)** with the classification stability of **Bi-directional Encoders (DeBERTa-v3)**.

The solution addresses high-class imbalance and noisy unstructured text (emojis, obfuscated URLs) to achieve a **0.884 AUC** on the leaderboard.

## 🏗️ Technical Architecture

### 1. Models & Fine-Tuning
* **LLM:** Fine-tuned **Qwen 1.7B/7B** using **QLoRA** (4-bit quantization) via `bitsandbytes` and `PEFT` to fit within consumer GPU memory constraints while maintaining high reasoning capabilities.
* **Encoder:** Fine-tuned **DeBERTa-v3-Small** for robust feature extraction and low-latency classification.
* **Orchestration:** Utilized **Hugging Face TRL** (Transformer Reinforcement Learning) library for Supervised Fine-Tuning (SFT).

### 2. Inference Optimization (MLOps)
* **vLLM Integration:** Replaced standard inference loops with **vLLM (PageAttention)**, significantly increasing throughput and reducing memory fragmentation during generation.
* **Mixed Precision:** Employed FP16/BF16 training to optimize for modern GPU tensor cores (Tesla T4/P100).

### 3. Data Engineering
* **Semantic URL Parsing:** Developed a custom regex pipeline to extract semantic meaning from raw URLs (e.g., converting `http://soccerstreams.com/man-u-vs-chelsea` -> `soccer streams man u chelsea`), enriching the context window for the model.
* **Prompt Engineering:** Designed role-playing system prompts (`"You are an expert moderator bot..."`) to condition the LLM for binary classification tasks.

### 4. Ensemble Strategy
* **Optimization:** Used **Optuna** to perform a Bayesian search for the optimal weighted average between the LLM logits and the Encoder probabilities.
* **Validation:** Implemented **Stratified Group K-Fold** to prevent data leakage across subreddits.

## 💻 Code Snippet: vLLM Inference
```python
from vllm import LLM, SamplingParams

# Initializing the high-throughput vLLM engine
llm = LLM(
    model=BASE_MODEL_PATH,
    quantization="gptq",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.98,
    enable_lora=True,
    max_lora_rank=64
)

# Generating classification logits
outputs = llm.generate(
    texts,
    SamplingParams(
        max_tokens=1,
        logits_processors=[mclp], # Constrained decoding (Yes/No)
        logprobs=2
    ),
    lora_request=LoRARequest("default", 1, LORA_PATH)
)
