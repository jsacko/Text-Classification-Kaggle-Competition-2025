# 🛡️ AI Content Moderator: Hybrid LLM & Encoder Ensemble
### Jigsaw Agile Community Rules Classification (Kaggle)

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow?style=for-the-badge)](https://huggingface.co/)
[![vLLM](https://img.shields.io/badge/Inference-vLLM-blue?style=for-the-badge)](https://github.com/vllm-project/vllm)

## 📖 Overview
This project implements a **hybrid NLP architecture** to predict rule violations in Reddit comments. To uphold community norms across diverse subreddits, I engineered an ensemble combining the reasoning capabilities of **Instruction-Tuned LLMs (Qwen)** with the classification stability of **Bi-directional Encoders (DeBERTa-v3)**.

The solution addresses high-class imbalance and noisy unstructured text (emojis, obfuscated URLs) to achieve a **0.884 AUC** on the leaderboard.

🔗 **[View the Complete Solution Notebook & Leaderboard Position](https://www.kaggle.com/code/legdend/jigsaw-llm-classifier-reddit-rule-violation)**

## 🏗️ Technical Architecture

### 1. The 3-Model Ensemble, RAG & Fine tuning
The final prediction is a weighted average of three distinct architectures, optimized for different strengths:

* **Reasoner: Qwen 0.6B (GPTQ-Int8)**
    * **Role:** Primary classification using Chain-of-Thought (CoT) reasoning.
    * **Tech:** Fine-tuned with **QLoRA** and served via **vLLM** for high throughput. 
* **Retriever (RAG): Qwen-Embedding 0.6B**
    * **Role:** Semantic search and k-NN Classification.
    * **Method:** Embedded the entire training corpus. For every new comment, the system retrieves the **Top-K (e.g., 10)** most semantically similar past violations and assigns a probability based on their weighted scores. This effectively utilizes the training data as a dynamic knowledge base.
* **Baseline: DeBERTa-v3**
    * **Role:** Lightweight baseline for rapid iteration.
    * **Status:** Used primarily for stability and feature extraction; assigned a lower weight in the final ensemble due to lower relative accuracy compared to the generative approaches.

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
```

## 📊 Performance

- **Metric:** ROC-AUC
- **Score:** 0.884 (Public Leaderboard)
- **Key Insight:** The ensemble of *Clean Text + URL Semantics* outperformed raw text inputs by significant margins, proving that feature engineering remains crucial even in the LLM era.

## 🛠️ Tools Used

- **Libraries:** `pytorch`, `tensorflow`, `transformers`, `peft`, `trl`, `vllm`, `deepspeed`, `optimum`, `auto-gptq`, `optuna`
- **Hardware:** Nvidia Tesla T4 x2 (Kaggle Environment)

---

## 👨‍💻 Author

**Julien SACKO** | Machine Learning Engineer 

[LinkedIn](https://www.linkedin.com/in/julien-sacko/)
