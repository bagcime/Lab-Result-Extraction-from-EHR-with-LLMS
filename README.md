# Lab Result Extraction from EHR with LLMs

This repository explores the use of **Large Language Models (LLMs)** for structured **laboratory result extraction** from **Electronic Health Record (EHR)** text.  
It includes two main components:

1. **Ollama-based evaluation** — run local inference on biomedical notes using models like Llama-3, Qwen-2.5, etc.  
2. **Qwen fine-tuning pipeline** — train a domain-adapted model using 4-bit quantization and LoRA adapters with the [Unsloth](https://github.com/unslothai/unsloth) library.

---

## 🧱 Project Structure

| Folder/File | Description |
|--------------|-------------|
| `run_llm_eval.py` | Evaluate local LLMs via Ollama API (Llama-3 / Qwen-2.5 / etc.). |
| `train_eval_qwen_unsloth.py` | Fine-tune Qwen-2.5-7B-Instruct (4-bit) and evaluate JSON extraction accuracy. |
| `TestZero.csv`, `TestOne.csv`, `TestFew.csv` | Example EHR prompts for evaluation. |
| `TrainDataExtract.csv`, `TestDataExtract.csv` | Fine-tuning and evaluation datasets. |

---

## 🚀 1. Local Evaluation with Ollama

This script benchmarks **local** LLMs on EHR extraction tasks using the [Ollama](https://ollama.com) runtime.

### 🔧 Install Ollama

#### macOS or Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
