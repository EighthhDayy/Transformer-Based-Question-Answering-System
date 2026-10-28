---
title: Transformer QA - DistilBERT + Qwen3
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# Transformer QA · DistilBERT + Qwen3

Side-by-side extractive and generative question answering over user-supplied text.

## Model configuration

Set these **Space Variables** before expecting real inference:

- `DISTILBERT_MODEL_ID`: Hugging Face model repository containing the actual fine-tuned DistilBERT QA checkpoint and tokenizer
- `QWEN3_ADAPTER_ID`: Hugging Face model repository containing the actual Qwen3-0.6B QLoRA adapter and tokenizer

This Space **does not substitute base checkpoints** if fine-tuned weights are missing. Both inference paths are loaded lazily on first request. On CPU, Qwen3 runs in float32 (no bitsandbytes dependency); it may be slow. On GPU, it uses float16.

For installation and checkpoint upload instructions, see [the project deployment guide](https://github.com/EighthhDayy/Transformer-Based-Question-Answering-System/blob/main/docs/GRADIO_DEPLOYMENT.md). This is a pilot model demonstration, not a production API.
