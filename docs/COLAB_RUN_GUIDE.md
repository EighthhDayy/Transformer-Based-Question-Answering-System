# Running the DistilBERT + Qwen3 notebook in Google Colab

This guide covers the project's **DistilBERT extractive QA** and **Qwen3 generative QA** workflows on SQuAD 2.0.

## Setup

1. Open [the notebook](../Transformer_Based_Question_Answering_System.ipynb) in Google Colab with a CUDA GPU.
2. Run the first package installation cell **once**. It pins `transformers==4.57.1`, `trl==0.23.1`, `peft==0.17.1` and `huggingface-hub==0.36.2` to avoid the cross-version import conflicts observed in the pilot.
3. If packages were changed, restart the session and **rerun the data-loading cell**. Python variables, including `dataset`, are lost on restart.
4. Avoid unpinned package upgrades such as `transformers>=4.57`; they can load incompatible APIs.

## DistilBERT extractive model

- Run SQuAD data loading, preprocessing and extractive QA training cells.
- The local checkpoint is saved at `./squad_finetuned_model`.
- The extractive QA benchmark uses 40 answerable and 40 unanswerable validation examples with sampling seed 43, consistent with Qwen3's selection rule. Record **actual** SQuAD EM/F1 and no-answer statistics from execution.
- Earlier training/validation loss values in the README are historical training outputs, **not** verified extractive QA EM/F1.

## Qwen3 generative model

- After data loading, set `RUN_QWEN_QLORA = True` in the experiment switch cell.
- Run Qwen3-0.6B 4-bit NF4 + LoRA SFT. The intended configuration is one epoch, 150 answerable and 150 unanswerable training examples, 40+40 pilot validation examples and completion-only loss.
- On NVIDIA T4, use `torch.float16` as the 4-bit compute dtype. BF16 matrix multiplication may require compute capability >= 8.0. Qwen3 inference uses a CUDA FP16 autocast context to prevent mixed Float32/BFloat16 errors encountered in the pilot.
- Demo inference uses **greedy decoding**: `do_sample=False`. For a formal benchmark, score all 80 examples and save per-example predictions, exact-match/F1, abstention metrics and latency.
- Model adapter saves to `./qwen3_qlora_adapter`. Back it up to persistent storage: Colab runtime deletion removes local artifacts.

## Pilot result interpretation

The README records a prior executed **80-example paired Qwen3 comparison** (Base vs. QLoRA fine-tuned). That result is not a claim of a new run of this refactored notebook. A different earlier pilot reported 58.75% EM/62.31% F1; the latest paired comparison reported 56.25% EM/60.44% F1. Because the underlying run configurations were not reconciled, use the clearly labeled paired run consistently and explain the variation.

Qwen3 fine-tuning increased no-answer recall but also increased false abstentions. This is a **limitation**, not a problem the project claims to have fixed.

## Validation and scope

```bash
python scripts/validate_notebook.py
```

This checks notebook JSON, Python syntax (with notebook magics normalized), and important code markers. It does **not** run model training or establish model accuracy. Any clean-runtime end-to-end claims require a separate actual GPU run.

Do not claim hosted deployment, persistent Gradio, a full SQuAD official benchmark, or DistilBERT EM/F1 without executable evidence.
