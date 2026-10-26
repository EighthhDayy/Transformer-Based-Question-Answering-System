# Reproducing the experiments in Google Colab

> **Validation status:** The repository's GitHub Actions workflow checks notebook structure and Python syntax. It does **not** download checkpoints, train on a GPU, verify model quality or certify deployment.

[**Open the notebook in Google Colab**](https://colab.research.google.com/github/EighthhDayy/Transformer-Based-Question-Answering-System/blob/main/Transformer_Based_Question_Answering_System.ipynb)

## Before running

1. In Colab, go to **Runtime → Change runtime type → GPU** (for example, T4 if offered). GPU availability and quotas depend on your Colab account.
2. Prefer a **fresh runtime** for each experiment. Notebook outputs still include archived results from older runs; they are not new measurements of the revised code.
3. Run code cells in the relevant order. If pip upgrades packages and Colab asks for a restart, restart the runtime and rerun the necessary setup/data-loading cells.
4. Models and outputs saved to `./...` are ephemeral Colab files unless you explicitly export them.

## Path A: Legacy DistilBERT + DistilGPT2

Run the notebook sequentially from the installation cell through the Gradio demo:

- **Part 1–3:** Load SQuAD 2.0, inspect the data, tokenize long contexts and map annotated character spans to answer-token positions.
- **Part 4:** Train DistilBERT (3 epochs) and save `./squad_finetuned_model` with its tokenizer.
- **Part 5:** Plot the newly generated training/evaluation logs.
- **Part 6–8:** Train DistilGPT2, save `./gpt2_finetuned_squad`, and evaluate generative perplexity.
- **Part 10:** Evaluate an untuned DistilBERT, your fine-tuned DistilBERT and a published QA checkpoint on the **same** balanced SQuAD 2.0 pilot examples. Capture the EM, token-level F1 and no-answer F1 in `qa_metrics_pilot.csv`.
- **Part 11:** Run the separate DistilGPT2 LoRA experiment; this uses a **different training budget** from the three-epoch full fine-tuning.
- **Part 12:** Launch the Gradio demo. It loads your **local fine-tuned checkpoints**, rather than silently substituting a public fallback.

The EM/F1 cell defaults to **50 answerable + 50 unanswerable examples**. This is a small, deliberately balanced pilot, **not** an official full-set SQuAD 2.0 score. Do not present it as a full benchmark.

## Path B: Optional Qwen3 + QLoRA pilot

This can be run independently of the legacy training and Gradio sections:

1. Use a **fresh Colab GPU runtime**, then run the notebook's initial dependency installation and **Part 1: Setup and Load Data** to create `dataset`.
2. Find the cell starting with `# Optional dependency setup` and change **`RUN_QWEN_QLORA = False` to `True`**. Run that cell. It installs additional dependencies and may request a restart.
3. Run **`# Qwen3 QLoRA experiment`**. The script downloads `Qwen/Qwen3-0.6B`, uses 4-bit NF4 quantization with LoRA adapters, and performs a one-epoch prompt/completion pilot on a seeded sample of SQuAD 2.0.
4. After it completes, the notebook prints **actual training seconds, GPU type, peak allocated GPU memory, effective sample counts and held-out evaluation loss**. The adapter is saved in `./qwen3_qlora_adapter`.
5. Run **`# Optional Qwen3 inference demonstration`** and manually inspect both answerable and unanswerable cases.

**Sampling detail:** The initial sample contains 150 answerable and 150 unanswerable training examples (40 + 40 for validation). Long passages are windowed around annotated answer spans. Examples in which the answer cannot fit in the visible context, or the complete prompt/completion would exceed the token budget, are dropped. The **effective** class counts are printed and may no longer be exactly balanced.

**Resource note:** This experiment downloads model weights and requires a CUDA-compatible GPU with a supported `bitsandbytes` installation. GPU type, runtime duration and memory needs vary by environment. If Colab runs out of memory, use a fresh runtime and avoid loading the two legacy models first.

## What to report

| Claim | Evidence to save |
|---|---|
| Extractive answer quality | EM / token F1 / no-answer F1 from a clearly identified validation subset |
| Generative model quality | Perplexity and qualitative answer review; clearly identify which model |
| QLoRA parameter efficiency | Model/adaptor parameter count |
| Training efficiency | GPU model, peak allocated memory, wall-clock training time, consistent run settings |
| Reproducible implementation | Notebook version, seed, package versions and GitHub Actions result |

**Do not claim** an EM/F1 improvement, Qwen3 quality increase or LoRA/QLoRA speedup until both variants have been run and evaluated under a matching protocol.

## Troubleshooting

- **Model path not found:** Run the relevant fine-tuning cell first; local checkpoints are not tracked by Git.
- **No GPU:** Select a Colab GPU runtime. The optional 4-bit QLoRA path cannot use CPU-only Colab.
- **Package import fails after pip upgrade:** Restart Colab's runtime and rerun the setup/data-loading cells.
- **Poor or hallucinated explanations:** The legacy DistilGPT2 training examples are question/context/answer strings, not supervised explanation targets. Interpret its generated text cautiously.
- **Need a stronger benchmark:** Run the full SQuAD 2.0 validation split with a carefully tuned answerability threshold and record EM, F1 and error categories.

