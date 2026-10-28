# Gradio deployment — DistilBERT + Qwen3

This project has a deployable Gradio app in [`space/app.py`](../space/app.py). It compares the **fine-tuned DistilBERT extractive model** and **Qwen3-0.6B + the trained QLoRA adapter** on a shared passage. No default base-model substitutes are allowed.

> **What is still required:** the actual model checkpoints from your training environment, a Hugging Face account that can create a compute Space, and publishing the `space/` folder. Committing app code to GitHub does not itself deploy a live Space.

## 1. Verify your checkpoint folders still exist

In the Colab session where the models were trained, run:

```python
from pathlib import Path
for name in ["squad_finetuned_model", "qwen3_qlora_adapter"]:
    folder = Path(name)
    print(name, "exists:", folder.is_dir())
    if folder.is_dir():
        print("files:", sorted(p.name for p in folder.iterdir())[:15])
```

For DistilBERT, expect a weight file (`model.safetensors` or `pytorch_model.bin`), `config.json`, and tokenizer files. For Qwen3, expect `adapter_config.json`, a PEFT adapter weight file and tokenizer files. **Do not rerun training merely to create a model repository if you still have these folders.** If either folder has vanished when a Colab runtime was deleted, restore it from your persistent backup or retrain that model before claiming it is deployed.

## 2. Upload the actual checkpoints to Hugging Face

Run in the **same Colab session** with the two trained folders, after checking their contents:

```python
%pip install -q "huggingface-hub==0.36.2"
from pathlib import Path
from huggingface_hub import HfApi, notebook_login

notebook_login()  # Sign in privately; do not paste a token into GitHub or chat.
api = HfApi()
username = api.whoami()["name"]

model_folders = {
    "transformer-qa-distilbert": Path("./squad_finetuned_model"),
    "transformer-qa-qwen3-qlora": Path("./qwen3_qlora_adapter"),
}
for model_name, folder in model_folders.items():
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing checkpoint: {folder}")
    repo_id = f"{username}/{model_name}"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model")
    print("Uploaded:", f"https://huggingface.co/{repo_id}")
```

Model IDs are **examples constructed from your actual authenticated username**; do not assume these repositories already exist. Make sure your model-license and data-sharing choices allow public upload.

## 3. Create a Gradio Space

Hugging Face currently requires a paid account plan to create new Gradio/Docker compute Spaces, even on CPU Basic (which has no hourly charge). GPU hardware, when selected, has usage charges. See [official Spaces overview](https://huggingface.co/docs/hub/spaces-overview) and [billing](https://huggingface.co/docs/hub/spaces-gpus).

1. Visit [Create Space](https://huggingface.co/new-space). Choose **Gradio**, a name such as `transformer-qa-demo`, and public visibility if the demo is intended for your resume.
2. Upload the **contents** of GitHub `space/` as the Space **root** (`README.md`, `app.py`, `requirements.txt`). You can upload files in the UI, clone the Space, or use `HfApi.upload_folder`.
3. In **Space → Settings → Variables**, set:
   - `DISTILBERT_MODEL_ID` = `YOUR_HF_USERNAME/transformer-qa-distilbert`
   - `QWEN3_ADAPTER_ID` = `YOUR_HF_USERNAME/transformer-qa-qwen3-qlora`
4. For **private** model repositories only, put an access token in **Settings → Secrets** (not Variables or source code). Consider whether public models are appropriate for an openly accessible portfolio demo.
5. Check **Build Logs** and **App Logs**, then test the three bundled examples. Qwen3 on CPU can be slow; a GPU improves latency but incurs hardware charges.

### Automated upload option

Once you have logged into Hugging Face in your **local terminal**, clone the project and run:

```bash
python -m pip install "huggingface-hub==0.36.2"
python scripts/publish_qa_space.py --space-id YOUR_HF_USERNAME/transformer-qa-demo --distilbert-id YOUR_HF_USERNAME/transformer-qa-distilbert --qwen-adapter-id YOUR_HF_USERNAME/transformer-qa-qwen3-qlora
```

The publishing script attempts to create the Space, configures its variables and uploads the `space/` directory. Account subscription/billing is still governed by Hugging Face, and the script cannot supply missing checkpoints.

## 4. Confirm actual deployment

Open `https://huggingface.co/spaces/YOUR_HF_USERNAME/transformer-qa-demo` **after replacing the placeholder with the real Space ID**. Both model statuses must show "configured", and you should obtain two inference outputs. These checks must be verified on the live Space before adding a "Live Demo" link to your project README.

## Runtime and inference notes

- The current app loads `Qwen/Qwen3-0.6B` together with your **LoRA adapter**, using float32 on CPU or float16 on CUDA. This is faithful adapter inference, not a new fine-tuning run.
- Qwen3 uses non-thinking chat template, `max_new_tokens=64`, and deterministic greedy decoding.
- DistilBERT uses 384-token overlapping windows, stride 128, 30-token maximum answer span and a CLS-based no-answer threshold of 0.0.
- Qwen3 sees up to ~300 initial context tokens to fit a 450-token prompt. The UI warns if the passage was shortened.
- This is an interactive research demo: it is **not** a new benchmark or evidence of production reliability.
