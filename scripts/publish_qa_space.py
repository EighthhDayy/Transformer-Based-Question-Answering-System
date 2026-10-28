#!/usr/bin/env python3
"""Create/update the Gradio Space after Hugging Face login.

Never pass tokens as CLI arguments. Authenticate with hf auth login or notebook_login.
This script does not create or upload the model checkpoints.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Publish the Gradio demo to Hugging Face.")
    parser.add_argument("--space-id", required=True, help="username/space-name")
    parser.add_argument("--distilbert-id", required=True, help="username/model-name")
    parser.add_argument("--qwen-adapter-id", required=True, help="username/adapter-model-name")
    args = parser.parse_args()

    folder = Path(__file__).resolve().parent.parent / "space"
    for filename in ["README.md", "app.py", "requirements.txt"]:
        if not (folder / filename).is_file():
            raise FileNotFoundError(folder / filename)

    api = HfApi()
    logged_in = api.whoami()
    print("Authenticated as:", logged_in["name"])

    # Hugging Face may reject new Gradio Spaces without an eligible paid plan.
    api.create_repo(
        repo_id=args.space_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )

    api.add_space_variable(args.space_id, key="DISTILBERT_MODEL_ID", value=args.distilbert_id)
    api.add_space_variable(args.space_id, key="QWEN3_ADAPTER_ID", value=args.qwen_adapter_id)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.space_id,
        repo_type="space",
        commit_message="Deploy DistilBERT and Qwen3 Gradio comparison demo",
    )
    print("Space source uploaded:", f"https://huggingface.co/spaces/{args.space_id}")
    print("Verify live Build/App logs and predictions before sharing the URL.")


if __name__ == "__main__":
    main()
