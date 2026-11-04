"""Passage-grounded Qwen3 inference shared by the Gradio demo and API."""

from contextlib import nullcontext
from functools import lru_cache
import os

QWEN3_BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
QWEN3_MODEL_NAME = "Qwen3-0.6B-QLoRA"
ABSTENTION_TEXT = "No answer in the provided context."
SYSTEM_RULES = (
    "Answer using only the provided passage. Give a short factual answer. "
    "If the answer is absent, say: No answer in the provided context."
)
MAX_CONTEXT_CHARACTERS = 12000
MAX_QUESTION_CHARACTERS = 400


class QuestionTooLongError(ValueError):
    """The question alone exceeds the prompt token budget."""


@lru_cache(maxsize=1)
def load_qwen3():
    """Load Qwen3 base weights plus the user's separately trained LoRA adapter."""
    adapter_id = os.getenv("QWEN3_ADAPTER_ID", "").strip()
    if not adapter_id:
        raise ValueError("QWEN3_ADAPTER_ID is not configured.")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        QWEN3_BASE_MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_id)
    model = model.to(device).eval()

    # The training notebook saved tokenizer files alongside the LoRA adapter.
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_id)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_BASE_MODEL_ID)

    return tokenizer, model


def prepare_qwen_prompt(tokenizer, context: str, question: str):
    """Preserve the full question and cap the visible passage to fit 450 tokens."""
    offsets = tokenizer(
        context,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )["offset_mapping"]

    visible_tokens = min(300, len(offsets))
    while True:
        visible_context = ""
        if visible_tokens > 0:
            visible_context = context[:offsets[visible_tokens - 1][1]]

        messages = [
            {"role": "system", "content": SYSTEM_RULES},
            {
                "role": "user",
                "content": f"Passage:\n{visible_context}\n\nQuestion:\n{question}",
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(prompt_tokens) <= 450:
            break
        if visible_tokens == 0:
            raise QuestionTooLongError("Question is too long for the Qwen3 prompt budget.")
        visible_tokens = max(0, visible_tokens - 25)

    was_truncated = visible_tokens < len(offsets)
    return prompt, was_truncated


def generative_answer(context: str, question: str):
    """Generation is greedy, matching the Qwen3 benchmark configuration."""
    import torch

    tokenizer, model = load_qwen3()
    device = model.device
    prompt, was_truncated = prepare_qwen_prompt(tokenizer, context, question)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    autocast_context = nullcontext()
    if device.type == "cuda":
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)

    with torch.inference_mode():
        with autocast_context:
            generated = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    token_ids = generated[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    if not response:
        response = "(No text generated.)"
    return response, was_truncated
