"""Gradio demo for the user's fine-tuned DistilBERT and Qwen3 QA models.

The app NEVER silently substitutes public base models for missing checkpoints.
Configure DISTILBERT_MODEL_ID and QWEN3_ADAPTER_ID on Hugging Face Spaces.
"""

from contextlib import nullcontext
from functools import lru_cache
import os

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)

DISTILBERT_MODEL_ID = os.getenv("DISTILBERT_MODEL_ID", "").strip()
QWEN3_ADAPTER_ID = os.getenv("QWEN3_ADAPTER_ID", "").strip()
QWEN3_BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
ABSTENTION_TEXT = "No answer in the provided context."
SYSTEM_RULES = (
    "Answer using only the provided passage. Give a short factual answer. "
    "If the answer is absent, say: No answer in the provided context."
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CONTEXT_CHARACTERS = 12000


@lru_cache(maxsize=1)
def load_distilbert():
    """Load the actual fine-tuned extractive QA checkpoint."""
    if not DISTILBERT_MODEL_ID:
        raise ValueError("DISTILBERT_MODEL_ID is not configured.")
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_ID, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for offset mapping.")
    model = AutoModelForQuestionAnswering.from_pretrained(DISTILBERT_MODEL_ID)
    model = model.to(DEVICE).eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def load_qwen3():
    """Load Qwen3 base weights plus the user's separately trained LoRA adapter."""
    if not QWEN3_ADAPTER_ID:
        raise ValueError("QWEN3_ADAPTER_ID is not configured.")

    from peft import PeftModel

    dtype = torch.float32
    if DEVICE.type == "cuda":
        dtype = torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        QWEN3_BASE_MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, QWEN3_ADAPTER_ID)
    model = model.to(DEVICE).eval()

    # The training notebook saved tokenizer files alongside the LoRA adapter.
    try:
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_ADAPTER_ID)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_BASE_MODEL_ID)

    return tokenizer, model


def extractive_answer(context: str, question: str) -> str:
    """DistilBERT start/end span decoding with overflow windows and CLS null score."""
    tokenizer, model = load_distilbert()
    features = tokenizer(
        question.strip(),
        context,
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )

    best_answer = ""
    best_span_score = float("-inf")
    best_null_score = float("inf")
    max_answer_tokens = 30

    for window_idx in range(features["input_ids"].shape[0]):
        token_ids = features["input_ids"][window_idx].tolist()
        offsets = features["offset_mapping"][window_idx].tolist()
        segments = features.sequence_ids(window_idx)
        attention_mask = features["attention_mask"][window_idx].tolist()
        cls_index = token_ids.index(tokenizer.cls_token_id)

        with torch.inference_mode():
            result = model(
                input_ids=features["input_ids"][window_idx:window_idx + 1].to(DEVICE),
                attention_mask=features["attention_mask"][window_idx:window_idx + 1].to(DEVICE),
            )

        start_logits = result.start_logits[0].float().cpu().tolist()
        end_logits = result.end_logits[0].float().cpu().tolist()

        best_null_score = min(
            best_null_score,
            start_logits[cls_index] + end_logits[cls_index],
        )

        valid_tokens = [
            pos
            for pos, segment in enumerate(segments)
            if segment == 1
            and attention_mask[pos] == 1
            and offsets[pos][0] < offsets[pos][1]
        ]
        valid_set = set(valid_tokens)

        for start_idx in valid_tokens:
            end_limit = min(start_idx + max_answer_tokens, len(end_logits))
            for end_idx in range(start_idx, end_limit):
                if end_idx not in valid_set:
                    continue
                char_start = offsets[start_idx][0]
                char_end = offsets[end_idx][1]
                if char_start >= char_end:
                    continue
                score = start_logits[start_idx] + end_logits[end_idx]
                if score > best_span_score:
                    best_span_score = score
                    best_answer = context[char_start:char_end]

    if best_span_score == float("-inf"):
        return ABSTENTION_TEXT
    if best_null_score - best_span_score > 0.0:
        return ABSTENTION_TEXT
    return best_answer.strip()


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
            raise ValueError("Question is too long for the Qwen3 prompt budget.")
        visible_tokens = max(0, visible_tokens - 25)

    was_truncated = visible_tokens < len(offsets)
    return prompt, was_truncated


def generative_answer(context: str, question: str):
    """Generation is greedy, matching the Qwen3 benchmark configuration."""
    tokenizer, model = load_qwen3()
    prompt, was_truncated = prepare_qwen_prompt(tokenizer, context, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    autocast_context = nullcontext()
    if DEVICE.type == "cuda":
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


def compare_answers(context, question):
    context = (context or "").strip()
    question = (question or "").strip()

    if not context or not question:
        return "Enter a passage and a question.", "Enter a passage and a question.", ""
    if len(context) > MAX_CONTEXT_CHARACTERS:
        return (
            f"Passage too long (limit {MAX_CONTEXT_CHARACTERS:,} characters).",
            "Reduce passage length.",
            "",
        )

    notes = []
    if not DISTILBERT_MODEL_ID:
        extractive = "Checkpoint not configured: set DISTILBERT_MODEL_ID in Space Variables."
    else:
        try:
            extractive = extractive_answer(context, question)
        except Exception as exc:
            print(f"DistilBERT inference error: {type(exc).__name__}: {exc}", flush=True)
            extractive = "DistilBERT unavailable. Check the Space logs and model repository."

    if not QWEN3_ADAPTER_ID:
        generative = "Adapter not configured: set QWEN3_ADAPTER_ID in Space Variables."
    else:
        try:
            generative, was_truncated = generative_answer(context, question)
            if was_truncated:
                notes.append(
                    "Qwen3 used an initial passage window to fit its 450-token prompt; "
                    "relevant evidence later in the passage may not be visible."
                )
        except Exception as exc:
            print(f"Qwen3 inference error: {type(exc).__name__}: {exc}", flush=True)
            generative = "Qwen3 unavailable. Check the Space logs and adapter repository."

    return extractive, generative, "\n\n".join(notes)


def availability_message():
    extractive_status = "configured" if DISTILBERT_MODEL_ID else "checkpoint missing"
    generative_status = "configured" if QWEN3_ADAPTER_ID else "adapter missing"
    return (
        f"**Deployment configuration:** DistilBERT: {extractive_status} · "
        f"Qwen3: {generative_status} · Inference device: {DEVICE.type.upper()}. "
        "Only the user's fine-tuned checkpoints are used; there is no silent model fallback."
    )


with gr.Blocks(title="Transformer QA | DistilBERT + Qwen3") as demo:
    gr.Markdown("# Transformer Question Answering\n### DistilBERT × Qwen3-0.6B")
    gr.Markdown(
        "Compare extractive span prediction and QLoRA-adapted generative "
        "question answering on the **same supplied passage**. "
        "The models can abstain when an answer is not supported by the passage."
    )
    gr.Markdown(availability_message())

    with gr.Row():
        with gr.Column(scale=3):
            passage_input = gr.Textbox(
                label="Passage / Context",
                lines=11,
                max_lines=18,
                placeholder="Paste the paragraph that contains the evidence...",
            )
            question_input = gr.Textbox(
                label="Question",
                lines=2,
                placeholder="What would you like to know about the passage?",
            )
            submit = gr.Button("Compare answers", variant="primary")
        with gr.Column(scale=2):
            extracted_output = gr.Textbox(
                label="DistilBERT · Extractive answer",
                lines=4,
                interactive=False,
            )
            generated_output = gr.Textbox(
                label="Qwen3 · QLoRA generative answer",
                lines=4,
                interactive=False,
            )
            notes_output = gr.Markdown()

    submit.click(
        fn=compare_answers,
        inputs=[passage_input, question_input],
        outputs=[extracted_output, generated_output, notes_output],
    )

    gr.Examples(
        examples=[
            [
                "Normandy is a geographical and cultural region in northwestern France. "
                "It includes mainland Normandy and the Channel Islands.",
                "In which country is Normandy located?",
            ],
            [
                "The Tower District is a neighborhood in Fresno, California. "
                "Since renewal, the Tower District has become an attractive area "
                "for restaurants and other local businesses.",
                "What area has become attractive for restaurants?",
            ],
            [
                "The Moon is Earth's only natural satellite.",
                "Who invented the first telescope?",
            ],
        ],
        inputs=[passage_input, question_input],
    )
    gr.Markdown(
        "**Research demo:** SQuAD 2.0 pilot models, not a medical, legal, "
        "or production decision system. Abstentions and incorrect answers can occur. "
        "Qwen3 inference on CPU can be slow."
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
