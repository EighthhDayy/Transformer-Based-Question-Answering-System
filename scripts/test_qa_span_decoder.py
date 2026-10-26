#!/usr/bin/env python3
"""Synthetic, dependency-free tests for the notebook's QA span decoder."""
import ast
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

NOTEBOOK = Path(__file__).resolve().parent.parent / "Transformer_Based_Question_Answering_System.ipynb"


def load_decoder():
    doc = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(c["source"]) for c in doc["cells"]
        if "def extract_answer_span(" in "".join(c["source"])
    )
    func = next(
        n for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "extract_answer_span"
    )
    module = ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
    namespace = {"torch": SimpleNamespace(inference_mode=contextlib.nullcontext)}
    exec(compile(module, "<notebook extract_answer_span>", "exec"), namespace)
    return namespace["extract_answer_span"]


class FakeTensor:
    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        return (len(self.data), len(self.data[0]))

    def __getitem__(self, index):
        return FakeTensor(self.data[index]) if isinstance(index, slice) else FakeTensor(self.data[index])

    def to(self, device):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.data


class FakeBatch(dict):
    def __init__(self, input_ids, attention, offsets, segments):
        super().__init__(
            input_ids=FakeTensor(input_ids),
            attention_mask=FakeTensor(attention),
            offset_mapping=FakeTensor(offsets),
        )
        self.segments = segments

    def sequence_ids(self, i):
        return self.segments[i]


class FakeTokenizer:
    is_fast = True
    cls_token_id = 101

    def __init__(self, batch):
        self.batch = batch

    def __call__(self, *args, **kwargs):
        return self.batch


class FakeModel:
    def __init__(self, start_logits, end_logits):
        self.s = start_logits
        self.e = end_logits
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, **inputs):
        i = self.calls
        self.calls += 1
        return SimpleNamespace(
            start_logits=FakeTensor([self.s[i]]),
            end_logits=FakeTensor([self.e[i]]),
        )


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.decode = load_decoder()
        self.context = "Paris is France"
        ids = [101, 200, 102, 300, 301, 302, 102, 0]
        offs = [[0, 0], [0, 5], [0, 0], [0, 5], [6, 8], [9, 15], [0, 0], [0, 0]]
        segs = [None, 0, None, 1, 1, 1, None, None]
        self.tokenizer = FakeTokenizer(FakeBatch(
            [ids], [[1, 1, 1, 1, 1, 1, 1, 0]], [offs], [segs]
        ))

    def test_extracts_original_context_substring(self):
        # A tempting question token gets an even higher raw score, but
        # cannot be selected because only context token spans are allowed.
        s = [0, 30, 0, 0, 0, 10, 0, 0]
        e = [0, 30, 0, 0, 0, 10, 0, 0]
        model = FakeModel([s], [e])
        result = self.decode(model, self.tokenizer, "Where?", self.context, "cpu")
        self.assertEqual(result, "France")
        self.assertEqual(model.calls, 1)

    def test_null_score_can_trigger_abstention(self):
        s = [15, 0, 0, 0, 0, 10, 0, 0]
        e = [15, 0, 0, 0, 0, 10, 0, 0]
        result = self.decode(
            FakeModel([s], [e]), self.tokenizer,
            "Where?", self.context, "cpu"
        )
        self.assertEqual(result, "")

    def test_long_context_second_window(self):
        base = self.tokenizer.batch
        self.tokenizer = FakeTokenizer(FakeBatch(
            [base["input_ids"].data[0]] * 2,
            [base["attention_mask"].data[0]] * 2,
            [base["offset_mapping"].data[0]] * 2,
            [base.segments[0]] * 2
        ))
        no_score = [0, 0, 0, 1, 1, 1, 0, 0]
        yes_score = [0, 0, 0, 0, 0, 12, 0, 0]
        result = self.decode(
            FakeModel([no_score, yes_score], [no_score, yes_score]),
            self.tokenizer, "Where?", self.context, "cpu"
        )
        self.assertEqual(result, "France")

    def test_empty_input_is_safe(self):
        result = self.decode(
            FakeModel([], []), self.tokenizer, "", self.context, "cpu"
        )
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
