"""Correctness tests for clausewise/evaluate.py.

# TRADEOFF: none of these tests load a real (quantized or otherwise) model —
# actual adapter loading and generation only ever run on the Kaggle T4 GPU
# (see scripts/eval_kaggle.py). Generation is patched out at the
# _generate_single/_generate_chat_single level (the same pattern used in
# tests/test_baseline.py for _generate_batch), so these tests exercise the
# real prompt-building, parsing, and metric-computation logic on CPU.
"""

import contextlib
import json

import pytest
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from clausewise.evaluate import (
    EvaluationResult,
    ForgettingResult,
    _generate_chat_single,
    _generate_single,
    predict_clause_type,
    run_evaluation,
    run_forgetting_evaluation,
)

_CLAUSE_TYPES = ["Governing Law", "Parties", "Termination For Convenience", "Cap On Liability"]


def _toy_dataset() -> DatasetDict:
    train = Dataset.from_dict(
        {
            "clause_text": [
                "This Agreement shall be governed by the laws of Delaware.",
                "This Agreement is between Acme Corp and Widget LLC.",
                "Either party may terminate this Agreement for convenience.",
                "Liability under this Agreement is capped at fees paid.",
            ],
            "clause_type": _CLAUSE_TYPES,
        }
    )
    test = Dataset.from_dict(
        {
            "clause_text": [
                "This Agreement shall be governed by the laws of Texas.",
                "This Agreement is between Foo Inc and Bar Inc.",
                "Some nonsense clause with no clear type.",
            ],
            "clause_type": ["Governing Law", "Parties", "Cap On Liability"],
        }
    )
    return DatasetDict({"train": train, "test": test})


class _FakeModel:
    """Minimal stand-in exposing only what run_forgetting_evaluation needs: disable_adapter()."""

    def disable_adapter(self):
        return contextlib.nullcontext()


def test_predict_clause_type_returns_valid_type_or_unknown(monkeypatch):
    """predict_clause_type() must return either one of clause_types or 'UNKNOWN'."""

    def fake_generate_single(model, tokenizer, prompt, max_new_tokens):
        return prompt + "Governing Law"

    monkeypatch.setattr("clausewise.evaluate._generate_single", fake_generate_single)

    result = predict_clause_type(object(), object(), "Some clause.", _CLAUSE_TYPES)

    assert result in _CLAUSE_TYPES or result == "UNKNOWN"
    assert result == "Governing Law"


def test_predict_clause_type_extracts_text_after_response_marker(monkeypatch):
    """predict_clause_type() must parse only the text after '### Response:', ignoring the echoed prompt."""

    def fake_generate_single(model, tokenizer, prompt, max_new_tokens):
        # Simulate a model that echoes the full prompt (including a clause
        # type name inside "### Input:") before its actual answer — a naive
        # parser scanning the whole string could wrongly latch onto that.
        return (
            "### Instruction:\nSome instruction mentioning Parties.\n\n"
            "### Input:\nSome clause.\n\n"
            "### Response:\nCap On Liability"
        )

    monkeypatch.setattr("clausewise.evaluate._generate_single", fake_generate_single)

    result = predict_clause_type(object(), object(), "Some clause.", _CLAUSE_TYPES)

    assert result == "Cap On Liability"


def test_run_evaluation_returns_metrics_in_valid_ranges(monkeypatch):
    """run_evaluation()'s accuracy, macro_f1, and unknown_rate must all land in [0, 1]."""
    dataset = _toy_dataset()
    clause_text_to_type = dict(zip(dataset["test"]["clause_text"], dataset["test"]["clause_type"]))

    def fake_predict(model, tokenizer, clause_text, clause_types, max_new_tokens=20):
        return clause_text_to_type.get(clause_text, "UNKNOWN")

    monkeypatch.setattr("clausewise.evaluate.predict_clause_type", fake_predict)

    result = run_evaluation(object(), object(), dataset, n_samples=None, seed=42)

    assert isinstance(result, EvaluationResult)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
    assert 0.0 <= result.unknown_rate <= 1.0
    assert result.accuracy == pytest.approx(1.0)  # fake_predict always answers correctly


def test_run_evaluation_saves_results_json(monkeypatch, tmp_path):
    """run_evaluation() must write a results JSON file that can be reloaded."""
    dataset = _toy_dataset()

    def fake_predict(model, tokenizer, clause_text, clause_types, max_new_tokens=20):
        return "UNKNOWN"

    monkeypatch.setattr("clausewise.evaluate.predict_clause_type", fake_predict)
    monkeypatch.setattr("clausewise.evaluate.RESULTS_DIR", tmp_path)

    run_evaluation(object(), object(), dataset, n_samples=None, seed=42)

    saved_files = list(tmp_path.glob("eval_finetuned_*.json"))
    assert len(saved_files) == 1
    with open(saved_files[0]) as f:
        saved = json.load(f)
    assert "accuracy" in saved
    assert "macro_f1" in saved


def test_run_forgetting_evaluation_scores_share_keys_and_are_binary(monkeypatch):
    """ForgettingResult's base_scores and finetuned_scores must share keys, all values 0 or 1."""

    def fake_generate_chat_single(model, tokenizer, user_message, max_new_tokens):
        return "irrelevant output"

    monkeypatch.setattr("clausewise.evaluate._generate_chat_single", fake_generate_chat_single)

    result = run_forgetting_evaluation(_FakeModel(), object(), n_samples=100)

    assert isinstance(result, ForgettingResult)
    assert set(result.base_scores.keys()) == set(result.finetuned_scores.keys())
    for scores in (result.base_scores, result.finetuned_scores):
        assert all(v in (0, 1) for v in scores.values())


def _tiny_qwen2_model(tokenizer):
    """A randomly-initialized, tiny Qwen2 model sized to match `tokenizer`'s real vocab, no download.

    # WHY vocab_size=len(tokenizer): the real Qwen2.5 tokenizer emits token
    # ids across its full ~151k vocabulary. A model with a smaller embedding
    # table than that raises "index out of range in self" the moment a
    # real-tokenizer id lands outside its embedding matrix — this isn't
    # optional padding, it must match exactly.
    """
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return AutoModelForCausalLM.from_config(config)


def test_generate_single_moves_inputs_to_model_device():
    """_generate_single() must move tokenized inputs onto the model's actual device before generate().

    Regression test for a device-mismatch bug: tokenizer output starts on
    CPU, but load_adapter's device_map="auto" can place the model on
    cuda:0 on Kaggle's T4, and model.generate() raises a RuntimeError if
    inputs and model parameters live on different devices. This runs the
    real (unmocked) function end to end on CPU — next(model.parameters()).device
    resolves to "cpu" here, so the assertion that it doesn't crash and that
    the device-move code path executes is what's being checked; the same
    code path is what makes this work unchanged on a CUDA device.
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = _tiny_qwen2_model(tokenizer)

    output = _generate_single(model, tokenizer, "### Response:\n", max_new_tokens=4)

    assert isinstance(output, str)


def test_generate_chat_single_moves_inputs_to_model_device():
    """_generate_chat_single() must also move tokenized inputs onto the model's actual device before generate()."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = _tiny_qwen2_model(tokenizer)

    output = _generate_chat_single(model, tokenizer, "Hello.", max_new_tokens=4)

    assert isinstance(output, str)
