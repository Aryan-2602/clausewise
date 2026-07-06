"""Correctness tests for clausewise/baseline.py.

# TRADEOFF: none of these tests load real model weights — run_baseline_evaluation
# is exercised against a small fake model/tokenizer pair that returns
# deterministic canned text, so the suite stays fast and CPU-only per
# CLAUDE.md's testing rules, while still checking the parsing/metric-shape
# contract that a real model's output has to satisfy.
"""

from datasets import Dataset, DatasetDict

from clausewise.baseline import (
    BaselineResult,
    build_few_shot_prompt,
    build_zero_shot_prompt,
    parse_model_output,
    run_baseline_evaluation,
    select_diverse_examples,
)

_CLAUSE_TYPES = [
    "Governing Law",
    "Parties",
    "Termination For Convenience",
    "Cap On Liability",
    "Audit Rights",
]


def _toy_dataset() -> DatasetDict:
    train = Dataset.from_dict(
        {
            "clause_text": [
                "This Agreement shall be governed by the laws of Delaware.",
                "This Agreement is between Acme Corp and Widget LLC.",
                "Either party may terminate this Agreement for convenience.",
                "Liability under this Agreement is capped at fees paid.",
                "Company may audit Distributor's records once per year.",
            ],
            "clause_type": _CLAUSE_TYPES,
        }
    )
    test = Dataset.from_dict(
        {
            "clause_text": [
                "This Agreement shall be governed by the laws of Texas.",
                "This Agreement is between Foo Inc and Bar Inc.",
            ],
            "clause_type": ["Governing Law", "Parties"],
        }
    )
    return DatasetDict({"train": train, "test": test})


class _FakeTokenizer:
    """Stands in for a HF tokenizer: run_baseline_evaluation only reads these attributes directly (generation itself is patched out via _patch_generate_batch_always_correct)."""

    pad_token = None
    eos_token = "<eos>"
    pad_token_id = 0
    padding_side = "right"


def _patch_generate_batch_always_correct(monkeypatch, clause_text_to_type):
    """Patch _generate_batch to always output the true label, regardless of dataset.shuffle() ordering."""

    def fake_generate_batch(model, tokenizer, prompts, max_new_tokens=16):
        outputs = []
        for prompt in prompts:
            match = next(text for text in clause_text_to_type if text in prompt)
            outputs.append(clause_text_to_type[match])
        return outputs

    monkeypatch.setattr("clausewise.baseline._generate_batch", fake_generate_batch)


def test_build_zero_shot_prompt_contains_all_clause_types():
    """The zero-shot prompt must list every clause type name so the model can pick from them."""
    prompt = build_zero_shot_prompt("Some clause text.", _CLAUSE_TYPES)
    for clause_type in _CLAUSE_TYPES:
        assert clause_type in prompt


def test_build_zero_shot_prompt_contains_clause_text():
    """The zero-shot prompt must embed the clause text to be classified."""
    clause_text = "This is a unique clause about indemnification obligations."
    prompt = build_zero_shot_prompt(clause_text, _CLAUSE_TYPES)
    assert clause_text in prompt


def test_build_few_shot_prompt_contains_exactly_n_shots_examples():
    """build_few_shot_prompt must render exactly n_shots example blocks."""
    pool = [
        {"clause_text": f"Example clause {i}", "clause_type": _CLAUSE_TYPES[i % len(_CLAUSE_TYPES)]}
        for i in range(5)
    ]
    examples = select_diverse_examples(pool, n_shots=3, seed=42)
    prompt = build_few_shot_prompt("Target clause text.", _CLAUSE_TYPES, examples, n_shots=3)
    assert prompt.count("Clause type:") == 3


def test_parse_model_output_exact_match():
    """An exact (case-sensitive) label string must be returned unchanged."""
    assert parse_model_output("Governing Law", _CLAUSE_TYPES) == "Governing Law"


def test_parse_model_output_fuzzy_match_close_variation():
    """A close variant like 'Governing law' (wrong case) must still resolve to 'Governing Law'."""
    assert parse_model_output("Governing law", _CLAUSE_TYPES) == "Governing Law"


def test_parse_model_output_returns_unknown_for_nonsense():
    """Nonsense output unrelated to any clause type must map to UNKNOWN, not a wrong guess."""
    assert parse_model_output("asdkjfh qwoeiru zzz 12345", _CLAUSE_TYPES) == "UNKNOWN"


def test_baseline_result_fields_are_valid_ranges(monkeypatch):
    """accuracy and macro_f1 from a real run_baseline_evaluation call must land in [0, 1]."""
    dataset = _toy_dataset()
    clause_text_to_type = dict(zip(dataset["test"]["clause_text"], dataset["test"]["clause_type"]))
    _patch_generate_batch_always_correct(monkeypatch, clause_text_to_type)

    result = run_baseline_evaluation(
        model=object(),
        tokenizer=_FakeTokenizer(),
        dataset=dataset,
        mode="zero_shot",
        n_samples=2,
        seed=42,
        batch_size=8,
    )

    assert isinstance(result, BaselineResult)
    assert result.mode == "zero_shot"
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
    assert all(0.0 <= f1 <= 1.0 for f1 in result.per_class_f1.values())
    assert 0.0 <= result.unknown_rate <= 1.0
    assert result.n_samples == 2
    assert result.duration_seconds >= 0.0
    # This toy run predicts both labels correctly.
    assert result.accuracy == 1.0
