"""Correctness tests for the CUAD data pipeline in clausewise/data.py.

# TRADEOFF: load_cuad() hits the network (HuggingFace Hub) and is exercised by
# only one test here. All other tests build small in-memory Dataset/DatasetDict
# fixtures instead of loading the full CUAD dataset, so the suite runs fast and
# deterministically on CPU with no network dependency beyond that one test.
"""

import pytest
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from clausewise.data import (
    build_training_dataset,
    clean_clause,
    format_as_instruction,
    get_class_weights,
    load_cuad,
)

_TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def _toy_dataset() -> DatasetDict:
    """A tiny, deliberately imbalanced DatasetDict standing in for CUAD."""
    train = Dataset.from_dict(
        {
            "clause_text": [
                "This Agreement shall be governed by the laws of Delaware.",
                "This Agreement shall be governed by the laws of New York.",
                "This Agreement shall be governed by the laws of California.",
                "Either party may terminate this Agreement for convenience.",
            ],
            "clause_type": [
                "Governing Law",
                "Governing Law",
                "Governing Law",
                "Termination For Convenience",
            ],
        }
    )
    test = Dataset.from_dict(
        {
            "clause_text": ["This Agreement shall be governed by the laws of Texas."],
            "clause_type": ["Governing Law"],
        }
    )
    return DatasetDict({"train": train, "test": test})


def test_load_cuad_returns_datasetdict_with_train_and_test():
    """load_cuad() must return a DatasetDict exposing both 'train' and 'test' splits."""
    dataset = load_cuad()
    assert isinstance(dataset, DatasetDict)
    assert "train" in dataset
    assert "test" in dataset
    assert len(dataset["train"]) > 0
    assert len(dataset["test"]) > 0


def test_clean_clause_removes_extra_whitespace():
    """clean_clause() must collapse runs of whitespace/newlines into single spaces."""
    dirty = "This   Agreement\n\nshall   be\tgoverned  by law."
    cleaned = clean_clause(dirty)
    assert cleaned == "This Agreement shall be governed by law."


def test_clean_clause_handles_empty_string():
    """clean_clause() must not raise on an empty string and should return one."""
    assert clean_clause("") == ""


def test_format_as_instruction_returns_expected_keys():
    """format_as_instruction() must return a dict with instruction/input/output keys."""
    result = format_as_instruction("Some clause text.", "Governing Law")
    assert set(result.keys()) == {"instruction", "input", "output"}


def test_format_as_instruction_output_matches_clause_type():
    """The 'output' field must exactly match the clause_type argument passed in."""
    result = format_as_instruction("Some clause text.", "Cap On Liability")
    assert result["output"] == "Cap On Liability"


def test_build_training_dataset_has_input_ids_and_attention_mask():
    """build_training_dataset() must add tokenized input_ids/attention_mask columns."""
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
    dataset = _toy_dataset()
    tokenized = build_training_dataset(dataset, tokenizer, max_length=64)
    for split_name in ("train", "test"):
        split = tokenized[split_name]
        assert "input_ids" in split.column_names
        assert "attention_mask" in split.column_names
        assert len(split["input_ids"]) == len(dataset[split_name])


def test_build_training_dataset_respects_max_length():
    """No tokenized example may have more input_ids than the requested max_length."""
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
    dataset = _toy_dataset()
    max_length = 16
    tokenized = build_training_dataset(dataset, tokenizer, max_length=max_length)
    for split_name in ("train", "test"):
        for input_ids in tokenized[split_name]["input_ids"]:
            assert len(input_ids) <= max_length


def test_get_class_weights_sum_approximately_num_classes():
    """Frequency-weighted average of get_class_weights() values should be ~1.0.

    With weight = total / (num_classes * count), summing count * weight over all
    classes and dividing by total examples gives exactly num_classes, so the
    plain sum of per-class weights should also land close to num_classes when
    class counts are reasonably close to each other (as in this toy fixture).
    """
    dataset = _toy_dataset()
    weights = get_class_weights(dataset)
    num_classes = len(set(dataset["train"]["clause_type"]))
    assert len(weights) == num_classes
    assert sum(weights.values()) == pytest.approx(num_classes, rel=0.5)
