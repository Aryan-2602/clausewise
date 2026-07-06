"""Correctness tests for clausewise/train.py.

# TRADEOFF: none of these tests load Qwen2.5-0.5B or any quantized weights —
# actual training only ever happens on the Kaggle T4 GPU. prepare_model_for_qlora
# is exercised against a tiny randomly-initialized Qwen2 model built entirely
# from a Qwen2Config (no download, no bitsandbytes), which has the same
# q_proj/v_proj module naming as the real model so LoraConfig's target_modules
# actually attach — this keeps the suite CPU-only and fast per CLAUDE.md.
"""

import random

import pytest
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from clausewise.train import build_trainer, format_training_example, load_config, prepare_model_for_qlora

_VALID_CONFIG = {
    "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct", "max_length": 512},
    "qlora": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
    },
    "training": {
        "output_dir": "checkpoints/",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2.0e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "fp16": False,
        "bf16": False,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "seed": 42,
    },
    "data": {
        "max_train_samples": None,
        "max_eval_samples": None,
        "use_class_weights": True,
        "min_samples_per_class": 50,
    },
}


def _tiny_qwen2_model():
    """A randomly-initialized, tiny Qwen2 model — same q_proj/v_proj naming as the real model, no download.

    # WHY these particular sizes: the <2% trainable-param assertion is a
    # real-model-scale property (LoRA overhead on a ~494M-param model is
    # tiny). A too-small toy model makes LoRA's fixed per-module overhead
    # (2*r*hidden_size per target module) a much larger fraction of the total
    # — e.g. hidden_size=16 landed at ~7%, hidden_size=128/vocab=1000 at
    # ~1.76% (too close to the threshold to be a stable regression check).
    # vocab_size=5000 pushes the embedding table large enough that the ratio
    # (~0.84%) has real margin, while staying small enough to build instantly.
    """
    config = Qwen2Config(
        vocab_size=5000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return AutoModelForCausalLM.from_config(config)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


def test_load_config_returns_dict_with_required_keys(tmp_path):
    """load_config() must return a dict containing all 5 required top-level sections."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(_VALID_CONFIG))

    config = load_config(str(config_path))

    for key in ("model", "qlora", "quantization", "training", "data"):
        assert key in config


def test_load_config_raises_on_missing_required_key(tmp_path):
    """load_config() must raise ValueError if a required top-level section is absent."""
    incomplete_config = {k: v for k, v in _VALID_CONFIG.items() if k != "quantization"}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(incomplete_config))

    with pytest.raises(ValueError):
        load_config(str(config_path))


def test_format_training_example_returns_input_ids_and_labels(tokenizer):
    """format_training_example() must return a dict with input_ids and labels of equal length."""
    example = {"clause_text": "This Agreement is governed by Delaware law.", "clause_type": "Governing Law"}
    result = format_training_example(example, tokenizer, max_length=512)

    assert "input_ids" in result
    assert "labels" in result
    assert len(result["input_ids"]) == len(result["labels"])


def test_format_training_example_masks_instruction_and_input(tokenizer):
    """Labels for the instruction+input span must be -100 so loss ignores them."""
    example = {"clause_text": "This Agreement is governed by Delaware law.", "clause_type": "Governing Law"}
    result = format_training_example(example, tokenizer, max_length=512)

    assert result["labels"][0] == -100
    assert result["labels"][:5].count(-100) == 5


def test_format_training_example_leaves_output_tokens_unmasked(tokenizer):
    """At least one label position (the clause-type output) must not be -100."""
    example = {"clause_text": "This Agreement is governed by Delaware law.", "clause_type": "Governing Law"}
    result = format_training_example(example, tokenizer, max_length=512)

    assert any(label != -100 for label in result["labels"])


def test_prepare_model_for_qlora_trainable_percent_under_two_percent():
    """QLoRA adapter should train well under 2% of parameters — confirms it's not full fine-tuning."""
    model = _tiny_qwen2_model()
    peft_model = prepare_model_for_qlora(model, _VALID_CONFIG)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_pct = trainable_params / total_params * 100

    assert trainable_params > 0
    assert trainable_pct < 2.0


def test_build_trainer_evaluate_handles_variable_length_batches():
    """build_trainer's Trainer.evaluate() must not crash on eval examples with wildly different lengths.

    Regression test: this reproduces the exact Kaggle T4 failure. A hand-written
    compute_metrics previously assumed the model always returns standard
    (batch, seq, vocab) logits — it doesn't under SFTConfig's default
    loss_type="chunked_nll" (a memory-efficient loss that never materializes
    full logits), which crashed eval with "array is 0-dimensional". The fix
    was to drop the custom compute_metrics and rely on SFTTrainer's own
    built-in eval_mean_token_accuracy — this test exercises the real
    build_trainer -> Trainer.evaluate() path end to end (tiny model, CPU,
    no download) with per-example lengths spanning 3 to 60 tokens, which is
    what triggers different per-batch padded shapes.
    """
    tiny_config = Qwen2Config(
        vocab_size=200,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = AutoModelForCausalLM.from_config(tiny_config)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    rng = random.Random(0)

    def make_example(length):
        ids = [rng.randint(0, 199) for _ in range(length)]
        labels = [-100] * (length - 2) + ids[-2:]
        return {"input_ids": ids, "attention_mask": [1] * length, "labels": labels}

    lengths = [5, 40, 8, 60, 12, 3, 25, 50, 6, 33, 9, 4]
    eval_dataset = Dataset.from_list([make_example(length) for length in lengths])

    tiny_training_config = {
        "model": {"name": "tiny", "max_length": 64},
        "training": {
            "output_dir": "/tmp/test_build_trainer_ckpt",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2.0e-4,
            "lr_scheduler_type": "constant",
            "warmup_ratio": 0.0,
            "weight_decay": 0.0,
            "fp16": False,
            "bf16": False,
            "logging_steps": 100,
            "save_strategy": "no",
            "evaluation_strategy": "no",
            "seed": 42,
        },
    }

    trainer = build_trainer(model, tok, eval_dataset, eval_dataset, tiny_training_config)
    metrics = trainer.evaluate()

    assert "eval_loss" in metrics
    assert "eval_mean_token_accuracy" in metrics
    assert 0.0 <= metrics["eval_mean_token_accuracy"] <= 1.0
