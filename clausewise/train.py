"""QLoRA fine-tuning pipeline for CUAD clause classification.

Every hyperparameter lives in configs/qlora_config.yaml — nothing here is
hardcoded. Actual training only ever runs on the Kaggle T4 GPU (see
scripts/train_kaggle.py); this module's CPU-safe parts (config loading,
formatting/masking, LoRA adapter construction) are what tests/test_train.py
exercises locally.
"""

import json
import time
from pathlib import Path

import yaml
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from clausewise.data import INSTRUCTION, clean_clause, load_cuad, oversample_minority_classes

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

_REQUIRED_TOP_LEVEL_KEYS = ("model", "qlora", "quantization", "training", "data")

# WHY: transformers renamed "evaluation_strategy" -> "eval_strategy" (the name
# CLAUDE.md's config template uses is the old one). Rather than silently
# breaking when TrainingArguments/SFTConfig rejects an unknown kwarg, we
# rename it here so the YAML can keep the familiar key name.
_TRAINING_KEY_RENAMES = {"evaluation_strategy": "eval_strategy"}


def load_config(config_path: str) -> dict:
    """Load and validate configs/qlora_config.yaml.

    Raises ValueError if any of the top-level sections
    (model, qlora, quantization, training, data) are missing.
    Returns the parsed config dict.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    missing = [key for key in _REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing:
        raise ValueError(f"config at {config_path} is missing required keys: {missing}")

    return config


def setup_quantization(config: dict) -> BitsAndBytesConfig:
    """Build a BitsAndBytesConfig for 4-bit QLoRA loading from `config['quantization']`.

    # WHY NF4 over plain INT4: pretrained transformer weights are approximately
    # normally distributed (this is the empirical premise QLoRA's paper builds
    # on), so a data type whose quantization buckets are spaced to match a
    # normal distribution's density — NF4 (4-bit NormalFloat) — allocates more
    # precision where most weight mass actually sits. A uniform INT4 grid
    # wastes bits on the tails and under-resolves the dense center, which
    # empirically increases quantization error for the same 4 bits/weight.
    # WHY double quantization: the first quantization pass still needs a
    # per-block scaling constant (typically fp32) to map int levels back to
    # real values; double quantization quantizes those scaling constants
    # themselves with a second, smaller quantizer. This trims roughly another
    # ~0.4 bits/parameter of memory with negligible additional error, which
    # matters when the whole point is fitting a model + optimizer state in a
    # 16GB T4.
    """
    quant_cfg = config["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=quant_cfg["bnb_4bit_compute_dtype"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )


def load_model_and_tokenizer(config: dict) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model in 4-bit and its tokenizer, per `config['model']`.

    # TRADEOFF: Qwen2.5-0.5B-Instruct ships with no dedicated pad token, so we
    # reuse eos_token as pad_token (the standard workaround for causal LMs).
    # The risk: if a batch's attention mask doesn't correctly zero out padded
    # positions, the model can attend to trailing pad/eos tokens and, since
    # they're literally eos, learn to associate "end of sequence" with
    # whatever follows padding rather than with the true end of each example.
    # We accept this because format_training_example's per-example tokenization
    # (no cross-example packing) combined with the data collator's attention
    # mask keeps padded positions correctly excluded from attention and loss;
    # the risk would only materialize if padding/masking were done carelessly.
    """
    model_cfg = config["model"]
    quantization_config = setup_quantization(config)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=quantization_config,
        device_map="auto",
    )
    return model, tokenizer


def prepare_model_for_qlora(model, config: dict):
    """Wrap `model` for k-bit training and attach a LoRA adapter per `config['qlora']`.

    Prints the trainable parameter count and its percentage of total
    parameters, then returns the resulting PEFT model.
    """
    model = prepare_model_for_kbit_training(model)

    qlora_cfg = config["qlora"]
    lora_config = LoraConfig(
        r=qlora_cfg["r"],
        lora_alpha=qlora_cfg["lora_alpha"],
        lora_dropout=qlora_cfg["lora_dropout"],
        target_modules=qlora_cfg["target_modules"],
        bias=qlora_cfg["bias"],
        task_type=qlora_cfg["task_type"],
    )
    peft_model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    # MATH: percentage of parameters QLoRA actually updates, out of every
    # parameter (frozen base + adapter). Should land under ~2% for r=8 on a
    # 0.5B model — that's the whole point of LoRA versus full fine-tuning.
    trainable_pct = trainable_params / total_params * 100
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({trainable_pct:.4f}%)")

    return peft_model


def format_training_example(example: dict, tokenizer, max_length: int = 512) -> dict:
    """Tokenize one (clause_text, clause_type) example with the output-only loss mask.

    Builds the "### Instruction / ### Input / ### Response" string, tokenizes
    it, and sets labels = input_ids everywhere except the instruction+input
    span, which is set to -100 (ignored by cross-entropy).

    # WHY: masking the instruction+input tokens means the model is never
    # penalized for "failing to predict" text it was only ever supposed to
    # read, not generate. Without masking, cross-entropy loss would spend most
    # of its gradient signal teaching the model to reproduce the (fixed,
    # already-known) instruction template and the input clause verbatim —
    # exactly the wrong thing for a classifier whose only actual output is a
    # short clause-type label.
    # TRADEOFF: because the output is just a clause-type name (a handful of
    # tokens), each training example contributes a very small number of
    # non-masked label positions relative to sequence length. This makes
    # per-example loss noisy/high-variance (a single wrong token in a 3-token
    # label swings loss much more than one wrong token would in a long
    # generative response), and can make convergence look noisier than a
    # typical instruction-tuning run even when the model is still learning.
    """
    clause_type = example["clause_type"]
    clause_text = clean_clause(example["clause_text"])

    prefix = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{clause_text}\n\n### Response:\n"
    full_text = prefix + clause_type + tokenizer.eos_token

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        full_text, truncation=True, max_length=max_length, add_special_tokens=False
    )
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    prefix_len = min(len(prefix_ids), len(input_ids))
    labels = [-100] * prefix_len + input_ids[prefix_len:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _preprocess_logits_for_metrics(logits, labels):
    """Reduce (batch, seq_len, vocab_size) logits to argmax token ids before Trainer accumulates them.

    # TRADEOFF: Trainer's default eval loop concatenates every batch's raw
    # logits in memory before calling compute_metrics. With a ~150k
    # vocabulary and 512-token sequences, that's tens of GB across a full
    # eval split — well past a T4's 16GB. Argmax-ing per batch here throws
    # away the actual probabilities (so we can only ever compute discrete
    # accuracy, not e.g. log-loss) but keeps eval memory bounded by sequence
    # length instead of vocab size.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def _compute_metrics(eval_pred) -> dict[str, float]:
    """Compute exact-match accuracy over non-masked label positions for Trainer's eval loop.

    # WHY the shift: Trainer's eval_pred hands back raw, unshifted logits and
    # labels — logits at position t are the model's prediction for the token
    # at position t+1, but labels[t] is the actual token at position t.
    # Comparing them position-for-position without shifting silently computes
    # an off-by-one-misaligned accuracy (transformers' own loss functions do
    # this shift internally, which is easy to forget when reimplementing a
    # metric outside that path).
    """
    import numpy as np

    predictions, labels = eval_pred
    predictions = np.asarray(predictions)[:, :-1]
    labels = np.asarray(labels)[:, 1:]

    mask = labels != -100
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / max(mask.sum(), 1)
    return {"accuracy": float(accuracy)}


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: dict,
) -> SFTTrainer:
    """Construct an SFTTrainer from pre-tokenized datasets and `config['training']`.

    `train_dataset`/`eval_dataset` are expected to already carry input_ids,
    attention_mask, and labels (i.e. already run through format_training_example)
    — SFTTrainer detects a pre-processed dataset via the presence of the
    "input_ids" column and skips its own formatting/tokenization step.

    # WHY not wired here: config['data']['use_class_weights'] is read by
    # clausewise.data.get_class_weights() but not applied to this trainer's
    # loss. SFTTrainer's default loss is token-level cross-entropy computed
    # after `labels` has already been popped from the batch — there's no
    # per-example hook that also has access to which clause_type produced
    # each sequence without a custom Trainer subclass overriding compute_loss.
    # That's a real feature, not a one-line fix, and out of scope for this
    # phase's explicit spec; flagged here so it isn't silently forgotten.
    """
    training_cfg = dict(config["training"])
    for old_key, new_key in _TRAINING_KEY_RENAMES.items():
        if old_key in training_cfg:
            training_cfg[new_key] = training_cfg.pop(old_key)

    sft_config = SFTConfig(
        max_length=config["model"]["max_length"],
        **training_cfg,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
        preprocess_logits_for_metrics=_preprocess_logits_for_metrics,
    )
    return trainer


def _build_formatted_dataset(dataset: DatasetDict, tokenizer, config: dict) -> DatasetDict:
    """Tokenize every split via format_training_example.

    Assumes any sampling/oversampling has already been applied to `dataset`'s
    splits (see run_training) — this step only formats and tokenizes.
    """
    max_length = config["model"]["max_length"]

    def _map_fn(example):
        return format_training_example(example, tokenizer, max_length=max_length)

    return DatasetDict(
        {
            split_name: split.map(_map_fn, remove_columns=split.column_names)
            for split_name, split in dataset.items()
        }
    )


def run_training(config_path: str) -> str:
    """Run the full QLoRA fine-tuning pipeline and return the path to the saved adapter.

    Steps: load config -> quantization config -> model/tokenizer -> LoRA adapter
    -> load CUAD -> (truncate if configured) -> oversample rare classes in the
    training split -> tokenize -> build trainer -> train -> save adapter to
    checkpoints/final/ -> save training metrics to results/training_{timestamp}.json.
    """
    config = load_config(config_path)

    model, tokenizer = load_model_and_tokenizer(config)
    model = prepare_model_for_qlora(model, config)

    raw_dataset = load_cuad()
    data_cfg = config["data"]

    train_split = raw_dataset["train"]
    if data_cfg.get("max_train_samples"):
        train_split = train_split.select(range(min(data_cfg["max_train_samples"], len(train_split))))
    # WHY oversample after truncation, not before: max_train_samples exists for
    # the quick Kaggle smoke test (500 rows). Oversampling the full ~11k-row
    # split first and then slicing to 500 would mostly just take the first 500
    # original rows, since concatenate_datasets appends oversampled rows at
    # the end — the rare-class boost would never survive the slice. Rebalancing
    # the already-truncated split guarantees every class clears the floor in
    # whatever subset actually gets trained on.
    train_split = oversample_minority_classes(
        train_split,
        min_samples_per_class=data_cfg["min_samples_per_class"],
        seed=config["training"]["seed"],
    )

    eval_split = raw_dataset["test"]
    if data_cfg.get("max_eval_samples"):
        eval_split = eval_split.select(range(min(data_cfg["max_eval_samples"], len(eval_split))))

    formatted_dataset = _build_formatted_dataset(
        DatasetDict({"train": train_split, "test": eval_split}), tokenizer, config
    )

    trainer = build_trainer(
        model, tokenizer, formatted_dataset["train"], formatted_dataset["test"], config
    )

    train_result = trainer.train()

    adapter_path = CHECKPOINTS_DIR / "final"
    adapter_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_path))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    metrics_path = RESULTS_DIR / f"training_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "config_path": config_path,
                "train_result_metrics": train_result.metrics,
                "log_history": trainer.state.log_history,
            },
            f,
            indent=2,
        )

    return str(adapter_path)
