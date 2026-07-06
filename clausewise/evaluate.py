"""Phase 4 evaluation: exact-match accuracy/F1 for the fine-tuned adapter, and a forgetting check.

Produces the apples-to-apples numbers against Phase 2's baseline
(results/bench_baseline_*.json) — same metric (exact clause-type match),
same test split. Actual adapter loading and generation only ever run on the
Kaggle T4 GPU (see scripts/eval_kaggle.py); this module's CPU-safe parts are
what tests/test_evaluate.py exercises locally with a mocked model.
"""

import contextlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from clausewise.baseline import parse_model_output
from clausewise.train import build_prompt_prefix, setup_quantization

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_UNKNOWN_LABEL = "UNKNOWN"


def load_adapter(base_model_name: str, adapter_path: str, config: dict) -> tuple[PeftModel, AutoTokenizer]:
    """Load the quantized base model plus its LoRA adapter, ready for evaluation.

    # WHY eval mode matters here: LoRA's lora_dropout (0.05 in
    # configs/qlora_config.yaml) is only meant to regularize during training.
    # A module in .train() mode keeps dropout active, so the adapter would
    # randomly zero ~5% of its activations on every forward pass — the same
    # clause fed twice could get two different predictions, and evaluation
    # numbers would be irreproducible run to run. model.eval() switches
    # dropout (and any other train/eval-sensitive layers) into their
    # deterministic inference behavior.
    """
    quantization_config = setup_quantization(config)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )
    # WHY load tokenizer from adapter_path, not base_model_name: trainer.save_model()
    # saved the tokenizer alongside the adapter with the pad-token fix applied
    # during training (load_model_and_tokenizer sets pad_token = eos_token when
    # missing) — loading from the adapter directory guarantees eval uses the
    # exact same tokenizer state training did, not a fresh unpatched copy.
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


def _generate_single(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Greedily generate a continuation for a raw (non-chat-templated) prompt string.

    Returns the full decoded sequence (prompt + continuation) so callers can
    split on a known marker (e.g. "### Response:") rather than relying on
    exact prompt-length bookkeeping across tokenizers.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    # WHY: the base model is loaded with device_map="auto" (load_adapter),
    # which places it on cuda:0 on Kaggle's T4 — but tokenizer output always
    # starts on CPU. Without moving inputs to the model's device first,
    # model.generate() raises a RuntimeError from mixing CPU and CUDA
    # tensors. next(model.parameters()).device reads the actual device the
    # model landed on rather than hardcoding "cuda", so this also works
    # unchanged on the CPU-only tests/dev machine.
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def _generate_chat_single(model, tokenizer, user_message: str, max_new_tokens: int) -> str:
    """Greedily generate a reply to `user_message` via the tokenizer's chat template.

    # WHY a chat template here but not in predict_clause_type: clause
    # classification was trained on the raw "### Instruction/### Input/###
    # Response" completion format (see build_prompt_prefix), so eval must
    # match that exactly. The forgetting probes are general-capability
    # questions unrelated to that format — feeding them as a bare string
    # would understate the base Instruct model's real ability, since it was
    # itself instruction-tuned to expect chat-formatted input. Using the
    # standard chat template gives both the base and fine-tuned model a fair,
    # realistic prompt for this comparison.
    """
    import torch

    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)


def predict_clause_type(
    model,
    tokenizer,
    clause_text: str,
    clause_types: list[str],
    max_new_tokens: int = 20,
) -> str:
    """Predict one clause's type by generating and parsing a completion.

    Builds the exact same prompt template used at training time
    (build_prompt_prefix), generates a completion, extracts the text after
    "### Response:", and maps it onto a valid clause type via
    parse_model_output (or "UNKNOWN" if no match).

    # TRADEOFF: max_new_tokens=20 comfortably covers every CUAD clause-type
    # name (the longest, "Unlimited/All-You-Can-Eat-License", is well under
    # 20 tokens) plus an EOS token. The risk is a degenerate generation that
    # doesn't stop cleanly — e.g. the model repeating itself or trailing off
    # into unrelated text — would get truncated mid-string at 20 tokens,
    # which could turn a would-be fuzzy-matchable output into gibberish that
    # parse_model_output can only call UNKNOWN. We accept this because the
    # alternative (a much larger max_new_tokens) would slow down a
    # 1200+-example full-test-set eval for a failure mode that, per the
    # trained model's low unknown_rate expectations, should be rare.
    """
    prompt = build_prompt_prefix(clause_text)
    full_output = _generate_single(model, tokenizer, prompt, max_new_tokens)
    response = full_output.split("### Response:")[-1].strip()
    return parse_model_output(response, clause_types)


@dataclass
class EvaluationResult:
    """Container for one exact-match evaluation run's metrics."""

    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float] = field(default_factory=dict)
    unknown_rate: float = 0.0
    n_samples: int = 0
    confusion_matrix: dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def save(self, path: Path | None = None) -> Path:
        """Serialize this result to JSON under results/, appending a timestamp so past runs are never overwritten."""
        if path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = RESULTS_DIR / f"eval_finetuned_{timestamp}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path


def run_evaluation(
    model,
    tokenizer,
    dataset,
    n_samples: int | None = None,
    seed: int = 42,
) -> EvaluationResult:
    """Run exact-match clause classification evaluation over `dataset['test']`.

    `dataset` is the DatasetDict returned by clausewise.data.load_cuad() (its
    "train" split is only used to derive the 41-way label vocabulary).
    n_samples=None evaluates the full test set; otherwise a seeded random
    subset of that size is used. Automatically saves the result to
    results/eval_finetuned_{timestamp}.json.
    """
    clause_types = sorted(set(dataset["train"]["clause_type"]))

    test_split = dataset["test"]
    if n_samples is not None:
        test_split = test_split.shuffle(seed=seed).select(range(min(n_samples, len(test_split))))

    start = time.time()
    predictions = []
    true_labels = []
    for example in test_split:
        prediction = predict_clause_type(model, tokenizer, example["clause_text"], clause_types)
        predictions.append(prediction)
        true_labels.append(example["clause_type"])
    duration_seconds = time.time() - start

    unknown_rate = sum(1 for p in predictions if p == _UNKNOWN_LABEL) / len(predictions)
    accuracy = accuracy_score(true_labels, predictions)
    # MATH: zero_division=0 avoids a ZeroDivisionWarning/NaN when a class has
    # no predicted or true samples in this subset — several CUAD classes have
    # single-digit test-split counts (see results/data_exploration.json).
    macro_f1 = f1_score(true_labels, predictions, labels=clause_types, average="macro", zero_division=0)
    per_class_scores = f1_score(true_labels, predictions, labels=clause_types, average=None, zero_division=0)
    per_class_f1 = dict(zip(clause_types, per_class_scores))

    # WHY only misclassified pairs: a confusion matrix's diagonal (correct
    # predictions) isn't a "confusion" and would just crowd out the pairs
    # that actually reveal where the model struggles.
    mistakes = Counter(
        (true, pred) for true, pred in zip(true_labels, predictions) if true != pred
    )
    confusion_matrix = {
        f"{true} -> {pred}": count for (true, pred), count in mistakes.most_common(10)
    }

    result = EvaluationResult(
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        per_class_f1={k: float(v) for k, v in per_class_f1.items()},
        unknown_rate=float(unknown_rate),
        n_samples=len(test_split),
        confusion_matrix=confusion_matrix,
        duration_seconds=float(duration_seconds),
    )
    result.save()
    return result


@dataclass
class ForgettingResult:
    """Container for one general-capability forgetting check's scores."""

    base_scores: dict[str, int] = field(default_factory=dict)
    finetuned_scores: dict[str, int] = field(default_factory=dict)
    forgetting_delta: dict[str, int] = field(default_factory=dict)

    def save(self, path: Path | None = None) -> Path:
        """Serialize this result to JSON under results/, appending a timestamp so past runs are never overwritten."""
        if path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = RESULTS_DIR / f"eval_forgetting_{timestamp}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path


# NOTE (documented per CLAUDE.md's evaluation-philosophy + README limitation
# requirement): this is a 3-prompt sanity check, not a general-capability
# benchmark. A real forgetting evaluation would run something like MMLU or
# a held-out instruction-following suite before/after fine-tuning; these
# three fixed prompts only catch gross regressions (e.g. the model
# collapsing into always emitting clause-type names for any input), not
# subtle capability loss.
_FORGETTING_TASKS = {
    "prime_numbers": {
        "prompt": "List the first 5 prime numbers.",
        "check": lambda text: all(str(n) in text for n in (2, 3, 5, 7, 11)),
    },
    "capital_of_france": {
        "prompt": "What is the capital of France?",
        "check": lambda text: "paris" in text.lower(),
    },
    "apple_arithmetic": {
        "prompt": "If I have 3 apples and eat 1, how many do I have?",
        "check": lambda text: "2" in text,
    },
}


def run_forgetting_evaluation(model, tokenizer, n_samples: int = 100) -> ForgettingResult:
    """Score `model` (fine-tuned) and its base (adapter disabled) on 3 fixed general-capability prompts.

    # NOTE: n_samples is kept in the signature per the phase spec but is
    # currently unused — there are exactly 3 fixed prompts here, not a
    # sampled set. Kept for interface stability in case a future revision
    # adds multiple paraphrases per task.
    # WHY model.disable_adapter(): PeftModel's disable_adapter() context
    # manager runs the underlying base model's forward pass without applying
    # the LoRA weights, giving true "no fine-tuning" behavior from the exact
    # same loaded model object — this avoids loading a second full copy of
    # the base model just for this comparison (real memory savings on a T4).
    """
    max_new_tokens = 50

    def _score_all(use_adapter: bool) -> dict[str, int]:
        scores = {}
        context = contextlib.nullcontext() if use_adapter else model.disable_adapter()
        with context:
            for name, task in _FORGETTING_TASKS.items():
                output = _generate_chat_single(model, tokenizer, task["prompt"], max_new_tokens)
                scores[name] = int(task["check"](output))
        return scores

    finetuned_scores = _score_all(use_adapter=True)
    base_scores = _score_all(use_adapter=False)
    forgetting_delta = {
        name: finetuned_scores[name] - base_scores[name] for name in _FORGETTING_TASKS
    }

    result = ForgettingResult(
        base_scores=base_scores,
        finetuned_scores=finetuned_scores,
        forgetting_delta=forgetting_delta,
    )
    result.save()
    return result
