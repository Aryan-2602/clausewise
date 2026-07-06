"""Zero-shot and few-shot baseline evaluation for CUAD clause classification.

This is the control we must beat with QLoRA fine-tuning (CLAUDE.md Phase 2).
Every number here comes from actually running the base model — no theoretical
estimates.
"""

import difflib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_UNKNOWN_LABEL = "UNKNOWN"
# TRADEOFF: 0.6 is difflib's own default cutoff. Raising it would miss
# legitimate near-misses (capitalization, punctuation); lowering it increases
# the risk described in parse_model_output's docstring below. 0.6 is a
# starting point, not a tuned value — revisit once we see real UNKNOWN rates.
_FUZZY_MATCH_CUTOFF = 0.6


def build_zero_shot_prompt(clause_text: str, clause_types: list[str]) -> str:
    """Build a zero-example classification prompt listing all 41 CUAD clause types.

    # TRADEOFF: listing all 41 types by name (rather than e.g. just asking the
    # model to "name the clause type") is necessary because the model has no
    # other way to know CUAD's exact label vocabulary — clause type names like
    # "Rofr/Rofo/Rofn" or "Affiliate License-Licensor" aren't things the model
    # would reliably reproduce from parametric knowledge alone, and without the
    # exact string, parse_model_output's exact-match path would always miss.
    # The cost is real: ~41 lines of label text added to every single prompt,
    # which at 200 samples x 2 modes multiplies token usage and CPU eval time
    # considerably. We pay it anyway because an unconstrained free-text
    # response would make evaluation unreliable, not just expensive.
    """
    numbered_types = "\n".join(f"{i + 1}. {ct}" for i, ct in enumerate(clause_types))
    return (
        "Classify the following legal contract clause into exactly one of "
        f"these {len(clause_types)} CUAD clause types:\n\n"
        f"{numbered_types}\n\n"
        f'Clause:\n"""\n{clause_text}\n"""\n\n'
        "Respond with ONLY the exact clause type name from the list above, "
        "and nothing else."
    )


def build_few_shot_prompt(
    clause_text: str,
    clause_types: list[str],
    examples: list[dict],
    n_shots: int = 3,
) -> str:
    """Build a classification prompt preceded by n_shots labeled examples.

    `examples` is expected to already be the diverse subset selected by the
    caller (see select_diverse_examples) — this function just renders them.
    """
    # WHY: diverse examples matter more than a random sample because CUAD is
    # ~78x imbalanced (see results/data_exploration.json). A random draw of
    # n_shots examples is dominated by high-frequency categories like
    # "Parties", giving the model zero exposure to rare label strings and
    # biasing its few-shot prior toward the common classes it would have
    # guessed anyway — defeating the purpose of showing examples at all.
    shots = examples[:n_shots]
    shot_blocks = []
    for shot in shots:
        shot_blocks.append(
            f'Clause:\n"""\n{shot["clause_text"]}\n"""\nClause type: {shot["clause_type"]}'
        )
    shots_text = "\n\n".join(shot_blocks)

    numbered_types = "\n".join(f"{i + 1}. {ct}" for i, ct in enumerate(clause_types))
    return (
        "Classify the following legal contract clause into exactly one of "
        f"these {len(clause_types)} CUAD clause types:\n\n"
        f"{numbered_types}\n\n"
        f"Here are {len(shots)} example classifications:\n\n"
        f"{shots_text}\n\n"
        "Now classify this clause:\n"
        f'Clause:\n"""\n{clause_text}\n"""\n\n'
        "Respond with ONLY the exact clause type name from the list above, "
        "and nothing else."
    )


def select_diverse_examples(
    pool: list[dict], n_shots: int, seed: int = 42
) -> list[dict]:
    """Sample n_shots examples from `pool` covering as many distinct clause types as possible.

    `pool` is a list of {"clause_text", "clause_type"} dicts. One example is
    picked per distinct clause_type (randomly, but seeded for reproducibility)
    until n_shots types are covered.
    """
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for example in pool:
        by_type.setdefault(example["clause_type"], []).append(example)

    types = list(by_type.keys())
    rng.shuffle(types)
    chosen = []
    for clause_type in types[:n_shots]:
        chosen.append(rng.choice(by_type[clause_type]))
    return chosen


def parse_model_output(output: str, valid_types: list[str]) -> str:
    """Map a model's raw text response onto one of `valid_types`, or "UNKNOWN".

    Tries an exact (case/whitespace-insensitive) match first, then falls back
    to fuzzy string matching for near-miss phrasing.

    # TRADEOFF: fuzzy matching (difflib.get_close_matches) operates on
    # character-level similarity, not legal meaning. CUAD's label vocabulary
    # has many pairs that are textually close but semantically distinct —
    # "Non-Compete" / "Non-Disparagement" / "Non-Transferable License" all
    # share a "Non-" prefix, and "Affiliate License-Licensor" vs "Affiliate
    # License-Licensee" differ by one word. A fuzzy match can silently credit
    # the model for a wrong-but-similar-looking label, inflating accuracy in a
    # way that looks like correctness but isn't. We accept this risk only
    # because the alternative (exact-match only) would count minor
    # capitalization/punctuation drift as failures that have nothing to do
    # with classification quality; the cutoff is kept conservative and
    # unknown_rate is reported honestly so this tradeoff stays visible.
    """
    cleaned = output.strip().strip('"').strip("'").strip(".").strip()
    if not cleaned:
        return _UNKNOWN_LABEL

    normalized_lookup = {ct.lower().strip(): ct for ct in valid_types}
    exact = normalized_lookup.get(cleaned.lower().strip())
    if exact is not None:
        return exact

    close = difflib.get_close_matches(
        cleaned, valid_types, n=1, cutoff=_FUZZY_MATCH_CUTOFF
    )
    if close:
        return close[0]

    return _UNKNOWN_LABEL


@dataclass
class BaselineResult:
    """Container for one baseline evaluation run's metrics."""

    mode: str
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float] = field(default_factory=dict)
    unknown_rate: float = 0.0
    n_samples: int = 0
    duration_seconds: float = 0.0

    def save(self, path: Path | None = None) -> Path:
        """Serialize this result to JSON under results/, appending a timestamp so past runs are never overwritten."""
        if path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = RESULTS_DIR / f"baseline_{self.mode}_{timestamp}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path


def _generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int = 16) -> list[str]:
    """Run one batched greedy-decoding generation pass and return decoded completions.

    # TRADEOFF: batching prompts together (rather than looping one at a time)
    # is not in the original spec but is necessary to make a 200-sample x
    # 2-mode CPU dev run finish in a reasonable time — batched matmuls make
    # far better use of CPU vector units than repeated single-sequence calls.
    """
    import torch

    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    completions = output_ids[:, prompt_len:]
    return [tokenizer.decode(c, skip_special_tokens=True) for c in completions]


def run_baseline_evaluation(
    model,
    tokenizer,
    dataset,
    mode: str,
    n_samples: int = 200,
    seed: int = 42,
    n_shots: int = 3,
    batch_size: int = 8,
) -> BaselineResult:
    """Evaluate `model` on `n_samples` CUAD test clauses in "zero_shot" or "few_shot" mode.

    `dataset` is the DatasetDict returned by clausewise.data.load_cuad() (needs
    both "train", used as the few-shot example pool, and "test", the eval set).
    Automatically saves the result to results/baseline_{mode}_{timestamp}.json.
    """
    if mode not in ("zero_shot", "few_shot"):
        raise ValueError(f"mode must be 'zero_shot' or 'few_shot', got {mode!r}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    clause_types = sorted(set(dataset["train"]["clause_type"]))

    test_split = dataset["test"].shuffle(seed=seed)
    n_samples = min(n_samples, len(test_split))
    test_split = test_split.select(range(n_samples))

    few_shot_examples = []
    if mode == "few_shot":
        train_pool = [
            {"clause_text": t, "clause_type": c}
            for t, c in zip(dataset["train"]["clause_text"], dataset["train"]["clause_type"])
        ]
        few_shot_examples = select_diverse_examples(train_pool, n_shots, seed=seed)

    prompts = []
    true_labels = []
    for example in test_split:
        clause_text = example["clause_text"]
        true_labels.append(example["clause_type"])
        if mode == "zero_shot":
            prompts.append(build_zero_shot_prompt(clause_text, clause_types))
        else:
            prompts.append(
                build_few_shot_prompt(clause_text, clause_types, few_shot_examples, n_shots)
            )

    start = time.time()
    raw_outputs = []
    for batch_start in range(0, len(prompts), batch_size):
        batch = prompts[batch_start : batch_start + batch_size]
        raw_outputs.extend(_generate_batch(model, tokenizer, batch))
    duration_seconds = time.time() - start

    predictions = [parse_model_output(out, clause_types) for out in raw_outputs]

    unknown_rate = sum(1 for p in predictions if p == _UNKNOWN_LABEL) / len(predictions)
    accuracy = accuracy_score(true_labels, predictions)
    # MATH: zero_division=0 avoids a ZeroDivisionWarning/NaN when a class has
    # no predicted or true samples in this subset (expected here — CUAD's
    # rarest classes have single-digit counts in the full test split, so a
    # 200-sample random draw will often miss them entirely).
    macro_f1 = f1_score(
        true_labels, predictions, labels=clause_types, average="macro", zero_division=0
    )
    per_class_f1_scores = f1_score(
        true_labels, predictions, labels=clause_types, average=None, zero_division=0
    )
    per_class_f1 = dict(zip(clause_types, per_class_f1_scores))

    result = BaselineResult(
        mode=mode,
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        per_class_f1={k: float(v) for k, v in per_class_f1.items()},
        unknown_rate=float(unknown_rate),
        n_samples=n_samples,
        duration_seconds=float(duration_seconds),
    )
    result.save()
    return result
