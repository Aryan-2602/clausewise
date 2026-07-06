"""Phase 2 baseline benchmark: zero-shot vs few-shot Qwen2.5-0.5B-Instruct on CUAD.

Run with: .venv/bin/python benchmarks/bench_baseline.py

# TRADEOFF: this runs on CPU for dev speed/cost. Absolute wall-clock numbers
# (duration_seconds) are not meaningful outside this machine; accuracy/F1 are
# what carry over. Per CLAUDE.md, this gets rerun on the Kaggle T4 GPU before
# Phase 3 training for the numbers that actually gate the fine-tuning decision.
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from clausewise.baseline import BaselineResult, run_baseline_evaluation
from clausewise.data import load_cuad

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
N_SAMPLES = 200
SEED = 42
N_SHOTS = 3
BATCH_SIZE = 16  # WHY: batching gives ~5% throughput gain on CPU matmuls; kept modest since prompts are long (41-line label list).


def _print_comparison_table(zero_shot: BaselineResult, few_shot: BaselineResult) -> None:
    """Print the zero-shot vs few-shot metric comparison table."""
    print("=" * 55)
    print(f"{'Metric':<15} | {'Zero-Shot':>9} | {'3-Shot':>9} | {'Delta':>9}")
    print("-" * 55)
    rows = [
        ("Accuracy", zero_shot.accuracy, few_shot.accuracy),
        ("Macro F1", zero_shot.macro_f1, few_shot.macro_f1),
        ("Unknown Rate", zero_shot.unknown_rate, few_shot.unknown_rate),
        ("Duration (s)", zero_shot.duration_seconds, few_shot.duration_seconds),
    ]
    for name, zs_val, fs_val in rows:
        delta = fs_val - zs_val
        print(f"{name:<15} | {zs_val:9.4f} | {fs_val:9.4f} | {delta:+9.4f}")
    print("=" * 55)


def _print_top_bottom_classes(result: BaselineResult, dataset, n_samples: int, seed: int, label: str) -> None:
    """Print the top 5 best and worst classified clause types by F1, restricted to classes with test support.

    # WHY: with 41 classes and only n_samples test rows drawn from a ~78x
    # imbalanced pool (see results/data_exploration.json), many rare classes
    # have zero true occurrences in the sample. Under zero_division=0 those
    # classes report F1=0 without ever being wrongly predicted — including
    # them in a "worst classified" ranking would be misleading, since it's
    # sampling absence, not classification failure. We filter to classes that
    # actually appeared in this evaluation's true labels.
    """
    test_sample = dataset["test"].shuffle(seed=seed).select(range(min(n_samples, len(dataset["test"]))))
    present_classes = set(test_sample["clause_type"])
    scored = [(ct, f1) for ct, f1 in result.per_class_f1.items() if ct in present_classes]
    scored.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop 5 best classified clause types ({label}, by F1, among classes present in sample):")
    for clause_type, f1 in scored[:5]:
        print(f"  {f1:.4f}  {clause_type}")

    print(f"\nTop 5 worst classified clause types ({label}, by F1, among classes present in sample):")
    for clause_type, f1 in scored[-5:]:
        print(f"  {f1:.4f}  {clause_type}")


def main() -> None:
    """Run zero-shot and few-shot baseline evaluation and print/save the comparison."""
    torch.set_num_threads(8)
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    print("Loading CUAD dataset...")
    dataset = load_cuad()

    print(f"Running zero-shot evaluation on {N_SAMPLES} samples...")
    zero_shot = run_baseline_evaluation(
        model, tokenizer, dataset, mode="zero_shot", n_samples=N_SAMPLES, seed=SEED, batch_size=BATCH_SIZE
    )

    print(f"Running {N_SHOTS}-shot evaluation on {N_SAMPLES} samples...")
    few_shot = run_baseline_evaluation(
        model, tokenizer, dataset, mode="few_shot", n_samples=N_SAMPLES, seed=SEED,
        n_shots=N_SHOTS, batch_size=BATCH_SIZE,
    )

    _print_comparison_table(zero_shot, few_shot)
    _print_top_bottom_classes(zero_shot, dataset, N_SAMPLES, SEED, "zero-shot")
    _print_top_bottom_classes(few_shot, dataset, N_SAMPLES, SEED, f"{N_SHOTS}-shot")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    combined_path = RESULTS_DIR / f"bench_baseline_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "n_samples": N_SAMPLES,
                "seed": SEED,
                "n_shots": N_SHOTS,
                "zero_shot": zero_shot.__dict__,
                "few_shot": few_shot.__dict__,
            },
            f,
            indent=2,
        )
    print(f"\nSaved combined benchmark results to {combined_path}")


if __name__ == "__main__":
    main()
