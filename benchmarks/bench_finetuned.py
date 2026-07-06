"""Phase 4 benchmark: fine-tuned adapter vs Phase 2's zero-shot/3-shot baseline on CUAD.

Run with: .venv/bin/python benchmarks/bench_finetuned.py

# TRADEOFF: this loads the adapter in 4-bit (same quantization as training)
# and generates on GPU — like bench_baseline.py, it is not meant to run on
# the local CPU dev machine. See scripts/eval_kaggle.py for the Kaggle T4
# entry point.
"""

import json
import time
from pathlib import Path

from clausewise.data import load_cuad
from clausewise.evaluate import load_adapter, run_evaluation, run_forgetting_evaluation
from clausewise.train import load_config

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "qlora_config.yaml")
ADAPTER_PATH = str(Path(__file__).resolve().parent.parent / "results" / "adapter")


def _load_latest_baseline() -> dict:
    """Load the most recent results/bench_baseline_*.json (Phase 2's zero-shot/3-shot numbers)."""
    candidates = sorted(RESULTS_DIR.glob("bench_baseline_*.json"))
    if not candidates:
        raise FileNotFoundError(
            "No results/bench_baseline_*.json found — run benchmarks/bench_baseline.py (Phase 2) first."
        )
    with open(candidates[-1]) as f:
        return json.load(f)


def _print_comparison_table(baseline: dict, finetuned) -> None:
    """Print the Zero-Shot / 3-Shot / Fine-Tuned comparison table."""
    zero_shot = baseline["zero_shot"]
    few_shot = baseline["few_shot"]

    print("=" * 64)
    print(f"{'Metric':<15} | {'Zero-Shot':>9} | {'3-Shot':>7} | {'Fine-Tuned':>10} | {'Delta':>10}")
    print("-" * 64)
    rows = [
        ("Accuracy", zero_shot["accuracy"], few_shot["accuracy"], finetuned.accuracy),
        ("Macro F1", zero_shot["macro_f1"], few_shot["macro_f1"], finetuned.macro_f1),
        ("Unknown Rate", zero_shot["unknown_rate"], few_shot["unknown_rate"], finetuned.unknown_rate),
    ]
    for name, zs_val, fs_val, ft_val in rows:
        delta = ft_val - fs_val
        print(f"{name:<15} | {zs_val:9.4f} | {fs_val:7.4f} | {ft_val:10.4f} | {delta:+10.4f}")
    print("=" * 64)


def _print_top_bottom_classes(finetuned, few_shot_per_class_f1: dict) -> None:
    """Print the top 5 most improved and least improved/regressed clause types by F1 delta vs 3-shot."""
    deltas = {
        clause_type: finetuned.per_class_f1.get(clause_type, 0.0) - few_shot_per_class_f1.get(clause_type, 0.0)
        for clause_type in finetuned.per_class_f1
    }
    sorted_deltas = sorted(deltas.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 5 most improved clause types (fine-tuned F1 - 3-shot F1):")
    for clause_type, delta in sorted_deltas[:5]:
        print(f"  {delta:+.4f}  {clause_type}")

    print("\nTop 5 least improved / regressed clause types:")
    for clause_type, delta in sorted_deltas[-5:]:
        print(f"  {delta:+.4f}  {clause_type}")


def _print_forgetting_table(forgetting) -> None:
    """Print the Base / Fine-Tuned / Delta forgetting check table."""
    task_labels = {
        "prime_numbers": "Prime numbers",
        "capital_of_france": "Capital of France",
        "apple_arithmetic": "Apple arithmetic",
    }
    print("\nForgetting check:")
    print(f"{'Task':<24} | {'Base':>4} | {'Fine-Tuned':>10} | {'Delta':>5}")
    for task_name, label in task_labels.items():
        base = forgetting.base_scores[task_name]
        finetuned_score = forgetting.finetuned_scores[task_name]
        delta = forgetting.forgetting_delta[task_name]
        print(f"{label:<24} | {base:>4} | {finetuned_score:>10} | {delta:>+5}")
    print(
        "\nNOTE: this is a simple 3-prompt sanity check, not a general capability "
        "benchmark — see clausewise/evaluate.py's module docstring for the limitation."
    )


def main() -> None:
    """Load the fine-tuned adapter, run full evaluation + forgetting check, print/save the comparison."""
    config = load_config(CONFIG_PATH)
    baseline = _load_latest_baseline()

    print(f"Loading adapter from {ADAPTER_PATH}...")
    model, tokenizer = load_adapter(config["model"]["name"], ADAPTER_PATH, config)

    print("Loading CUAD dataset...")
    dataset = load_cuad()

    print("Running full test-set evaluation (this may take a while)...")
    finetuned = run_evaluation(model, tokenizer, dataset, n_samples=None, seed=42)

    print("Running forgetting evaluation...")
    forgetting = run_forgetting_evaluation(model, tokenizer)

    _print_comparison_table(baseline, finetuned)
    _print_top_bottom_classes(finetuned, baseline["few_shot"]["per_class_f1"])
    _print_forgetting_table(forgetting)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    combined_path = RESULTS_DIR / f"bench_finetuned_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(
            {
                "adapter_path": ADAPTER_PATH,
                "baseline_source": str(sorted(RESULTS_DIR.glob("bench_baseline_*.json"))[-1]),
                "finetuned_evaluation": finetuned.__dict__,
                "forgetting_evaluation": forgetting.__dict__,
            },
            f,
            indent=2,
        )
    print(f"\nSaved combined benchmark results to {combined_path}")


if __name__ == "__main__":
    main()
