"""Full QLoRA fine-tuning run for Clausewise, meant for a Kaggle T4 GPU notebook.

Run this as a Kaggle notebook cell (or `!python scripts/train_kaggle.py` in a
Kaggle terminal) — it is NOT meant to run on the local CPU dev machine.

# Setup (run once per Kaggle session, e.g. as a notebook cell or shell commands):
#   !pip install -q datasets transformers peft trl bitsandbytes accelerate scikit-learn pyyaml
#   !git clone https://github.com/Aryan-2602/clausewise.git
#   %cd clausewise
#   !git pull   # if the repo already exists from a previous session

Estimated runtime on a T4 (16GB): 45-90 minutes for 3 epochs on the full CUAD
train split (configs/qlora_config.yaml's defaults — full dataset, no sample cap).
"""

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from clausewise.train import run_training  # noqa: E402

CONFIG_PATH = str(REPO_ROOT / "configs" / "qlora_config.yaml")
ADAPTER_DOWNLOAD_DIR = REPO_ROOT / "results" / "adapter"


def _print_metrics_table(log_history: list[dict]) -> None:
    """Print an Epoch | Train Loss | Eval Loss | Eval Accuracy table from Trainer's log history.

    # WHY: Trainer logs training-loss and eval-metric dicts as separate
    # entries in log_history (a train-loss entry has no "eval_loss" key and
    # vice versa) — we merge them by nearest epoch so each row of the table
    # reflects one epoch's train + eval numbers together.
    """
    by_epoch: dict[float, dict] = {}
    for entry in log_history:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        row = by_epoch.setdefault(round(epoch), {})
        if "loss" in entry:
            row["train_loss"] = entry["loss"]
        if "eval_loss" in entry:
            row["eval_loss"] = entry["eval_loss"]
        if "eval_accuracy" in entry:
            row["eval_accuracy"] = entry["eval_accuracy"]

    def _fmt(value: float | None, width: int) -> str:
        return f"{value:>{width}.4f}" if value is not None else f"{'—':>{width}}"

    print(f"{'Epoch':<7} | {'Train Loss':>10} | {'Eval Loss':>10} | {'Eval Accuracy':>13}")
    print("-" * 52)
    for epoch in sorted(by_epoch):
        row = by_epoch[epoch]
        train_loss = _fmt(row.get("train_loss"), 10)
        eval_loss = _fmt(row.get("eval_loss"), 10)
        eval_acc = _fmt(row.get("eval_accuracy"), 13)
        print(f"{epoch:<7} | {train_loss} | {eval_loss} | {eval_acc}")


def main() -> None:
    """Run full QLoRA training and print the final metrics table."""
    print(f"Starting full QLoRA training with config: {CONFIG_PATH}")
    start = time.time()

    adapter_path = run_training(CONFIG_PATH)

    duration_minutes = (time.time() - start) / 60
    print(f"\nTraining finished in {duration_minutes:.1f} minutes.")
    print(f"Adapter saved to: {adapter_path}")

    ADAPTER_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copytree(adapter_path, ADAPTER_DOWNLOAD_DIR, dirs_exist_ok=True)
    print(f"Adapter also copied to {ADAPTER_DOWNLOAD_DIR} for download from Kaggle's Output tab.")

    results_dir = REPO_ROOT / "results"
    training_jsons = sorted(results_dir.glob("training_*.json"))
    if training_jsons:
        with open(training_jsons[-1]) as f:
            metrics = json.load(f)
        print("\nFinal training metrics:")
        _print_metrics_table(metrics["log_history"])


if __name__ == "__main__":
    main()
