"""Quick QLoRA smoke test for Clausewise, meant for a Kaggle T4 GPU notebook.

Same pipeline as scripts/train_kaggle.py, but capped to a small sample count
and a single epoch so it finishes in ~5-10 minutes on a T4 — run this FIRST
to confirm the pipeline (quantized load, LoRA attach, tokenization/masking,
Trainer loop, adapter save) works end to end before spending 45-90 minutes
on the full run.

# Setup (run once per Kaggle session):
#   !pip install -q datasets transformers peft trl bitsandbytes accelerate scikit-learn pyyaml
#   !git clone https://github.com/Aryan-2602/clausewise.git
#   %cd clausewise
#   !git pull   # if the repo already exists from a previous session
"""

import sys
import tempfile
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from clausewise.train import load_config, run_training  # noqa: E402

BASE_CONFIG_PATH = str(REPO_ROOT / "configs" / "qlora_config.yaml")

# TRADEOFF: these overrides trade statistical validity for turnaround speed —
# 500 train / 100 eval samples and 1 epoch are enough to catch pipeline bugs
# (crashes, shape mismatches, OOM) but the resulting accuracy/loss numbers
# are not meaningful and must never be reported as the real baseline-beating
# result. Only scripts/train_kaggle.py's full run counts for that.
QUICK_OVERRIDES = {
    "data": {"max_train_samples": 500, "max_eval_samples": 100},
    "training": {"num_train_epochs": 1},
}


def _build_quick_config_file() -> str:
    """Write a temp copy of qlora_config.yaml with the quick-smoke-test overrides applied."""
    config = load_config(BASE_CONFIG_PATH)
    for section, overrides in QUICK_OVERRIDES.items():
        config[section].update(overrides)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_quick_qlora_config.yaml", delete=False
    )
    yaml.safe_dump(config, tmp)
    tmp.close()
    return tmp.name


def main() -> None:
    """Run the capped smoke-test training and report pass/fail."""
    quick_config_path = _build_quick_config_file()
    print(f"Running quick smoke test with overrides: {QUICK_OVERRIDES}")
    print(f"(temp config: {quick_config_path})")

    start = time.time()
    adapter_path = run_training(quick_config_path)
    duration_seconds = time.time() - start

    print(f"\nSmoke test finished in {duration_seconds / 60:.1f} minutes.")
    print(f"Adapter saved to: {adapter_path}")
    print(
        "\nPipeline OK. These numbers are NOT the real baseline-beating result "
        "(only 500/100 samples, 1 epoch) — run scripts/train_kaggle.py for that."
    )


if __name__ == "__main__":
    main()
