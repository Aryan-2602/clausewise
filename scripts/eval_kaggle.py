"""Phase 4 evaluation run for Clausewise, meant for a Kaggle T4 GPU notebook.

Run this as a Kaggle notebook cell (or `!python scripts/eval_kaggle.py` in a
Kaggle terminal) — it is NOT meant to run on the local CPU dev machine.

Assumes the LoRA adapter from Phase 3 is already at results/adapter/ (either
committed to the repo, or copied into place in this Kaggle session — e.g. from
a prior session's Output tab download). This script does NOT train anything.

# Setup (run once per Kaggle session):
#   !pip install -q datasets transformers peft trl bitsandbytes accelerate scikit-learn pyyaml
#   !git clone https://github.com/Aryan-2602/clausewise.git
#   %cd clausewise
#   !git pull   # if the repo already exists from a previous session
#   # If the adapter isn't already committed to the repo, copy it into place:
#   # !cp -r /kaggle/input/<your-dataset>/adapter results/adapter

Estimated runtime on a T4 (16GB): 15-30 minutes for the full CUAD test set
(1,244 examples) plus the 3-prompt forgetting check.
"""

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_finetuned import main as run_bench_finetuned  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
KAGGLE_WORKING_DIR = Path("/kaggle/working")


def main() -> None:
    """Run the Phase 4 fine-tuned benchmark and copy all results to /kaggle/working/ for download."""
    run_bench_finetuned()

    if not KAGGLE_WORKING_DIR.exists():
        print(f"\n{KAGGLE_WORKING_DIR} not found (not running on Kaggle) — skipping result copy.")
        return

    for pattern in ("eval_finetuned_*.json", "eval_forgetting_*.json", "bench_finetuned_*.json"):
        for result_file in RESULTS_DIR.glob(pattern):
            shutil.copy(result_file, KAGGLE_WORKING_DIR / result_file.name)
    print(f"\nAll Phase 4 result files copied to {KAGGLE_WORKING_DIR} for download from Kaggle's Output tab.")


if __name__ == "__main__":
    main()
