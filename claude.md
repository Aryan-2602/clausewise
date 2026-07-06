# CLAUSEWISE — Legal Contract Clause Classifier

## Project Overview
Clausewise fine-tunes small LLMs on CUAD (Contract Understanding Atticus Dataset)
using QLoRA to classify legal contract clauses into 41 clause types.
Every accuracy and F1 claim must be measured and reproducible. No theoretical numbers.

## Goals
- Phase 1: Data pipeline — load, clean, and format CUAD for instruction fine-tuning
- Phase 2: Baseline evaluation — measure zero-shot and few-shot performance before
  any fine-tuning (this is the control we beat)
- Phase 3: QLoRA fine-tuning — fine-tune Qwen2.5-0.5B on CUAD clause classification
- Phase 4: Evaluation — measure accuracy, F1 per class, and general capability
  degradation (catastrophic forgetting check)
- Phase 5: Validation — repeat key metrics on Llama-3.2-1B to show approach scales
- Stretch: Serve fine-tuned model through Inferno inference engine

## The Task
Input: a contract clause (paragraph of text from a legal contract)
Output: clause type from 41 CUAD categories
  e.g. "Governing Law", "Termination for Convenience", "Indemnification"

Evaluation metrics:
- Accuracy (overall)
- Macro F1 (handles class imbalance across 41 clause types)
- Per-class F1 for top 10 most common clause types
- Forgetting metric: general benchmark score before vs after fine-tuning

## Environment
- Python 3.12
- Virtual environment: venv
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Dev model: Qwen/Qwen2.5-0.5B-Instruct
- Validation model: meta-llama/Llama-3.2-1B-Instruct
- Dataset: CUAD (via HuggingFace datasets)
- Training: Kaggle T4 GPU (16GB VRAM)

## Project Structure
clausewise/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── clausewise/
│   ├── __init__.py
│   ├── data.py          # CUAD loading, cleaning, formatting as instructions
│   ├── baseline.py      # Zero-shot and few-shot evaluation before fine-tuning
│   ├── train.py         # QLoRA fine-tuning pipeline
│   ├── evaluate.py      # Accuracy, F1, forgetting metrics
│   └── utils.py         # Formatting helpers, logging, result saving
├── configs/
│   └── qlora_config.yaml  # All hyperparameters in one place
├── benchmarks/
│   ├── bench_baseline.py   # Zero-shot vs few-shot comparison
│   └── bench_finetuned.py  # Fine-tuned vs baseline comparison table
├── tests/
│   ├── test_data.py        # Data pipeline correctness tests
│   ├── test_baseline.py    # Baseline evaluation sanity checks
│   └── test_evaluate.py    # Metric computation correctness tests
└── results/
    └── (all benchmark outputs as JSON with timestamps)

## Coding Standards
- Type hints on every function signature
- Docstring on every function explaining what it does and why
- No magic numbers — all hyperparameters in configs/qlora_config.yaml
- Every benchmark script saves results to results/ as JSON before printing
- Never delete a benchmark result — append with timestamp
- Add # TRADEOFF: comments on every key design decision
- Add # WHY: comments explaining non-obvious implementation choices

## QLoRA Configuration (starting point — tune based on results)
- rank r: 8
- alpha: 16 (alpha = 2r is standard starting point)
- dropout: 0.05
- target modules: q_proj, v_proj (standard for instruction fine-tuning)
- learning rate: 2e-4
- epochs: 3
- batch size: 4 (gradient accumulation steps: 4 for effective batch of 16)
- quantization: 4-bit NF4 with double quantization
- optimizer: paged_adamw_8bit

## Testing Rules
- Write tests alongside each module, not after
- Tests must run on CPU with no GPU required
- Each test must have a clear docstring explaining what correctness
  property it checks
- Run tests with: pytest tests/ -v

## Benchmarking Rules
- Always measure: accuracy, macro F1, and per-class F1 for top 10 classes
- Always run baseline and fine-tuned in the same script for fair comparison
- Print a comparison table at the end of every benchmark run
- Save raw results as JSON to results/ with timestamp

## Evaluation Philosophy
- Report forgetting: run a general benchmark before AND after fine-tuning
- If fine-tuned model underperforms baseline on any metric, report it honestly
- The goal is a rigorous benchmark, not inflated numbers

## Claude Code Behavior
- Read and understand the relevant module fully before editing it
- Write tests for each function before or immediately after implementing it
- After completing any phase, print a summary of what was built and
  what the tests cover
- Never skip a test because it is hard to write — flag it and explain why
- When implementing math (F1, quantization), add a # MATH: comment
- If a design decision has a tradeoff, add a # TRADEOFF: comment
- All hyperparameters must live in configs/qlora_config.yaml,
  never hardcoded
