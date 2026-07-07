# Clausewise — Legal Contract Clause Classifier

## What This Is

Clausewise fine-tunes Qwen2.5-0.5B-Instruct with QLoRA to classify legal
contract clauses into 41 CUAD categories. Every number in this README is
measured and saved as a JSON file under `results/` — nothing here is a
theoretical or back-of-envelope estimate. The full pipeline (baseline →
fine-tuning → evaluation) was benchmarked end-to-end on a Tesla T4 GPU
(Kaggle).

## The Task

- **Input**: a contract clause (an extracted span from a legal contract)
- **Output**: one of 41 CUAD clause types (e.g. "Governing Law", "Parties",
  "Termination For Convenience")
- **Dataset**: CUAD — 510 contracts, 41 clause types, contract-disjoint
  train/test split (`results/data_exploration.json`)
- **Key challenge**: ~78x class imbalance — the most common class,
  "Parties" (2,113 examples across train+test), outnumbers the rarest,
  "Price Restrictions" (27 examples), by a factor of ~78
  (`results/data_exploration.json` → `class_imbalance_ratio`)

## Results

### Main Benchmark (Tesla T4 GPU, Qwen2.5-0.5B)

| Metric | Zero-Shot | 3-Shot | Fine-Tuned |
|---|---|---|---|
| Accuracy | 6.50% | 14.00% | **75.24%** |
| Macro F1 | 0.0335 | 0.0860 | **0.6076** |
| Unknown Rate | 12.5% | 5.0% | **0.0%** |

Sources: `results/bench_baseline_20260706_024308.json` (zero-shot/3-shot,
200 sampled test clauses), `results/bench_finetuned_20260707_000616.json`
and `results/eval_finetuned_20260707_000603.json` (fine-tuned, **full**
1,244-example test set — see the note on sample sizes below).

**Why zero-shot is so low.** A 0.5B-parameter instruct model, given a
41-way label list and zero legal domain training, has no reliable way to
map a bare contract clause onto CUAD's specific label vocabulary
("Rofr/Rofo/Rofn", "Affiliate License-Licensor", etc.) — most of these
aren't things the model would recognize from general pretraining alone.

**Why 3-shot roughly doubles accuracy.** The jump (6.5% → 14.0%) mostly
reflects the model learning the *response format* from the three examples
(what a valid label looks like, that it should answer with only the label)
rather than acquiring legal understanding — the unknown rate drop (12.5% →
5.0%) is consistent with this: examples mostly fixed *format* compliance, not
classification accuracy.

**Why fine-tuning reaches 75.2% / 0.608 macro F1.** QLoRA updates actual
task-specific weights (540,672 trainable parameters — see "Why QLoRA"
below) instead of relying on in-context pattern matching, and
`oversample_minority_classes()` (`clausewise/data.py`) ensures every class
had at least 50 training examples despite the 78x imbalance, so the model
had a real chance to learn rare labels rather than defaulting to common
ones. This directly and convincingly beats the 14.0% / 0.0860 baseline.

**Why the unknown rate is 0.0%.** After instruction fine-tuning on the
exact `### Instruction / ### Input / ### Response` template, the model
reliably outputs a label-shaped string every time — `parse_model_output()`
never needs to fall back to "UNKNOWN" because there's simply no
unparseable output to fall back from. This is a format-compliance result,
not a correctness guarantee (a wrong-but-valid-looking label still counts
toward accuracy/F1 the normal way).

**Note on sample sizes.** The baseline numbers (zero-shot/3-shot) were
measured on 200 randomly sampled test clauses (a deliberate CPU-time
tradeoff — see Phase 2). The fine-tuned numbers are measured on the
**full** 1,244-example test set. This makes the fine-tuned number *more*
reliable, not less, but it means the two aren't from an identically-sized
sample — worth knowing when reading the table.

### Per-Class Highlights

Computed as fine-tuned F1 − 3-shot F1, from `results/eval_finetuned_20260707_000603.json`
and `results/bench_baseline_20260706_024308.json`.

**Top 5 most improved:**

| Clause Type | 3-Shot F1 | Fine-Tuned F1 | Δ |
|---|---|---|---|
| Source Code Escrow | 0.0000 | 1.0000 | +1.000 |
| Document Name | 0.0000 | 0.9901 | +0.990 |
| Parties | 0.0000 | 0.9569 | +0.957 |
| Insurance | 0.0000 | 0.9538 | +0.954 |
| Audit Rights | 0.0000 | 0.9091 | +0.909 |

(Source Code Escrow's perfect score is measured on just 1 test example —
a real number, but not a statistically meaningful one on its own.)

**Top 5 least improved / regressed:**

| Clause Type | 3-Shot F1 | Fine-Tuned F1 | Δ | Test support |
|---|---|---|---|---|
| Affiliate License-Licensee | 0.0000 | 0.0000 | 0.000 | 12 |
| Affiliate License-Licensor | 0.0000 | 0.0000 | 0.000 | 6 |
| Price Restrictions | 0.0000 | 0.0000 | 0.000 | 0 |
| Non-Disparagement | 0.8000 | 0.7500 | -0.050 | 7 |
| Effective Date | 0.4878 | 0.3404 | **-0.147** | 70 |

- **Price Restrictions has zero examples in the test split.** `zero_division=0`
  makes F1 mathematically 0 regardless of what the model predicts — this
  is a data-split artifact, not a model failure, and it's the only one of
  the three zero-F1 classes where that's strictly true.
- **Affiliate License-Licensee (12 test examples)** is a genuine model
  failure, not just a data artifact: the confusion matrix
  (`results/eval_finetuned_20260707_000603.json`) shows 7 of these
  misclassified as "License Grant" — a plausible confusion, since
  affiliate license clauses often *are* embedded inside a broader license
  grant.
- **Affiliate License-Licensor (6 test examples)** has too little test
  support to draw a strong conclusion either way.
- **Effective Date regressed -0.147** despite having solid test support
  (70 examples) — this is the most interesting real regression. The
  confusion matrix shows 44 "Effective Date" clauses predicted as
  "Agreement Date" and 9 as "Expiration Date" (with 7 of the reverse,
  "Agreement Date" → "Effective Date"). These three clause types often
  share near-identical surface text (a single date near the top of a
  contract) — in 3-shot prompting, having the actual clause type name
  visible in the prompt's example list may have nudged the model toward
  the right guess via a lucky lexical association, whereas fine-tuning on
  short, decontextualized spans seems to have blurred the boundary between
  these three date-adjacent categories rather than sharpening it.

### Forgetting Check

From `results/bench_finetuned_20260707_000616.json` (`forgetting_evaluation`).

| Task | Base | Fine-Tuned | Delta |
|---|---|---|---|
| Prime numbers | 0 | 1 | +1 |
| Capital of France | 1 | 1 | 0 |
| Apple arithmetic | 1 | 1 | 0 |

**This is a 3-prompt sanity check, not a general capability benchmark.**
On these three probes, fine-tuning showed no detectable forgetting —
if anything, the fine-tuned model answered the prime-numbers prompt
correctly where the base model didn't, which is more likely prompt-level
noise (greedy decoding on a 0.5B model is not highly stable run to run)
than a genuine capability gain from clause-classification fine-tuning.
A real forgetting evaluation would run something like MMLU or a held-out
instruction-following suite before and after fine-tuning; three fixed
prompts can only catch gross regressions, not subtle capability loss.

## Why QLoRA

**Memory math.** Full fine-tuning of Qwen2.5-0.5B (494M parameters) would
need to hold weights + gradients + Adam optimizer states (2 extra copies)
in memory — roughly 4 bytes × 494M × ~4 ≈ 8GB just for that, before
activations, on top of a 16GB T4 shared with everything else in the Kaggle
environment. QLoRA instead keeps the base model frozen and quantized to
4-bit (NF4), and only trains small LoRA adapter matrices in BF16 — the
frozen base needs no gradient or optimizer memory at all.

**Trainable parameters: 540,672 / 494,573,440 ≈ 0.1093%** — measured
directly via `prepare_model_for_qlora()`'s parameter count on the real
model, confirming this is genuinely parameter-efficient fine-tuning, not
full fine-tuning wearing a QLoRA label. This is comfortably small enough
to fit on a T4 16GB at `per_device_train_batch_size=4`
(`configs/qlora_config.yaml`).

**NF4 vs INT4.** Pretrained transformer weights are empirically close to
normally distributed — most weight mass clusters near zero, with a long
thin tail. A plain INT4 grid spaces its quantization buckets uniformly
across the weight range, wasting resolution on the sparse tails and
under-resolving the dense center. NF4 (4-bit NormalFloat) instead spaces
its quantization levels to match a normal distribution's quantiles, so
more of the available 4 bits go where the weights actually are. Double
quantization then additionally quantizes the per-block scaling constants
themselves (normally kept in fp32), trimming roughly another ~0.4
bits/parameter with negligible added error — see the `# WHY` comments in
`clausewise/train.py`'s `setup_quantization()`.

**Rank tradeoff.** `r=8` was chosen as a starting point balancing two
failure modes: too small a rank (e.g. r=1-2) may not have enough capacity
to separate 41 classes cleanly, while too large a rank risks overfitting
given only 510 source contracts (even after oversampling rare classes to
a floor of 50). r=8 with `q_proj`/`v_proj` as target modules is the
standard starting configuration for instruction fine-tuning and, per the
results above, was sufficient to reach 75.2% accuracy without any
rank tuning sweep — a real avenue for future improvement, not yet
explored here.

## Architecture & Pipeline

### Data Pipeline (`clausewise/data.py`)

- CUAD on HuggingFace is a SQuAD-style extractive QA format (context +
  question + answer spans), not a flat classification table — flattening
  was required: keep only QA rows with a non-empty answer, and parse the
  clause type out of the question template (`_extract_category()`).
- 11,270 of 22,450 raw QA rows were dropped as empty-answer rows (a
  category simply not present in that contract) — these have no text to
  classify and are not treated as a 42nd "None" class
  (`results/data_exploration.json`).
- `oversample_minority_classes()` duplicates (with replacement) any
  class below 50 training examples, floor set in
  `configs/qlora_config.yaml`'s `data.min_samples_per_class` — this
  addresses the 78x imbalance directly in the training data rather than
  in the loss function (see "Key Engineering Decisions" below).
- Instruction format: `### Instruction: / ### Input: / ### Response:`,
  shared identically between training and evaluation via
  `build_prompt_prefix()`.

### Training (`clausewise/train.py`)

- QLoRA config: `r=8`, `alpha=16`, `dropout=0.05`, target modules
  `q_proj`/`v_proj` (`configs/qlora_config.yaml`)
- 3 epochs, `lr=2e-4` cosine schedule, effective batch size 16
  (4 per device × 4 gradient accumulation steps)
- Training results (`results/training_20260706_225019.json`):

  | Epoch | Train Loss | Eval Loss | Eval Token Accuracy |
  |---|---|---|---|
  | 1 | 0.2420 | 0.2942 | 91.83% |
  | 2 | 0.1844 | 0.1887 | 94.08% |
  | 3 | 0.1750 | 0.1839 | 94.04% |

  (`Eval Token Accuracy` is next-token prediction accuracy on the masked
  output span during training — a different metric than the exact-match
  clause-type accuracy reported in the Results section above; see the
  `# WHY` comment in `scripts/train_kaggle.py` for why the two aren't
  directly comparable.)
- Full 3-epoch run measured ~107 minutes on a T4 (`train_runtime` in
  `results/training_20260706_225019.json`).
- Adapter saved to `results/adapter/` (~13MB: `adapter_model.safetensors`
  + tokenizer files).

### Evaluation (`clausewise/evaluate.py`)

- Uses the exact same `### Instruction / ### Input / ### Response` prompt
  template as training, via the shared `build_prompt_prefix()` — a
  template mismatch between training and eval is one of the most common
  silent failure modes in fine-tuning evaluation (a model can look like
  it's failing when it's actually just being asked in an unfamiliar format).
- Falls back to fuzzy string matching (`difflib`) when the model's raw
  output isn't an exact label match, before giving up and returning
  "UNKNOWN".
- Full 1,244-example test set evaluated in ~6.5 minutes on a T4
  (`duration_seconds` in `results/eval_finetuned_20260707_000603.json`).

## Key Engineering Decisions

1. **Oversampling vs. loss weighting.** `get_class_weights()` computes
   per-class weights, but SFTTrainer's loss is token-level cross-entropy
   computed after `labels` are detached from which example (and thus which
   class) produced them — there's no clean per-example class hook without
   writing a custom `Trainer` subclass that overrides `compute_loss`.
   Oversampling training data (duplicating rare-class rows before
   tokenization) sidesteps this entirely and is simpler to reason about.
   It produced strong results in practice — e.g. "Parties" went from
   0.0 F1 (3-shot) to 0.957 F1 (fine-tuned).

2. **Shared prompt template.** `build_prompt_prefix()`
   (`clausewise/train.py`) is the single source of truth for the
   instruction template, used by both `format_training_example()`
   (training) and `predict_clause_type()` (evaluation). This was
   refactored out of two separately-maintained copies during Phase 4
   design specifically because a silent template mismatch between
   training and eval is a well-known way to destroy fine-tuning results
   without any obvious error — fixed proactively, not after hitting it.

3. **CUAD format discovery.** The canonical `load_dataset("cuad")` loader
   is broken under `datasets>=4.0` (which dropped HF "loading script"
   support entirely). Used the community mirror
   `theatticusproject/cuad-qa` instead, with manual flattening from its
   SQuAD-style QA format into clause classification rows (see "Data
   Pipeline" above and `results/data_exploration.json`'s `findings`).

4. **Device-mismatch fix.** `_generate_single()`/`_generate_chat_single()`
   (`clausewise/evaluate.py`) originally tokenized inputs without moving
   them onto the model's device — tokenizer output starts on CPU, but
   `load_adapter()`'s `device_map="auto"` places the model on `cuda:0` on
   Kaggle's T4, so `model.generate()` raised a `RuntimeError` from mixing
   CPU and CUDA tensors. Fixed by reading `next(model.parameters()).device`
   at call time (works unchanged on both the CPU dev machine and the GPU),
   rather than hardcoding `"cuda"`.

## Limitations & Honest Findings

- **Three classes still at F1=0.0.** One ("Price Restrictions") has zero
  test examples, making F1=0 a data-split artifact rather than a real
  model failure. The other two ("Affiliate License-Licensee",
  "Affiliate License-Licensor") have low but nonzero test support (12 and
  6 examples) — "Affiliate License-Licensee" shows a genuine, explainable
  confusion with "License Grant" in the confusion matrix; "Affiliate
  License-Licensor" has too little test data to draw a firm conclusion.
- **"Effective Date" regressed -0.147** from the 3-shot baseline despite
  having solid test support (70 examples) — plausibly because it shares
  near-identical surface text (a bare date) with "Agreement Date" and
  "Expiration Date", and fine-tuning on short decontextualized spans may
  have blurred rather than sharpened that boundary. See "Per-Class
  Highlights" above for the confusion-matrix evidence.
- **The forgetting check is 3 fixed prompts, not a rigorous general
  capability benchmark.** It can catch gross regressions but not subtle
  capability loss — a real forgetting evaluation needs a proper benchmark
  suite (e.g. MMLU) run before and after fine-tuning.
- **0.5B is a small model.** These results are strong for this specific,
  narrow 41-way classification task on short extracted spans; they say
  nothing about whether this approach would generalize to more complex
  multi-clause reasoning or full-contract review without a larger base
  model.
- **Class weights are computed but not wired into the training loss** —
  `get_class_weights()` exists in `clausewise/data.py` but is unused by
  `build_trainer()`, since token-level LM loss has no clean per-example
  class hook (see "Key Engineering Decisions" above). Oversampling was
  used instead. A custom weighted-loss `Trainer` subclass is flagged as
  real future work, not implemented here.
- **Baseline and fine-tuned accuracy were measured on different sample
  sizes** (200 vs. the full 1,244-example test set) — see the note under
  "Main Benchmark" above.

## Reproducing Results

### Local (CPU — tests and data pipeline only)

```bash
git clone https://github.com/Aryan-2602/clausewise
cd clausewise
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest tests/ -v  # 30/30 tests
```

### Training (Kaggle T4 GPU)

```bash
# Smoke test first (~5-10 min, 500 train / 100 eval samples, 1 epoch):
python scripts/quick_train_kaggle.py
# Full training (~107 min measured, 3 epochs, full dataset):
python scripts/train_kaggle.py
```

### Evaluation (Kaggle T4 GPU)

```bash
python scripts/eval_kaggle.py
```

Assumes the adapter from training is already at `results/adapter/`
(committed to this repo) — this script does not train anything.

## Test Coverage

**30/30 tests passing** across the data pipeline, baseline evaluation,
training pipeline, and evaluation modules. All tests run on CPU with no
GPU, no quantized weights, and no network access beyond one deliberate
`load_cuad()` integration test.

- `tests/test_data.py` (9 tests) — CUAD loading/flattening, clause text
  cleaning, instruction formatting, tokenization + max-length truncation,
  class weight computation, and minority-class oversampling.
- `tests/test_baseline.py` (7 tests) — zero-shot/few-shot prompt
  construction, fuzzy-match output parsing, and baseline evaluation
  metric ranges (with generation mocked out).
- `tests/test_train.py` (7 tests) — config validation, instruction
  formatting with loss masking (instruction/input tokens masked to -100),
  LoRA trainable-parameter percentage, and a real (unmocked)
  `build_trainer` → `Trainer.evaluate()` integration test on
  variable-length sequences — a regression test for a real crash hit on
  Kaggle (SFTTrainer's default `chunked_nll` loss never materializes full
  logits, which broke a naively-written custom `compute_metrics`).
- `tests/test_evaluate.py` (7 tests) — clause-type prediction and
  response-marker parsing, evaluation metric ranges, results JSON
  persistence, forgetting-check score validity, and two regression tests
  for a device-mismatch bug (CPU tokenizer output vs. a GPU-placed model)
  using a real, unmocked tiny model.
