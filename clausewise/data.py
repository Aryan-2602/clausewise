"""CUAD data pipeline: load, clean, and format contract clauses for instruction fine-tuning.

FINDING: CUAD on the HuggingFace Hub is not a flat (clause_text, clause_type) table.
It is a SQuAD-style extractive QA dataset — each row pairs a full contract (context)
with a category-specific question ("Highlight the parts related to X...") and an
answers field holding 0 or 1 extracted spans. To get classification examples we
flatten: keep only rows with a nonempty answer, and pair (answer_text, category)
where category is parsed out of the question string. See results/data_exploration.json
for the full exploration and additional FINDINGs.
"""

import re
import unicodedata
from collections import Counter

from datasets import Dataset, DatasetDict, load_dataset

# WHY: the canonical load_dataset("cuad") loader relies on a HF "loading script",
# and datasets>=4.0 removed script execution support entirely. This community
# mirror ships the same data pre-converted to parquet, so it works with modern
# `datasets` versions without needing trust_remote_code.
_CUAD_HUB_REPO = "theatticusproject/cuad-qa"
_CUAD_REVISION = "refs/convert/parquet"

_CATEGORY_PATTERN = re.compile(r'related to "(.+?)"')

INSTRUCTION = (
    "Classify the following legal contract clause into one of the 41 "
    "CUAD clause types."
)


def _extract_category(question: str) -> str:
    """Pull the clause-type name out of a CUAD question string.

    CUAD questions follow the fixed template
    'Highlight the parts (if any) of this contract related to "<category>" ...',
    so the category is always the quoted substring after "related to".
    """
    match = _CATEGORY_PATTERN.search(question)
    if match is None:
        raise ValueError(f"Could not extract clause category from question: {question!r}")
    return match.group(1)


def _flatten_qa_split(qa_split: Dataset) -> Dataset:
    """Convert one CUAD QA split into (clause_text, clause_type) rows.

    # WHY: only rows with a nonempty answer correspond to an actual clause —
    # empty-answer rows just record that a given category doesn't appear in
    # that contract, and have no text to classify, so they are dropped rather
    # than kept as a 42nd "None" class.
    """
    clause_texts = []
    clause_types = []
    for example in qa_split:
        answers = example["answers"]["text"]
        if not answers:
            continue
        clause_texts.append(answers[0])
        clause_types.append(_extract_category(example["question"]))
    return Dataset.from_dict({"clause_text": clause_texts, "clause_type": clause_types})


def load_cuad() -> DatasetDict:
    """Load CUAD train/test splits and flatten them into clause classification rows.

    Returns a DatasetDict with "train" and "test" splits, each containing
    "clause_text" and "clause_type" columns. The official HF train/test split
    is contract-disjoint, which we reuse as-is to avoid a contract's boilerplate
    language leaking across splits.
    """
    raw = load_dataset(_CUAD_HUB_REPO, revision=_CUAD_REVISION)
    return DatasetDict(
        {
            "train": _flatten_qa_split(raw["train"]),
            "test": _flatten_qa_split(raw["test"]),
        }
    )


def clean_clause(text: str) -> str:
    """Normalize a raw clause string extracted from a contract PDF/OCR text.

    Returns the cleaned text.
    """
    # WHY: CUAD source text comes from OCR'd/converted legal PDFs, which use a
    # mix of curly quotes, non-breaking spaces, and other unicode variants of
    # otherwise-ASCII punctuation. NFKC normalization collapses these to their
    # canonical compatibility form so the same word doesn't tokenize two ways.
    text = unicodedata.normalize("NFKC", text)
    # WHY: PDF-to-text extraction leaves multi-space runs and hard line breaks
    # in the middle of sentences (column layouts, page breaks). Collapsing all
    # whitespace runs to a single space keeps the clause as one clean sentence
    # without losing word boundaries.
    text = re.sub(r"\s+", " ", text)
    # WHY: leading/trailing whitespace left over from the collapse above (or
    # from the original span boundaries) adds no signal for classification.
    text = text.strip()
    return text


def format_as_instruction(clause_text: str, clause_type: str) -> dict:
    """Format one (clause_text, clause_type) pair as an instruction-tuning example.

    Returns a dict with "instruction", "input", and "output" keys, matching the
    format the QLoRA training pipeline expects.
    """
    return {
        "instruction": INSTRUCTION,
        "input": clean_clause(clause_text),
        "output": clause_type,
    }


def _instruction_to_text(example: dict) -> str:
    """Render one instruction-formatted example as a single prompt+response string."""
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )


def build_training_dataset(dataset: DatasetDict, tokenizer, max_length: int = 512) -> DatasetDict:
    """Format and tokenize every split of `dataset` for QLoRA training.

    # TRADEOFF: max_length=512 is chosen from the exploration stats in
    # results/data_exploration.json — clause text alone has a 95th-percentile
    # length of ~160 tokens (Qwen2.5 tokenizer), so 512 leaves generous room
    # for the instruction template and response while still keeping sequences
    # short enough to fit comfortably in T4 16GB VRAM at a batch size of 4.
    # A handful of outlier clauses (max ~738 tokens) will be truncated; this is
    # accepted rather than raising max_length for the whole dataset, since it
    # would inflate compute cost for the other 95%+ of examples.
    Returns a DatasetDict of the same splits, each with "input_ids" and
    "attention_mask" columns added (raw text columns are kept alongside them).
    """

    def _format_and_tokenize(batch: dict) -> dict:
        texts = []
        outputs = []
        for clause_text, clause_type in zip(batch["clause_text"], batch["clause_type"]):
            example = format_as_instruction(clause_text, clause_type)
            texts.append(_instruction_to_text(example))
            outputs.append(example["output"])
        tokenized = tokenizer(texts, truncation=True, max_length=max_length)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    return DatasetDict(
        {
            split_name: split.map(_format_and_tokenize, batched=True)
            for split_name, split in dataset.items()
        }
    )


def get_class_weights(dataset: DatasetDict) -> dict:
    """Compute inverse-frequency class weights for the "train" split of `dataset`.

    # WHY: CUAD is severely imbalanced (see results/data_exploration.json — the
    # most common clause type "Parties" outnumbers the rarest "Unlimited/All-You-
    # Can-Eat-License" by ~78x). Without reweighting, a model can score high
    # accuracy by defaulting to frequent classes while never learning rare ones,
    # which macro F1 (CLAUDE.md's primary metric) would expose. These weights
    # are meant to be handed to a weighted cross-entropy loss during training.
    Returns a dict mapping clause_type -> weight.
    """
    train_split = dataset["train"]
    counts = Counter(train_split["clause_type"])
    total = sum(counts.values())
    num_classes = len(counts)
    # MATH: weight = total_examples / (num_classes * class_count). A perfectly
    # balanced dataset gives every class weight 1.0; rarer classes get a weight
    # above 1.0 proportional to how underrepresented they are, and weights
    # average to 1.0 across classes weighted by frequency, so overall loss
    # magnitude stays comparable to the unweighted case.
    return {
        clause_type: total / (num_classes * count)
        for clause_type, count in counts.items()
    }
