#!/usr/bin/env python3
"""
MMLU accuracy evaluation using modeling_qwen3_5_logits.exe

For each MMLU question the script:
  1. Formats the prompt (matching mmlu_combined_shuffled.yaml doc_to_text)
  2. Runs modeling_qwen3_5_logits.exe with --output-tokens 1
  3. Reads row 0 of logits.bin (prefill output — predicts first token after "Answer:")
  4. Picks the answer choice A/B/C/D with the highest logit score

Usage (Windows):
  python mmlu_eval_logits.py ^
      --exe  "C:\\Release\\modeling_qwen3_5_logits.exe" ^
      --model "C:\\models\\Qwen3.5-35B-A3B" ^
      --limit 50

Requirements:
  pip install transformers datasets numpy
"""

import argparse
import io
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for Greek/math chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# logits.bin reader
# ---------------------------------------------------------------------------

def load_logits(path: str) -> np.ndarray:
    """Return shape (num_tokens, vocab_size) float32 array."""
    with open(path, "rb") as f:
        magic, vocab_size, num_tokens = struct.unpack("<III", f.read(12))
    assert magic == 0x4C475453, f"Bad magic: {magic:#010x}  (expected 0x4C475453)"
    data = np.fromfile(path, dtype=np.float32, offset=12)
    assert data.size == num_tokens * vocab_size, (
        f"File size mismatch: expected {num_tokens * vocab_size} floats, got {data.size}"
    )
    return data.reshape(num_tokens, vocab_size)


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def build_choice_token_ids(model_path: str) -> dict:
    """
    Find the single-token IDs that Qwen3.5 assigns to A / B / C / D
    as they would appear right after 'Answer:'.

    Returns a dict like {"A": 362, "B": 425, "C": 356, "D": 423}
    together with the tokenizer object.
    """
    from transformers import AutoTokenizer

    print(f"[*] Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    choice_ids: dict[str, int] = {}
    for label in ("A", "B", "C", "D"):
        # After "Answer:" the most likely surface form is " A" (with leading space)
        for surface in (f" {label}", label):
            ids = tokenizer.encode(surface, add_special_tokens=False)
            if len(ids) == 1:
                choice_ids[label] = ids[0]
                break
        else:
            # Fallback: take first token even if multi-token
            ids = tokenizer.encode(f" {label}", add_special_tokens=False)
            choice_ids[label] = ids[0]
            print(
                f"  WARNING: ' {label}' encodes to {ids}; using first token {ids[0]}"
            )

    print(f"[*] Choice token IDs: {choice_ids}")
    return tokenizer, choice_ids


# ---------------------------------------------------------------------------
# Prompt formatting  (matches mmlu_combined_shuffled.yaml doc_to_text)
# ---------------------------------------------------------------------------

LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def format_prompt(doc: dict, fewshot_docs: list | None = None) -> str:
    """
    Format one MMLU document as a plain text prompt.

    With fewshot_docs, prepends answered examples (same format used by
    lm-evaluation-harness with fewshot_config.sampler=first_n).
    """
    def _single(d: dict, include_answer: bool = True) -> str:
        c = d["choices"]
        text = (
            f"{d['question'].strip()}\n"
            f"A. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\n"
            f"Answer:"
        )
        if include_answer:
            text += f" {LABEL_MAP[d['answer']]}\n\n"
        return text

    prefix = "".join(_single(fs) for fs in (fewshot_docs or []))
    return prefix + _single(doc, include_answer=False)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]


def load_mmlu_combined(dataset_cache: str | None = None, shuffle_seed: int = 117):
    """
    Load and combine all 57 MMLU subcategories.
    Returns (test_docs, dev_docs) as plain Python lists of dicts.

    dataset_cache:
        Path to HuggingFace datasets cache directory, e.g.
        "C:\\datasets" (expects sub-folder cais___mmlu inside).
        If None, downloads from HuggingFace Hub.
    """
    import datasets as hf_datasets

    load_kwargs = {}
    if dataset_cache:
        load_kwargs["cache_dir"] = dataset_cache

    all_test, all_dev = [], []

    print(f"[*] Loading {len(MMLU_SUBJECTS)} MMLU subjects...")
    for i, subject in enumerate(MMLU_SUBJECTS, 1):
        try:
            test_ds = hf_datasets.load_dataset(
                "cais/mmlu", name=subject, split="test", **load_kwargs
            )
            dev_ds = hf_datasets.load_dataset(
                "cais/mmlu", name=subject, split="dev", **load_kwargs
            )
        except Exception as exc:
            print(f"  WARNING: cannot load '{subject}': {exc}")
            continue

        for doc in test_ds:
            d = dict(doc)
            d.setdefault("subject", subject)
            all_test.append(d)
        for doc in dev_ds:
            d = dict(doc)
            d.setdefault("subject", subject)
            all_dev.append(d)

        if i % 10 == 0:
            print(f"  {i}/{len(MMLU_SUBJECTS)} subjects loaded …")

    # Shuffle test docs with fixed seed (matching mmlu_combined_shuffled task)
    import random
    rng = random.Random(shuffle_seed)
    rng.shuffle(all_test)

    print(f"[*] Total test: {len(all_test)},  dev: {len(all_dev)}")
    return all_test, all_dev


# ---------------------------------------------------------------------------
# Exe runner
# ---------------------------------------------------------------------------

# Environment variables required by the readme
_QUANT_ENV = {
    "OV_GENAI_USE_MODELING_API": "1",
    "OV_GPU_MOE_DISABLE_ONEDNN": "1",
    "OV_GENAI_INFLIGHT_QUANT_MODE": "int4_asym",
    "OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE": "128",
    "OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE": "int4_asym",
}


def run_exe(
    exe_path: str,
    model_path: str,
    prompt: str,
    work_dir: str,
    think: int = 0,
    device: str = "GPU",
) -> np.ndarray | None:
    """
    Run exe, return (num_tokens, vocab_size) logits array or None on failure.
    logits.bin is read from work_dir after the process completes.
    """
    logits_path = Path(work_dir) / "logits.bin"
    if logits_path.exists():
        logits_path.unlink()

    env = {**os.environ, **_QUANT_ENV}

    cmd = [
        exe_path,
        "--model", model_path,
        "--cache-model",
        "--mode", "text",
        "--prompt", prompt,
        "--output-tokens", "1",   # Only prefill logits needed for MC tasks
        "--temperature", "0",     # Greedy (doesn't affect saved logits, but clean)
        "--think", str(think),
    ]

    result = subprocess.run(
        cmd,
        env=env,
        cwd=work_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        print(f"  [EXE ERROR rc={result.returncode}] {result.stderr[:300]}")
        return None

    if not logits_path.exists():
        print(f"  [ERROR] logits.bin not found at {logits_path}")
        return None

    return load_logits(str(logits_path))


# ---------------------------------------------------------------------------
# Answer prediction from logits
# ---------------------------------------------------------------------------

def predict_answer(logits: np.ndarray, choice_token_ids: dict) -> tuple[str, dict]:
    """
    Use row 0 (prefill output) of the logits tensor.
    Returns (predicted_label, scores_dict).
    """
    prefill = logits[0]  # shape (vocab_size,)
    scores = {label: float(prefill[tid]) for label, tid in choice_token_ids.items()}
    pred = max(scores, key=scores.get)
    return pred, scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MMLU accuracy evaluation using modeling_qwen3_5_logits.exe"
    )
    parser.add_argument(
        "--exe", required=True,
        help="Path to modeling_qwen3_5_logits.exe",
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to Qwen3.5 HF model directory (used for tokenizer AND model)",
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Number of MMLU samples to evaluate (default: 10)",
    )
    parser.add_argument(
        "--num-fewshot", type=int, default=0,
        help="Number of few-shot examples prepended to each question (default: 0)",
    )
    parser.add_argument(
        "--dataset-cache", default=None,
        help=(
            "Path to local HuggingFace datasets cache dir "
            "(e.g. C:\\datasets).  Inside it expects the 'cais___mmlu' folder "
            "as downloaded by `datasets.load_dataset('cais/mmlu', ...)`.  "
            "If omitted, downloads from HuggingFace Hub."
        ),
    )
    parser.add_argument(
        "--work-dir", default=".",
        help="Directory where the exe writes logits.bin (default: current dir)",
    )
    parser.add_argument(
        "--output", default="mmlu_results.json",
        help="Path for output JSON results file (default: mmlu_results.json)",
    )
    parser.add_argument(
        "--think", type=int, default=0, choices=[0, 1],
        help="Pass --think 0|1 to exe (default: 0 = thinking off, recommended for MMLU)",
    )
    parser.add_argument(
        "--shuffle-seed", type=int, default=117,
        help="Random seed used to shuffle combined dataset (default: 117, matches task yaml)",
    )
    parser.add_argument(
        "--subject", default=None,
        help="Evaluate only this MMLU subject (e.g. 'abstract_algebra')",
    )
    args = parser.parse_args()

    # ---- tokenizer -------------------------------------------------------
    _tokenizer, choice_token_ids = build_choice_token_ids(args.model)

    # ---- dataset ---------------------------------------------------------
    all_test, all_dev = load_mmlu_combined(
        dataset_cache=args.dataset_cache,
        shuffle_seed=args.shuffle_seed,
    )

    if args.subject:
        all_test = [d for d in all_test if d.get("subject") == args.subject]
        all_dev = [d for d in all_dev if d.get("subject") == args.subject]
        if not all_test:
            print(f"[ERROR] No test docs found for subject '{args.subject}'")
            sys.exit(1)

    test_docs = all_test[: args.limit] if args.limit else all_test
    fewshot_docs = all_dev[: args.num_fewshot] if args.num_fewshot > 0 else None

    print(f"[*] Evaluating {len(test_docs)} samples  "
          f"(fewshot={args.num_fewshot}, think={args.think})")

    # ---- work dir --------------------------------------------------------
    work_dir = str(Path(args.work_dir).resolve())
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    # ---- evaluation loop -------------------------------------------------
    results = []
    correct = 0

    for i, doc in enumerate(test_docs):
        true_label = LABEL_MAP[doc["answer"]]
        prompt = format_prompt(doc, fewshot_docs)

        subject_str = doc.get("subject", "?")
        print(f"\n[{i+1}/{len(test_docs)}] {subject_str}")
        print(f"  Q: {doc['question'][:100].strip()} …")

        logits = run_exe(
            exe_path=args.exe,
            model_path=args.model,
            prompt=prompt,
            work_dir=work_dir,
            think=args.think,
        )

        if logits is None:
            results.append({
                "idx": i, "subject": subject_str,
                "pred": None, "true": true_label, "correct": False,
            })
            continue

        pred, scores = predict_answer(logits, choice_token_ids)
        is_correct = pred == true_label
        correct += int(is_correct)

        print(f"  Pred: {pred}   True: {true_label}   {'OK' if is_correct else 'FAIL'}")
        print(f"  Logit scores: { {k: f'{v:.3f}' for k, v in scores.items()} }")

        results.append({
            "idx": i,
            "subject": subject_str,
            "question": doc["question"][:120],
            "pred": pred,
            "true": true_label,
            "correct": is_correct,
            "scores": {k: float(v) for k, v in scores.items()},
        })

    # ---- summary ---------------------------------------------------------
    total = len(results)
    accuracy = correct / total if total else 0.0

    print(f"\n{'='*55}")
    print(f"Accuracy : {accuracy:.4f}   ({correct}/{total})")
    print(f"{'='*55}")

    output_data = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "config": {
            "exe": args.exe,
            "model": args.model,
            "limit": args.limit,
            "num_fewshot": args.num_fewshot,
            "think": args.think,
            "shuffle_seed": args.shuffle_seed,
            "subject": args.subject,
        },
        "choice_token_ids": choice_token_ids,
        "results": results,
    }

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[*] Results saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
