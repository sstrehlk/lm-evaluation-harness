#!/usr/bin/env python3
"""
MMLU accuracy evaluation using original HuggingFace model (transformers).

Equivalent to mmlu_eval_logits.py but runs inference via transformers instead of
modeling_qwen3_5_logits.exe.  Same prompt format, fewshot handling, logit-based
answer selection and output JSON schema — results are directly comparable.

Usage:
  python mmlu_eval_logits_hf.py \
      --model "/path/to/Qwen3.5-35B-A3B" \
      --limit 50 \
      --device cuda          # or cpu
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

# Force UTF-8 output (Windows compatibility)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Tokenizer + model helpers
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str, device: str, torch_dtype: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    dtype = dtype_map.get(torch_dtype, "auto")

    print(f"[*] Loading tokenizer from: {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"[*] Loading model  (device={device}, dtype={torch_dtype}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device if device not in ("cpu",) else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    return model, tokenizer


def build_choice_token_ids(tokenizer) -> dict:
    """
    Find single-token IDs for A / B / C / D as they appear after 'Answer:'.
    Identical logic to mmlu_eval_logits.py so token IDs are consistent.
    """
    choice_ids: dict[str, int] = {}
    for label in ("A", "B", "C", "D"):
        for surface in (f" {label}", label):
            ids = tokenizer.encode(surface, add_special_tokens=False)
            if len(ids) == 1:
                choice_ids[label] = ids[0]
                break
        else:
            ids = tokenizer.encode(f" {label}", add_special_tokens=False)
            choice_ids[label] = ids[0]
            print(
                f"  WARNING: ' {label}' encodes to {ids}; using first token {ids[0]}"
            )

    print(f"[*] Choice token IDs: {choice_ids}", flush=True)
    return choice_ids


# ---------------------------------------------------------------------------
# Prompt formatting  (identical to mmlu_eval_logits.py)
# ---------------------------------------------------------------------------

LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def format_prompt(doc: dict, fewshot_docs: list | None = None) -> str:
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
# Dataset loading  (identical to mmlu_eval_logits.py)
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
    import datasets as hf_datasets
    import random

    load_kwargs = {}
    if dataset_cache:
        load_kwargs["cache_dir"] = dataset_cache

    all_test, all_dev = [], []

    print(f"[*] Loading {len(MMLU_SUBJECTS)} MMLU subjects...", flush=True)
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
            print(f"  {i}/{len(MMLU_SUBJECTS)} subjects loaded …", flush=True)

    rng = random.Random(shuffle_seed)
    rng.shuffle(all_test)

    print(f"[*] Total test: {len(all_test)},  dev: {len(all_dev)}", flush=True)
    return all_test, all_dev


# ---------------------------------------------------------------------------
# Inference: get logits for first predicted token after the prompt
# ---------------------------------------------------------------------------

def get_prefill_logits(model, tokenizer, prompt: str, device: str) -> np.ndarray | None:
    """
    Tokenize prompt, run one forward pass, return logits for the last position
    as a 1-D numpy array of shape (vocab_size,).
    This is the equivalent of row 0 of logits.bin from modeling_qwen3_5_logits.exe.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # logits shape: (1, seq_len, vocab_size) — take last token position
    last_logits = outputs.logits[0, -1, :].float().cpu().numpy()
    return last_logits


# ---------------------------------------------------------------------------
# Answer prediction  (identical logic to mmlu_eval_logits.py)
# ---------------------------------------------------------------------------

def predict_answer(prefill_logits: np.ndarray, choice_token_ids: dict) -> tuple[str, dict]:
    scores = {label: float(prefill_logits[tid]) for label, tid in choice_token_ids.items()}
    pred = max(scores, key=scores.get)
    return pred, scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MMLU accuracy evaluation using HuggingFace transformers model"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to Qwen3.5 HF model directory",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device for inference: cpu / cuda / cuda:0 / auto (default: auto)",
    )
    parser.add_argument(
        "--dtype", default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype (default: auto)",
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
            "(e.g. /data/datasets).  Expects 'cais___mmlu' folder inside.  "
            "If omitted, downloads from HuggingFace Hub."
        ),
    )
    parser.add_argument(
        "--output", default="mmlu_results_hf.json",
        help="Path for output JSON results file (default: mmlu_results_hf.json)",
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

    # ---- model + tokenizer -----------------------------------------------
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.dtype)
    choice_token_ids = build_choice_token_ids(tokenizer)

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
          f"(fewshot={args.num_fewshot})", flush=True)

    # ---- evaluation loop -------------------------------------------------
    results = []
    correct = 0
    durations: list = []
    eval_start = time.monotonic()

    def _fmt_seconds(s):
        s = int(s)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        if h:
            return f"{h}h {m:02d}m {sec:02d}s"
        if m:
            return f"{m}m {sec:02d}s"
        return f"{sec}s"

    for i, doc in enumerate(test_docs):
        true_label = LABEL_MAP[doc["answer"]]
        prompt = format_prompt(doc, fewshot_docs)
        subject_str = doc.get("subject", "?")

        if durations:
            avg = sum(durations) / len(durations)
            eta = _fmt_seconds(avg * (len(test_docs) - i))
            elapsed = _fmt_seconds(time.monotonic() - eval_start)
            timing_str = f"  avg={avg:.1f}s  elapsed={elapsed}  ETA={eta}"
        else:
            timing_str = ""

        print(f"\n[{i+1}/{len(test_docs)}] {subject_str}{timing_str}", flush=True)
        print(f"  Q: {doc['question'][:100].strip()} …", flush=True)

        t0 = time.monotonic()
        try:
            prefill_logits = get_prefill_logits(model, tokenizer, prompt, args.device)
        except Exception as exc:
            print(f"  [ERROR] inference failed: {exc}", flush=True)
            prefill_logits = None
        sample_time = time.monotonic() - t0
        durations.append(sample_time)

        if prefill_logits is None:
            results.append({
                "idx": i, "subject": subject_str,
                "pred": None, "true": true_label, "correct": False,
                "time_s": round(sample_time, 2),
            })
            continue

        pred, scores = predict_answer(prefill_logits, choice_token_ids)
        is_correct = pred == true_label
        correct += int(is_correct)
        acc_so_far = correct / (i + 1)

        print(f"  Pred: {pred}   True: {true_label}   {'OK' if is_correct else 'FAIL'}"
              f"   acc={acc_so_far:.3f}   time={sample_time:.1f}s", flush=True)
        print(f"  Logit scores: { {k: f'{v:.3f}' for k, v in scores.items()} }", flush=True)

        results.append({
            "idx": i,
            "subject": subject_str,
            "question": doc["question"][:120],
            "pred": pred,
            "true": true_label,
            "correct": is_correct,
            "scores": {k: float(v) for k, v in scores.items()},
            "time_s": round(sample_time, 2),
        })

    # ---- summary ---------------------------------------------------------
    total = len(results)
    accuracy = correct / total if total else 0.0
    total_elapsed = time.monotonic() - eval_start
    avg_time = sum(durations) / len(durations) if durations else 0.0

    print(f"\n{'='*55}")
    print(f"Accuracy      : {accuracy:.4f}   ({correct}/{total})")
    print(f"Total time    : {_fmt_seconds(total_elapsed)}")
    print(f"Avg per sample: {avg_time:.1f}s")
    print(f"{'='*55}")

    output_data = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "total_time_s": round(total_elapsed, 1),
        "avg_time_per_sample_s": round(avg_time, 2),
        "config": {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "limit": args.limit,
            "num_fewshot": args.num_fewshot,
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
