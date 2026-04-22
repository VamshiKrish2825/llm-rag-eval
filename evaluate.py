"""
LLM Benchmarking & Evaluation Framework
========================================
Evaluates multiple LLM-backed RAG pipelines using RAGAS metrics:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall

Judge LLM  : Groq (free tier) — no OpenAI billing needed
Models tested: Llama 3.1 8B, Mistral 7B, Gemma 2 9B  (via Groq)
Dataset    : 20-sample finance Q&A (see data/qa_dataset.json)

Usage:
    python evaluate.py                   # run full benchmark
    python evaluate.py --samples 5      # quick smoke-test
    python evaluate.py --model llama    # single model
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset
from openai import OpenAI          # Groq is OpenAI-compatible
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

# ── Config ────────────────────────────────────────────────────────────────────

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Models available on Groq free tier (Apr 2025)
MODELS = {
    "llama":   "llama-3.1-8b-instant",
    "mistral": "open-mistral-7b",      # via Groq; fallback: mixtral-8x7b-32768
    "gemma":   "gemma2-9b-it",
}

# Metric names → RAGAS metric objects (instantiated per run to avoid state issues)
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

RESULTS_DIR = Path("results")
DATA_PATH   = Path("data/qa_dataset.json")

# ── Groq client helpers ───────────────────────────────────────────────────────

def get_groq_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Export it: export GROQ_API_KEY=gsk_..."
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def call_groq(client: OpenAI, model_id: str, prompt: str, max_tokens: int = 512) -> tuple[str, float, int]:
    """
    Call a Groq model and return (answer, latency_seconds, tokens_used).
    """
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    latency = time.perf_counter() - start
    answer  = resp.choices[0].message.content.strip()
    tokens  = resp.usage.total_tokens if resp.usage else 0
    return answer, latency, tokens

# ── Dataset loader ────────────────────────────────────────────────────────────

def load_dataset(path: Path, n: int | None = None) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if n:
        data = data[:n]
    return data

# ── RAG simulation ────────────────────────────────────────────────────────────

def build_rag_prompt(question: str, context: str) -> str:
    return (
        "You are a helpful financial analyst. "
        "Answer the question using ONLY the context below. "
        "Be concise and factual.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

# ── Evaluation runner ─────────────────────────────────────────────────────────

def run_model_evaluation(
    model_name: str,
    model_id: str,
    samples: list[dict],
    groq_client: OpenAI,
    judge_llm,
) -> pd.DataFrame:
    """
    Generate answers for all samples with the given model,
    then score them with RAGAS using the judge LLM.
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}  ({model_id})")
    print(f"{'='*60}")

    rows = []
    for i, sample in enumerate(samples):
        question = sample["question"]
        context  = sample["context"]
        reference = sample["reference_answer"]

        prompt = build_rag_prompt(question, context)
        try:
            answer, latency, tokens = call_groq(groq_client, model_id, prompt)
        except Exception as exc:
            print(f"  [!] Sample {i+1} failed: {exc}")
            answer, latency, tokens = "", 0.0, 0

        rows.append({
            "user_input":          question,
            "retrieved_contexts":  [context],
            "response":            answer,
            "reference":           reference,
            "latency_sec":         round(latency, 3),
            "tokens_used":         tokens,
        })
        print(f"  [{i+1:02d}/{len(samples)}] latency={latency:.2f}s  tokens={tokens}")

    # Build HuggingFace Dataset for RAGAS
    hf_data = {
        "user_input":         [r["user_input"]         for r in rows],
        "retrieved_contexts": [r["retrieved_contexts"] for r in rows],
        "response":           [r["response"]           for r in rows],
        "reference":          [r["reference"]          for r in rows],
    }
    hf_dataset = Dataset.from_dict(hf_data)

    # RAGAS metrics — freshly instantiated with the judge LLM
    metrics = [
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm),
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
    ]

    print(f"\n  Running RAGAS evaluation …")
    try:
        result = evaluate(hf_dataset, metrics=metrics)
        scores_df = result.to_pandas()
    except Exception as exc:
        print(f"  [!] RAGAS evaluation failed: {exc}")
        scores_df = pd.DataFrame(rows)
        for col in METRIC_NAMES:
            scores_df[col] = None
        return scores_df

    # Merge generation stats with RAGAS scores
    stats_df = pd.DataFrame(rows)
    for col in ["latency_sec", "tokens_used"]:
        scores_df[col] = stats_df[col].values

    scores_df["model"] = model_name
    return scores_df

# ── Summary builder ───────────────────────────────────────────────────────────

def build_summary(all_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate per-model mean scores + latency + tokens."""
    records = []
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for model_name, df in all_results.items():
        row = {"model": model_name}
        for col in metric_cols:
            if col in df.columns:
                row[col] = round(df[col].mean(), 4)
            else:
                row[col] = None
        row["avg_latency_sec"] = round(df["latency_sec"].mean(), 3)
        row["avg_tokens"]      = int(df["tokens_used"].mean())
        records.append(row)
    return pd.DataFrame(records).sort_values("faithfulness", ascending=False)

# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM RAG Benchmark with RAGAS + Groq")
    parser.add_argument("--samples", type=int, default=None, help="Limit dataset size (default: all)")
    parser.add_argument("--model",   type=str, default=None, help="Single model key: llama | mistral | gemma")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    # Load dataset
    samples = load_dataset(DATA_PATH, n=args.samples)
    print(f"Loaded {len(samples)} samples from {DATA_PATH}")

    # Groq client (used for both RAG generation AND as judge LLM in RAGAS)
    groq_client = get_groq_client()

    # Judge LLM: use a fast, capable model for evaluation
    judge_model_id = MODELS["llama"]    # llama-3.1-8b is free and fast
    judge_llm = llm_factory(
        judge_model_id,
        provider="openai",              # Groq is OpenAI-compatible
        client=groq_client,
    )

    # Select models to benchmark
    models_to_run = MODELS if not args.model else {args.model: MODELS[args.model]}

    all_results: dict[str, pd.DataFrame] = {}

    for model_name, model_id in models_to_run.items():
        df = run_model_evaluation(
            model_name=model_name,
            model_id=model_id,
            samples=samples,
            groq_client=groq_client,
            judge_llm=judge_llm,
        )
        all_results[model_name] = df
        out_path = RESULTS_DIR / f"{model_name}_results.csv"
        df.to_csv(out_path, index=False)
        print(f"\n  Saved → {out_path}")

    # Build and print summary
    summary = build_summary(all_results)
    summary_path = RESULTS_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*60)
    print("  BENCHMARK SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))
    print(f"\nSummary saved → {summary_path}")
    print("\nNext step: run  python visualize.py  to generate charts.")


if __name__ == "__main__":
    main()
