"""
generate_mock_results.py
========================
Generates realistic mock benchmark results so you can preview
all charts and the README outputs WITHOUT needing a Groq API key.

Usage:
    python generate_mock_results.py
    python visualize.py          # then generate charts from mock data
"""

import json
import random
from pathlib import Path

import pandas as pd
import numpy as np

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

random.seed(42)
np.random.seed(42)

# Realistic score distributions (mean, std) per model per metric
# Llama 3.1 is strongest, Mistral middle, Gemma slightly lower on faithfulness
SCORE_PROFILES = {
    "llama": {
        "faithfulness":      (0.87, 0.07),
        "answer_relevancy":  (0.85, 0.06),
        "context_precision": (0.82, 0.08),
        "context_recall":    (0.84, 0.07),
        "avg_latency_sec":   (1.3,  0.4),
        "avg_tokens":        (310,  40),
    },
    "mistral": {
        "faithfulness":      (0.79, 0.09),
        "answer_relevancy":  (0.81, 0.07),
        "context_precision": (0.76, 0.10),
        "context_recall":    (0.78, 0.08),
        "avg_latency_sec":   (1.6,  0.5),
        "avg_tokens":        (295,  35),
    },
    "gemma": {
        "faithfulness":      (0.62, 0.11),
        "answer_relevancy":  (0.75, 0.09),
        "context_precision": (0.70, 0.10),
        "context_recall":    (0.68, 0.09),
        "avg_latency_sec":   (1.1,  0.3),
        "avg_tokens":        (280,  30),
    },
}

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

with open("data/qa_dataset.json") as f:
    samples = json.load(f)

summary_rows = []

for model_name, profile in SCORE_PROFILES.items():
    rows = []
    for sample in samples:
        row = {
            "user_input":         sample["question"],
            "retrieved_contexts": [sample["context"]],
            "response":           f"[mock answer for {model_name}]",
            "reference":          sample["reference_answer"],
            "model":              model_name,
        }
        for metric in METRIC_COLS:
            mu, sigma = profile[metric]
            row[metric] = float(np.clip(np.random.normal(mu, sigma), 0.0, 1.0))

        lat_mu, lat_sigma = profile["avg_latency_sec"]
        row["latency_sec"] = float(np.clip(np.random.normal(lat_mu, lat_sigma), 0.2, 5.0))

        tok_mu, tok_sigma = profile["avg_tokens"]
        row["tokens_used"] = int(np.clip(np.random.normal(tok_mu, tok_sigma), 100, 800))

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"{model_name}_results.csv", index=False)

    summary_row = {"model": model_name}
    for metric in METRIC_COLS:
        summary_row[metric] = round(df[metric].mean(), 4)
    summary_row["avg_latency_sec"] = round(df["latency_sec"].mean(), 3)
    summary_row["avg_tokens"]      = int(df["tokens_used"].mean())
    summary_rows.append(summary_row)

    print(f"  {model_name}: faithfulness={summary_row['faithfulness']:.3f}  "
          f"relevancy={summary_row['answer_relevancy']:.3f}  "
          f"precision={summary_row['context_precision']:.3f}  "
          f"recall={summary_row['context_recall']:.3f}")

summary = pd.DataFrame(summary_rows).sort_values("faithfulness", ascending=False)
summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

print("\nMock results saved to results/")
print("Faithfulness gap (top vs bottom):",
      round(summary["faithfulness"].max() - summary["faithfulness"].min(), 3))
print("\nNext: run  python visualize.py  to generate all charts.")
