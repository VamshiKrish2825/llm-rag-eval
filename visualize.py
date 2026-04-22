"""
visualize.py
============
Generates all charts from the benchmark results in results/summary.csv.

Produces:
  results/radar_chart.png        — 4-metric radar for all models
  results/cost_accuracy.png      — cost-vs-accuracy scatter
  results/metric_bars.png        — grouped bar chart across metrics
  results/latency_comparison.png — latency bar chart

Run after evaluate.py:
    python visualize.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
METRIC_COLS  = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Groq per-token cost (approx. free-tier estimate for Llama 3.1 8B; adjust as needed)
# All models on Groq free tier are $0 (within limits) — for cost-accuracy chart
# we use a hypothetical token cost for demonstration purposes.
COST_PER_1K_TOKENS = {
    "llama":   0.06,   # $/1k tokens (Groq paid estimate)
    "mistral": 0.10,
    "gemma":   0.08,
}

MODEL_COLORS = {
    "llama":   "#3B82F6",   # blue
    "mistral": "#F59E0B",   # amber
    "gemma":   "#10B981",   # green
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_summary() -> pd.DataFrame:
    path = RESULTS_DIR / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run  python evaluate.py  first."
        )
    return pd.read_csv(path)


def model_color(name: str) -> str:
    return MODEL_COLORS.get(name.lower(), "#6B7280")

# ── Chart 1: Radar chart ───────────────────────────────────────────────────────

def plot_radar(df: pd.DataFrame):
    labels   = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"]
    num_vars = len(labels)
    angles   = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles  += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for _, row in df.iterrows():
        model  = row["model"]
        values = [row.get(c, 0) or 0 for c in METRIC_COLS]
        values += values[:1]
        color  = model_color(model)
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=model.capitalize())
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8, color="grey")
    ax.set_title("RAGAS Metric Comparison — Radar Chart", size=14, pad=20, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=10)

    plt.tight_layout()
    path = RESULTS_DIR / "radar_chart.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── Chart 2: Cost-vs-Accuracy scatter ─────────────────────────────────────────

def plot_cost_accuracy(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in df.iterrows():
        model    = row["model"]
        accuracy = row.get("faithfulness", 0) or 0         # primary accuracy proxy
        tokens   = row.get("avg_tokens", 0) or 0
        cost_per_query = (tokens / 1000) * COST_PER_1K_TOKENS.get(model, 0.08)
        color    = model_color(model)

        ax.scatter(cost_per_query, accuracy, s=200, color=color, zorder=3)
        ax.annotate(
            model.capitalize(),
            (cost_per_query, accuracy),
            textcoords="offset points",
            xytext=(10, 4),
            fontsize=11,
            color=color,
            fontweight="bold",
        )

    ax.set_xlabel("Estimated Cost per Query (USD)", fontsize=12)
    ax.set_ylabel("Faithfulness Score", fontsize=12)
    ax.set_title("Cost vs. Accuracy Trade-off", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axhline(y=0.8, color="grey", linestyle=":", linewidth=1, label="0.8 threshold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = RESULTS_DIR / "cost_accuracy.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


# ── Chart 3: Grouped bar chart ────────────────────────────────────────────────

def plot_metric_bars(df: pd.DataFrame):
    models  = df["model"].tolist()
    x       = np.arange(len(METRIC_COLS))
    n       = len(models)
    width   = 0.22
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (_, row) in enumerate(df.iterrows()):
        model  = row["model"]
        scores = [row.get(c, 0) or 0 for c in METRIC_COLS]
        color  = model_color(model)
        bars   = ax.bar(x + offsets[i], scores, width, label=model.capitalize(), color=color, alpha=0.85)
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.2f}",
                ha="center", va="bottom", fontsize=8.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (0–1)", fontsize=12)
    ax.set_title("Metric-by-Metric Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    path = RESULTS_DIR / "metric_bars.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


# ── Chart 4: Latency ──────────────────────────────────────────────────────────

def plot_latency(df: pd.DataFrame):
    models   = [m.capitalize() for m in df["model"]]
    latencies = [row.get("avg_latency_sec", 0) or 0 for _, row in df.iterrows()]
    colors   = [model_color(m.lower()) for m in models]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(models, latencies, color=colors, width=0.5, alpha=0.85)
    for bar, lat in zip(bars, latencies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{lat:.2f}s",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Avg Latency (seconds)", fontsize=12)
    ax.set_title("Average Response Latency per Model", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(latencies) * 1.3 if latencies else 5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    path = RESULTS_DIR / "latency_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    df = load_summary()
    print(f"Loaded summary for models: {df['model'].tolist()}")
    print("\nGenerating charts …")

    plot_radar(df)
    plot_cost_accuracy(df)
    plot_metric_bars(df)
    plot_latency(df)

    print("\nAll charts saved to results/")


if __name__ == "__main__":
    main()
