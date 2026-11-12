"""Turn the result files into charts and a Markdown summary.

    uv run python -m evals.report

Reads results/retrieval_dev.json, retrieval_test.json, ranks_test.parquet (and generation.json
when present). Writes results/chunk_size_dev.png, retrieval_test.png and RESULTS.md.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from govqa import config
from govqa.metrics import paired_bootstrap_diff, per_query_metrics

SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
# Fixed colour per method (identity), never per rank.
COLORS = {"bm25": "#2a78d6", "dense": "#eb6834", "hybrid": "#1baf7a", "hybrid_rerank": "#6250d6"}
LABELS = {
    "bm25": "BM25 (keywords)",
    "dense": "Dense (E5 embeddings)",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + reranker",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    }
)


def _load(name: str) -> list[dict] | None:
    path = config.RESULTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def chunk_size_chart(rows: list[dict], metric: str = "recall@1") -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = [m for m in COLORS if any(r["method"] == m for r in rows)]
    for method in methods:
        pts = sorted(
            (int(r["index"].lstrip("w")), r["metrics"][metric])
            for r in rows
            if r["method"] == method
        )
        if len(pts) < 2:
            continue
        xs = list(range(len(pts)))
        ys = [p[1]["mean"] for p in pts]
        ax.fill_between(
            xs,
            [p[1]["low"] for p in pts],
            [p[1]["high"] for p in pts],
            color=COLORS[method],
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            xs,
            ys,
            color=COLORS[method],
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=LABELS[method],
        )
        ax.annotate(
            f"{ys[-1]:.2f}",
            (xs[-1], ys[-1]),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            color=INK_2,
        )
        ax.set_xticks(xs, [f"{p[0]} words" for p in pts])
    n_sizes = len({r["index"] for r in rows})
    ax.set_xlim(-0.2, n_sizes - 0.6)
    ax.set_ylabel(metric.replace("recall", "Recall"))
    ax.set_title(f"Chunk size matters less than the method (dev, n={rows[0]['n_queries']})")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", ncols=3, bbox_to_anchor=(0, -0.28))
    fig.text(0.01, 0.01, "Bands: 95% bootstrap confidence interval", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(config.RESULTS_DIR / "chunk_size_dev.png", dpi=160)
    plt.close(fig)


def method_chart(rows: list[dict]) -> None:
    metrics = ["recall@1", "recall@5", "mrr@10"]
    rows = sorted(rows, key=lambda r: list(COLORS).index(r["method"]), reverse=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    lo = min(r["metrics"][m]["low"] for r in rows for m in metrics)
    for ax, metric in zip(axes, metrics, strict=True):
        for y, r in enumerate(rows):
            v = r["metrics"][metric]
            color = COLORS[r["method"]]
            ax.plot([v["low"], v["high"]], [y, y], color=color, linewidth=2, solid_capstyle="round")
            ax.plot(
                v["mean"],
                y,
                "o",
                color=color,
                markersize=9,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
            )
            ax.annotate(
                f"{v['mean']:.3f}",
                (v["mean"], y),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                color=INK_2,
                fontsize=9,
            )
        ax.set_title(metric.replace("recall", "Recall").replace("mrr", "MRR"), fontweight="normal")
        ax.set_xlim(max(0, lo - 0.03), 1.005)
        ax.set_ylim(-0.6, len(rows) - 0.3)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(rows)), [LABELS[r["method"]] for r in rows])
    fig.suptitle(
        f"Retrieval on the held-out test split (n={rows[0]['n_queries']} questions)",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=12,
    )
    fig.text(0.01, 0.01, "Lines: 95% bootstrap confidence interval", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(config.RESULTS_DIR / "retrieval_test.png", dpi=160)
    plt.close(fig)


def _table(rows: list[dict]) -> str:
    cols = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
    head = "| Index | Method | " + " | ".join(cols) + " | ms/query |\n"
    head += "|---|---|" + "---|" * (len(cols) + 1) + "\n"
    body = ""
    for r in rows:
        cells = [
            f"{r['metrics'][c]['mean']:.3f} ({r['metrics'][c]['low']:.3f}–{r['metrics'][c]['high']:.3f})"
            if c in ("recall@1", "mrr@10")
            else f"{r['metrics'][c]['mean']:.3f}"
            for c in cols
        ]
        body += f"| {r['index']} | {LABELS[r['method']]} | " + " | ".join(cells)
        body += f" | {r['ms_per_query']:.1f} |\n"
    return head + body


def significance(test_rows: list[dict]) -> str:
    ranks = pd.read_parquet(config.RESULTS_DIR / "ranks_test.parquet")
    cols = {r["method"]: f"{r['index']}|{r['method']}" for r in test_rows}
    best = max(test_rows, key=lambda r: r["metrics"]["mrr@10"]["mean"])["method"]
    out = f"Paired bootstrap, {LABELS[best]} minus each other method, same {len(ranks)} questions:\n\n"
    out += "| Compared with | Δ Recall@1 (95% CI) | Δ MRR@10 (95% CI) | Significant |\n|---|---|---|---|\n"
    a = per_query_metrics(ranks[cols[best]].to_numpy())
    for method, col in cols.items():
        if method == best:
            continue
        b = per_query_metrics(ranks[col].to_numpy())
        cells, sig = [], []
        for metric in ("recall@1", "mrr@10"):
            d, lo, hi = paired_bootstrap_diff(a[metric], b[metric])
            cells.append(f"{d:+.3f} ({lo:+.3f} to {hi:+.3f})")
            sig.append(lo > 0 or hi < 0)
        verdict = "yes" if all(sig) else ("partly" if any(sig) else "no")
        out += f"| {LABELS[method]} | {cells[0]} | {cells[1]} | {verdict} |\n"
    return out


def generation_section(gen: dict | None) -> str:
    if gen is None:
        return "_Not run yet. Needs an LLM key: `uv run python -m evals.run_generation`._\n"

    def pct(key: str) -> str:
        v = gen[key]
        if v["mean"] is None:
            return "n/a"
        return f"{v['mean']:.1%} ({v['low']:.1%}–{v['high']:.1%}, n={v['n']})"

    return (
        f"Model `{gen['llm_model']}`, judge `{gen['judge_model']}`, "
        f"{gen['n_answerable']} answerable + {gen['n_out_of_scope']} out-of-scope questions.\n\n"
        "| Metric | Value (95% CI) |\n|---|---|\n"
        f"| Correct page in the prompt | {pct('gold_in_context')} |\n"
        f"| Faithful to the sources (judge) | {pct('faithfulness')} |\n"
        f"| Fully correct vs official text (judge) | {pct('fully_correct')} |\n"
        f"| Cites the correct page | {pct('cited_gold_page')} |\n"
        f"| Wrongly refuses an answerable question | {pct('false_refusal_rate')} |\n"
        f"| Correctly refuses an out-of-scope question | {pct('correct_refusal_rate')} |\n"
        f"| Latency p50 / p95 | {gen['latency_p50_s']} s / {gen['latency_p95_s']} s |\n"
        f"| Cost per answer | ${gen['cost_per_answer_usd']:.5f} |\n"
    )


def main() -> None:
    dev, test = _load("retrieval_dev.json"), _load("retrieval_test.json")
    md = "# Results\n\nGenerated by `python -m evals.report`. Do not edit by hand.\n\n"
    if dev:
        chunk_size_chart(dev)
        md += "## Dev split: choosing the configuration\n\n" + _table(dev) + "\n"
    if test:
        method_chart(test)
        md += "## Test split: final retrieval numbers\n\n" + _table(test) + "\n"
        md += significance(test) + "\n"
    md += "## Generation\n\n" + generation_section(_load("generation.json"))
    (config.RESULTS_DIR / "RESULTS.md").write_text(md, encoding="utf-8")
    print(f"wrote {config.RESULTS_DIR / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
