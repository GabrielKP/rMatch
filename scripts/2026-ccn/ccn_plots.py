"""
plot_model_comparison.py

Produces one plot per testset (alice10, memsearch10, monthiversary6, repeat_reliability)
comparing LLMs on F1 score.

Layout mirrors the existing scatter_metrics.py:
  - Each model occupies a "region" on the x-axis (like metrics did before)
  - Individual subject F1 scores are jittered within that region
  - A solid bar shows the mean; a vertical line with caps shows ±SEM
  - All dots are a single colour (no story-level colouring)
"""

import json
import pickle
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_score, recall_score

# ── repeat-reliability metric to display ─────────────────────────────────────
# Choose ONE of:
#   "f1_vs_human"   – per-recall mean F1 vs human annotations (one dot per recall)
#   "pairwise_f1"   – mean pairwise F1 across repeats (self-consistency)
#   "both"          – side-by-side subplots showing both
RR_METRIC = "pairwise_f1"

# ── data root ────────────────────────────────────────────────────────────────
DATA_ROOT = Path("data/eval")

# ── run registry ─────────────────────────────────────────────────────────────
RUNS = [
    # ── alice10 ──────────────────────────────────────────────────────────────
    dict(
        run_dir="20260227_223616-cyoa_alice10-anthropic-m_claude-opus-4-6-seed_42",
        testset="alice10",
        model="Claude Opus",
    ),
    dict(
        run_dir="20260309_171354-cyoa_alice10-anthropic-m_claude-sonnet-4-6-seed_42",
        testset="alice10",
        model="Claude Sonnet",
    ),
    dict(
        run_dir="20260309_194145-cyoa_alice10-anthropic-m_claude-haiku-4-5-seed_42",
        testset="alice10",
        model="Claude Haiku",
    ),
    dict(
        run_dir="20260227_224618-cyoa_alice10-openai-m_gpt-5.2-seed_42",
        testset="alice10",
        model="GPT-5.2",
    ),
    # ── memsearch10 ──────────────────────────────────────────────────────────
    dict(
        run_dir="20260309_215439-memsearch10-anthropic-m_claude-opus-4-6-seed_42",
        testset="memsearch10",
        model="Claude Opus",
    ),
    dict(
        run_dir="20260309_213738-memsearch10-anthropic-m_claude-sonnet-4-6-seed_42",
        testset="memsearch10",
        model="Claude Sonnet",
    ),
    dict(
        run_dir="20260310_010740-memsearch10-anthropic-m_claude-haiku-4-5-seed_42",
        testset="memsearch10",
        model="Claude Haiku",
    ),
    dict(
        run_dir="20260227_232145-memsearch10-openai-m_gpt-5.2-seed_42",
        testset="memsearch10",
        model="GPT-5.2",
    ),
    # ── monthiversary6 ───────────────────────────────────────────────────────
    dict(
        run_dir="20260309_215727-cyoa_monthiversary6-anthropic-m_claude-haiku-4-5-seed_42",
        testset="monthiversary6",
        model="Claude Haiku",
    ),
    # ── repeat reliability ────────────────────────────────────────────────────
    dict(
        run_dir="20260310_003817-rr-cyoa_alice10-anthropic-m_claude-haiku-4-5-seed_42",
        testset="repeat_reliability",
        model="Haiku / alice10",
        rr=True,
    ),
    dict(
        run_dir="20260310_003831-rr-memsearch10-anthropic-m_claude-haiku-4-5-seed_42",
        testset="repeat_reliability",
        model="Haiku / memsearch10",
        rr=True,
    ),
]

# ── visual constants ──────────────────────────────────────────────────────────
# Per-model colours: Claude shades of orange, GPT a slate blue
MODEL_COLORS: dict[str, str] = {
    "Claude Opus": "#b34700",  # deep burnt orange
    "Claude Sonnet": "#e06010",  # mid orange
    "Claude Haiku": "#f5a623",  # light amber
    "GPT-5.2": "#4a6fa5",  # steel blue
}
DEFAULT_COLOR = "#888888"  # fallback for unregistered models

MEAN_COLOR = "#1a1a1a"
JITTER_SCALE = 0.18  # half-width of jitter cloud within a bar
BAR_WIDTH = 0.55  # width of each bar
DOT_SIZE = 7
DOT_OPACITY = 0.85
DOT_COLOR_OVERLAY = "white"  # scatter dots overlay on coloured bars
SPACING = 1.5  # distance between model region centres

# RR plot still uses the old line-based rendering; give it a single dot colour
RR_DOT_COLOR = "#4C72B0"

TESTSET_TITLES = {
    "alice10": "CYOA – Alice 10",
    "memsearch10": "Memsearch 10",
    "monthiversary6": "CYOA – Monthiversary 6",
    "repeat_reliability": "Repeat Reliability",
}

RR_METRIC_LABELS = {
    "f1_vs_human": "F1 vs Human Annotations",
    "pairwise_f1": "Mean Pairwise F1 (self-consistency)",
}

OUTPUT_DIR = Path("data/eval/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)


# ── data loaders ─────────────────────────────────────────────────────────────


def resolve(run_dir: str) -> Path:
    p = Path(run_dir)
    return p if p.is_absolute() else DATA_ROOT / p


def load_subject_f1s(run_dir: Path) -> list[float]:
    """
    Per-subject F1 scores from a standard (non-RR) eval run.
    Prefers computing from saved pickle matrices (full per-subject granularity);
    falls back to the single macro F1 in results.json.
    """
    model_pkl = run_dir / "recall_matrices_model.pkl"
    human_pkls = [
        p for p in run_dir.glob("recall_matrices_*.pkl") if "model" not in p.name
    ]

    if model_pkl.exists() and human_pkls:
        with open(model_pkl, "rb") as f:
            matrices_model = pickle.load(f)
        with open(human_pkls[0], "rb") as f:
            matrices_human = pickle.load(f)

        f1s = []
        for rm_model, rm_human in zip(matrices_model, matrices_human):
            rm_h = (rm_human > 0).astype(int).flatten()
            rm_m = rm_model.flatten()
            if (rm_h == 0).all() or (rm_m == 0).all():
                continue
            prec = precision_score(rm_h, rm_m, zero_division=0)
            rec = recall_score(rm_h, rm_m, zero_division=0)
            denom = prec + rec
            f1s.append((2 * prec * rec / denom) if denom > 0 else 0.0)
        return f1s

    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No results.json found in {run_dir}")
    with open(results_path) as fh:
        r = json.load(fh)
    return [r["f1_macro"]]


def load_rr_data(run_dir: Path) -> dict[str, list[float]]:
    """
    Load repeat-reliability data from results.json.
    Returns:
        f1_vs_human  – one value per recall (mean F1 vs human across repeats)
        pairwise_f1  – one value per recall (mean pairwise F1 across repeat pairs)

    Note: pairwise_f1 is only stored as a global scalar in results.json, so it
    is returned as a single-element list for uniform handling.
    """
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No results.json found in {run_dir}")
    with open(results_path) as fh:
        r = json.load(fh)

    # per-recall mean F1 vs human (one value per recall, averaged across repeats)
    if "f1s" in r:
        f1_vs_human = [float(np.mean(scores)) for scores in r["f1s"].values()]
    else:
        f1_vs_human = [r.get("overall_mean_f1", 0.0)]

    # pairwise F1 – only a global scalar is stored
    pairwise_f1 = [float(r["mean_pairwise_f1"])] if "mean_pairwise_f1" in r else []

    return {"f1_vs_human": f1_vs_human, "pairwise_f1": pairwise_f1}


# ── drawing primitives ────────────────────────────────────────────────────────

# ── drawing primitives ────────────────────────────────────────────


def _model_color(model_label: str) -> str:
    return MODEL_COLORS.get(model_label, DEFAULT_COLOR)


def draw_model_bar(
    fig: go.Figure,
    x_center: float,
    f1s: list[float],
    model_label: str,
):
    """
    Coloured bar (mean) with SEM error bar + jittered white scatter dots on top.
    Used for standard (non-RR) plots only – plain go.Figure(), no subplot grid.
    """
    vals = np.array(f1s, dtype=float)
    mean_v = float(np.mean(vals))
    sem_v = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    color = _model_color(model_label)

    # ─ coloured bar with SEM error bar ───────────────────────────────────────
    fig.add_trace(
        go.Bar(
            x=[x_center],
            y=[mean_v],
            width=BAR_WIDTH,
            marker=dict(
                color=color,
                line=dict(color=_darken(color, 0.25), width=1.2),
            ),
            error_y=dict(
                type="data",
                array=[sem_v],
                visible=True,
                color=MEAN_COLOR,
                thickness=1.8,
                width=6,
            ),
            name=model_label,
            hovertemplate=f"<b>{model_label}</b><br>mean F1: {mean_v:.3f}<br>SEM: {sem_v:.3f}<extra></extra>",  # noqa: E501
            showlegend=False,
        )
    )

    # mean annotation just above bar top + SEM
    fig.add_annotation(
        x=x_center,
        y=mean_v + sem_v,
        text=f"{mean_v:.2f}",
        showarrow=False,
        yshift=10,
        font=dict(size=10, color=MEAN_COLOR),
    )

    # ─ jittered scatter dots (white, so they read on any bar colour) ──────────
    xs = x_center + RNG.uniform(
        -JITTER_SCALE * BAR_WIDTH, JITTER_SCALE * BAR_WIDTH, size=len(vals)
    )
    hover = [f"<b>{model_label}</b><br>F1: {v:.3f}" for v in vals]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=vals,
            mode="markers",
            marker=dict(
                color=DOT_COLOR_OVERLAY,
                size=DOT_SIZE,
                opacity=DOT_OPACITY,
                line=dict(color=_darken(color, 0.15), width=0.8),
            ),
            text=hover,
            hoverinfo="text",
            showlegend=False,
            name=model_label,
        )
    )


def _darken(hex_color: str, amount: float) -> str:
    """Return a darkened version of a hex colour (amount in 0–1)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r = max(0, int(r * (1 - amount)))
    g = max(0, int(g * (1 - amount)))
    b = max(0, int(b * (1 - amount)))
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_model_region(
    fig: go.Figure,
    x_center: float,
    f1s: list[float],
    model_label: str,
    row: int | None = None,
    col: int | None = None,
    dot_color: str = RR_DOT_COLOR,
):
    """
    Line-based mean bar + SEM + jittered dots.
    Used for RR subplot figures where go.Bar doesn't play nicely with make_subplots
    row/col routing for shapes.
    """
    vals = np.array(f1s, dtype=float)
    xs = x_center + RNG.uniform(-JITTER_SCALE, JITTER_SCALE, size=len(vals))
    hover = [f"<b>{model_label}</b><br>F1: {v:.3f}" for v in vals]

    subplot_kwargs: dict[str, int] = {"row": row, "col": col} if row is not None else {}  # type: ignore

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=vals,
            mode="markers",
            marker=dict(color=dot_color, size=DOT_SIZE, opacity=DOT_OPACITY),
            text=hover,
            hoverinfo="text",
            showlegend=False,
            name=model_label,
        ),
        **subplot_kwargs,  # type: ignore
    )

    mean_val = float(np.mean(vals))
    sem_val = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    bar_half = BAR_WIDTH / 2

    fig.add_shape(
        type="line",
        x0=x_center - bar_half,
        x1=x_center + bar_half,
        y0=mean_val,
        y1=mean_val,
        line=dict(color=MEAN_COLOR, width=2.5),
        **subplot_kwargs,
    )
    fig.add_shape(
        type="line",
        x0=x_center,
        x1=x_center,
        y0=mean_val - sem_val,
        y1=mean_val + sem_val,
        line=dict(color=MEAN_COLOR, width=1.5),
        **subplot_kwargs,
    )
    for y_cap in [mean_val - sem_val, mean_val + sem_val]:
        fig.add_shape(
            type="line",
            x0=x_center - bar_half * 0.45,
            x1=x_center + bar_half * 0.45,
            y0=y_cap,
            y1=y_cap,
            line=dict(color=MEAN_COLOR, width=1.5),
            **subplot_kwargs,
        )
    fig.add_annotation(
        x=x_center,
        y=mean_val,
        text=f"{mean_val:.2f}",
        showarrow=False,
        yshift=14,
        font=dict(size=10, color=MEAN_COLOR),
        **subplot_kwargs,
    )


def build_standard_figure(testset: str, runs: list[dict]) -> go.Figure:
    """Bar + scatter overlay, one coloured bar per model."""
    models = [r["model"] for r in runs]
    x_centers = [i * SPACING for i in range(len(models))]

    fig = go.Figure()
    for i, run in enumerate(runs):
        rdir = resolve(run["run_dir"])
        try:
            f1s = load_subject_f1s(rdir)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue
        draw_model_bar(fig, x_centers[i], f1s, run["model"])

    x_range = [x_centers[0] - SPACING * 0.75, x_centers[-1] + SPACING * 0.75]
    fig.update_layout(
        title=dict(text=TESTSET_TITLES[testset], font=dict(size=15), x=0.5),
        barmode="overlay",
        xaxis=dict(
            tickvals=x_centers,
            ticktext=models,
            tickfont=dict(size=12),
            showgrid=False,
            zeroline=False,
            range=x_range,
        ),
        yaxis=dict(
            title="F1",
            range=[0, 1.18],
            gridcolor="#e8e8e8",
            gridwidth=1,
            zeroline=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=max(500, 200 * len(models)),
        height=540,
        margin=dict(l=65, r=30, t=65, b=60),
    )
    return fig


def build_rr_figure(runs: list[dict]) -> go.Figure:
    """
    Repeat-reliability figure, controlled by RR_METRIC at the top of this file.
    "both" -> two side-by-side panels; otherwise a single panel.
    Each model gets a simple coloured bar showing the mean value only.
    """
    models = [r["model"] for r in runs]
    x_centers = [i * SPACING for i in range(len(models))]

    if RR_METRIC == "both":
        show_metrics = ["f1_vs_human", "pairwise_f1"]
    else:
        show_metrics = [RR_METRIC]

    n_panels = len(show_metrics)
    fig = make_subplots(
        rows=1,
        cols=n_panels,
        subplot_titles=[RR_METRIC_LABELS[m] for m in show_metrics],
        horizontal_spacing=0.12,
    )

    for panel_idx, metric_key in enumerate(show_metrics):
        col = panel_idx + 1
        for i, run in enumerate(runs):
            rdir = resolve(run["run_dir"])
            try:
                rr_data = load_rr_data(rdir)
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")
                continue
            f1s = rr_data.get(metric_key, [])
            if not f1s:
                print(f"  WARNING: no data for {metric_key} in {run['run_dir']}")
                continue

            mean_v = float(np.mean(f1s))
            color = _model_color(run["model"])

            fig.add_trace(
                go.Bar(
                    x=[x_centers[i]],
                    y=[mean_v],
                    width=BAR_WIDTH,
                    marker=dict(
                        color=color,
                        line=dict(color=_darken(color, 0.25), width=1.2),
                    ),
                    name=run["model"],
                    hovertemplate=f"<b>{run['model']}</b><br>F1: {mean_v:.3f}<extra></extra>",  # noqa: E501
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            fig.add_annotation(
                x=x_centers[i],
                y=mean_v,
                text=f"{mean_v:.2f}",
                showarrow=False,
                yshift=10,
                font=dict(size=10, color=MEAN_COLOR),
                row=1,
                col=col,
            )

        x_key = "xaxis" if col == 1 else f"xaxis{col}"
        y_key = "yaxis" if col == 1 else f"yaxis{col}"
        x_range = [x_centers[0] - SPACING * 0.75, x_centers[-1] + SPACING * 0.75]
        fig.update_layout(
            **{
                x_key: dict(
                    tickvals=x_centers,
                    ticktext=models,
                    tickfont=dict(size=11),
                    showgrid=False,
                    zeroline=False,
                    range=x_range,
                ),
                y_key: dict(
                    title="F1",
                    range=[0, 1.18],
                    gridcolor="#e8e8e8",
                    gridwidth=1,
                    zeroline=False,
                ),
            }
        )

    panel_w = max(400, 200 * len(models))
    fig.update_layout(
        title=dict(
            text=TESTSET_TITLES["repeat_reliability"],
            font=dict(size=15),
            x=0.5,
        ),
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=panel_w * n_panels + 80,
        height=520,
        margin=dict(l=65, r=30, t=80, b=60),
    )
    return fig


def main():
    # group runs by testset, preserving registry order
    by_testset: dict[str, list[dict]] = {}
    for run in RUNS:
        ts = run.get("testset", "unknown")
        by_testset.setdefault(ts, []).append(run)

    for testset, runs in by_testset.items():
        print(f"\nBuilding plot: {testset}  ({len(runs)} run(s))")

        fig = (
            build_rr_figure(runs)
            if testset == "repeat_reliability"
            else build_standard_figure(testset, runs)
        )

        html_out = OUTPUT_DIR / f"comparison_{testset}.html"
        png_out = OUTPUT_DIR / f"comparison_{testset}.png"
        fig.write_html(str(html_out))
        fig.write_image(str(png_out))
        print(f"  → {html_out}")
        print(f"  → {png_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
