"""
plot_model_comparison_pearsonr.py

Produces one plot per testset for both ratings and repeat-reliability (RR):
  Ratings  : alice10, memsearch10, monthiversary6
  RR       : rr_alice10, rr_memsearch10, rr_monthiversary6

Each plot shows one coloured bar per model (mean Pearson r) with SEM error bar
and jittered white scatter dots for individual recalls/subjects.

All plots are rendered at the same fixed size (PLOT_WIDTH x PLOT_HEIGHT) so
they can be dropped directly into a paper.

RR plots show two side-by-side panels:
  left  - Pearson r vs human annotations (per recall, mean across repeats)
  right - mean pairwise Pearson r across repeats (self-consistency)
"""

import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr as scipy_pearsonr

# Choose ONE of:
#   "pearsonr_vs_human"  - left panel only
#   "pairwise_pearsonr"  - right panel only
#   "both"               - side-by-side
RR_METRIC = "pairwise_pearsonr"

# ── fixed plot dimensions (inches at 100 dpi = pixels) ───────────────────────
# All plots (ratings + RR both-panel) share these outer dimensions.
# RR single-panel will use PLOT_WIDTH // 2 so it stays proportional.
PLOT_WIDTH = 400  # px
PLOT_HEIGHT = 400  # px
MODEL_ORDER = ["Claude Opus", "Claude Haiku", "GPT-5.2", "Llama 3.3"]

# ── data root ─────────────────────────────────────────────────────────────────
DATA_ROOT = Path("data/eval")

# ── run registry ──────────────────────────────────────────────────────────────
# testset values:
#   "alice10" | "memsearch10" | "monthiversary6"       <- ratings runs
#   "rr_alice10" | "rr_memsearch10" | "rr_monthiversary6"  <- RR runs
RUNS = [
    # ── alice10 ratings ───────────────────────────────────────────────────────
    dict(
        run_dir="20260227_223616-cyoa_alice10-anthropic-m_claude-opus-4-6-seed_42",
        testset="alice10",
        model="Claude Opus",
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
    dict(
        run_dir="70B-alice10",
        testset="alice10",
        model="Llama 3.3",
    ),
    # ── memsearch10 ratings ───────────────────────────────────────────────────
    dict(
        run_dir="20260309_215439-memsearch10-anthropic-m_claude-opus-4-6-seed_42",
        testset="memsearch10",
        model="Claude Opus",
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
    dict(
        run_dir="70B-memsearch10",
        testset="memsearch10",
        model="Llama 3.3",
    ),
    # ── monthiversary6 ratings ────────────────────────────────────────────────
    dict(
        run_dir="20260330_211711-cyoa_monthiversary6-anthropic-m_claude-opus-4-6-seed_42",
        testset="monthiversary6",
        model="Claude Opus",
    ),
    dict(
        run_dir="20260309_215727-cyoa_monthiversary6-anthropic-m_claude-haiku-4-5-seed_42",
        testset="monthiversary6",
        model="Claude Haiku",
    ),
    dict(
        run_dir="20260330_212836-cyoa_monthiversary6-openai-m_gpt-5.2-seed_42",
        testset="monthiversary6",
        model="GPT-5.2",
    ),
    dict(
        run_dir="70B-monthiversary6",
        testset="monthiversary6",
        model="Llama 3.3",
    ),
    # ── alice10 RR ────────────────────────────────────────────────────────────
    dict(
        run_dir="20260401_130319-rr-cyoa_alice10-anthropic-m_claude-opus-4-6-seed_42",
        testset="rr_alice10",
        model="Claude Opus",
    ),
    dict(
        run_dir="20260310_003817-rr-cyoa_alice10-anthropic-m_claude-haiku-4-5-seed_42",
        testset="rr_alice10",
        model="Claude Haiku",
    ),
    dict(
        run_dir="20260401_132113-rr-cyoa_alice10-openai-m_gpt-5.2-seed_42",
        testset="rr_alice10",
        model="GPT-5.2",
    ),
    dict(
        run_dir="llama-3.3-70B-Instruct-alice10-rr",
        testset="rr_alice10",
        model="Llama 3.3",
    ),
    # ── memsearch10 RR ────────────────────────────────────────────────────────
    dict(
        run_dir="20260401_131330-rr-memsearch10-anthropic-m_claude-opus-4-6-seed_42",
        testset="rr_memsearch10",
        model="Claude Opus",
    ),
    dict(
        run_dir="20260310_003831-rr-memsearch10-anthropic-m_claude-haiku-4-5-seed_42",
        testset="rr_memsearch10",
        model="Claude Haiku",
    ),
    dict(
        run_dir="20260401_132123-rr-memsearch10-openai-m_gpt-5.2-seed_42",
        testset="rr_memsearch10",
        model="GPT-5.2",
    ),
    dict(
        run_dir="20260330_110508-rr-memsearch10-huggingface-m_meta-llama_Llama-3.3-70B-Instruct-seed_42",
        testset="rr_memsearch10",
        model="Llama 3.3",
    ),
    # ── monthiversary6 RR ─────────────────────────────────────────────────────
    dict(
        run_dir="20260331_013902-rr-cyoa_monthiversary6-anthropic-m_claude-opus-4-6-seed_42",
        testset="rr_monthiversary6",
        model="Claude Opus",
    ),
    dict(
        run_dir="20260401_132222-rr-cyoa_monthiversary6-anthropic-m_claude-haiku-4-5-seed_42",
        testset="rr_monthiversary6",
        model="Claude Haiku",
    ),
    dict(
        run_dir="20260331_013923-rr-cyoa_monthiversary6-openai-m_gpt-5.2-seed_42",
        testset="rr_monthiversary6",
        model="GPT-5.2",
    ),
    dict(
        run_dir="70B-rr-monthiversary6", testset="rr_monthiversary6", model="Llama 3.3"
    ),
]

# ── colours ───────────────────────────────────────────────────────────────────
MODEL_COLORS: dict[str, str] = {
    "Claude Opus": "#CC5500",  # deep burnt orange (Anthropic)
    "Claude Haiku": "#FF9A3C",  # light amber (Anthropic)
    "GPT-5.2": "#10A37F",  # OpenAI green
    "Llama 3.3": "#4A6FA5",  # meta blue
}
DEFAULT_COLOR = "#888888"
MEAN_COLOR = "#1a1a1a"

# ── layout constants ──────────────────────────────────────────────────────────
JITTER_SCALE = 0.27
BAR_WIDTH = 0.15
DOT_SIZE = 7
DOT_OPACITY = 0.85
DOT_COLOR_OVERLAY = "white"
SPACING = 0.2

# Map full model names to short labels
MODEL_SHORT_LABELS = {
    "Claude Opus": "O",
    "Claude Haiku": "H",
    "GPT-5.2": "G",
    "Llama 3.3": "L",
}

TESTSET_TITLES = {
    "alice10": "CYOA \u2013 Alice 10",
    "memsearch10": "Memsearch 10",
    "monthiversary6": "CYOA \u2013 Monthiversary 6",
    "rr_alice10": "Repeat Reliability \u2013 Alice 10",
    "rr_memsearch10": "Repeat Reliability \u2013 Memsearch 10",
    "rr_monthiversary6": "Repeat Reliability \u2013 Monthiversary 6",
}

RR_METRIC_LABELS = {
    "pearsonr_vs_human": "Pearson r vs Human",
    "pairwise_pearsonr": "Mean Pairwise Pearson r",
}

OUTPUT_DIR = Path("data/eval/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

# ── helpers ───────────────────────────────────────────────────────────────────


def resolve(run_dir: str) -> Path:
    p = Path(run_dir)
    return p if p.is_absolute() else DATA_ROOT / p


def _model_color(model_label: str) -> str:
    return MODEL_COLORS.get(model_label, DEFAULT_COLOR)


def _darken(hex_color: str, amount: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(
        max(0, int(r * (1 - amount))),
        max(0, int(g * (1 - amount))),
        max(0, int(b * (1 - amount))),
    )


def _pearsonr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(scipy_pearsonr(a, b)[0])


# ── data loaders ──────────────────────────────────────────────────────────────


def load_subject_pearsonrs(run_dir: Path) -> list[float]:
    """Per-subject Pearson r vs human from a standard ratings run."""
    model_pkl = run_dir / "recall_matrices_model.pkl"
    human_pkls = [
        p for p in run_dir.glob("recall_matrices_*.pkl") if "model" not in p.name
    ]

    if model_pkl.exists() and human_pkls:
        with open(model_pkl, "rb") as f:
            matrices_model = pickle.load(f)
        with open(human_pkls[0], "rb") as f:
            matrices_human = pickle.load(f)

        scores = []
        for rm_model, rm_human in zip(matrices_model, matrices_human):
            rm_h = rm_human.flatten()
            rm_m = rm_model.flatten()
            if (rm_h == 0).all():
                continue
            scores.append(_pearsonr_safe(rm_h, rm_m))
        return scores

    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No results.json found in {run_dir}")
    with open(results_path) as fh:
        r = json.load(fh)
    return [r["pearsonr_macro"]]


def load_rr_pearsonrs(run_dir: Path) -> dict[str, list[float]]:
    """
    Load RR Pearson r metrics.
      pearsonr_vs_human  - one value per recall (mean across repeats), from results.json
      pairwise_pearsonr  - one value per recall (mean across pairs), from pickle
    """
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No results.json found in {run_dir}")
    with open(results_path) as fh:
        r = json.load(fh)

    if "pearsonrs" in r:
        pearsonr_vs_human = [
            float(np.mean(scores)) for scores in r["pearsonrs"].values()
        ]
    else:
        pearsonr_vs_human = [r.get("overall_mean_pearsonr", 0.0)]

    pairwise_pearsonr: list[float] = []
    rr_pkl = run_dir / "recall_matrices_model_dct.pkl"
    if rr_pkl.exists():
        with open(rr_pkl, "rb") as f:
            rr_dct: dict[str, list[np.ndarray]] = pickle.load(f)
        for matrices in rr_dct.values():
            first5 = matrices[:5]
            flat = [m.flatten() for m in first5]
            pair_scores = [
                _pearsonr_safe(m_i, m_j) for m_i, m_j in combinations(flat, 2)
            ]
            if pair_scores:
                pairwise_pearsonr.append(float(np.mean(pair_scores)))
    else:
        print(f"  WARNING: RR pickle not found at {rr_pkl}")

    return {
        "pearsonr_vs_human": pearsonr_vs_human,
        "pairwise_pearsonr": pairwise_pearsonr,
    }


# ── statistics reporting ──────────────────────────────────────────────────────


def print_pearsonr_statistics(testset: str, runs: list[dict], is_rr: bool) -> None:
    """Print mean Pearson r and SEM for each model on this testset."""
    runs_by_model = {r["model"]: r for r in runs}

    print(f"\n{'=' * 60}")
    print(f"Statistics for {testset}")
    print(f"{'=' * 60}")

    if is_rr:
        # For RR, show both metrics if RR_METRIC == "both"
        if RR_METRIC == "both":
            metrics = ["pearsonr_vs_human", "pairwise_pearsonr"]
        else:
            metrics = [RR_METRIC]

        for metric in metrics:
            print(f"\n{RR_METRIC_LABELS.get(metric, metric)}:")
            print(f"{'-' * 60}")
            for model in MODEL_ORDER:
                run = runs_by_model.get(model)
                if run is None:
                    continue
                rdir = resolve(run["run_dir"])
                try:
                    rr_data = load_rr_pearsonrs(rdir)
                    scores = rr_data.get(metric, [])
                    if scores:
                        mean_r = np.mean(scores)
                        sem_r = (
                            np.std(scores, ddof=1) / np.sqrt(len(scores))
                            if len(scores) > 1
                            else 0.0
                        )
                        print(
                            f"  {model:20s}: r = {mean_r:.3f} ± {sem_r:.3f} (n={len(scores)})"  # noqa: E501
                        )
                except FileNotFoundError:
                    print(f"  {model:20s}: [data not found]")
    else:
        # For ratings
        print("\nPearson r vs Human:")
        print(f"{'-' * 60}")
        for model in MODEL_ORDER:
            run = runs_by_model.get(model)
            if run is None:
                continue
            rdir = resolve(run["run_dir"])
            try:
                scores = load_subject_pearsonrs(rdir)
                mean_r = np.mean(scores)
                sem_r = (
                    np.std(scores, ddof=1) / np.sqrt(len(scores))
                    if len(scores) > 1
                    else 0.0
                )
                print(
                    f"  {model:20s}: r = {mean_r:.3f} ± {sem_r:.3f} (n={len(scores)})"
                )
            except FileNotFoundError:
                print(f"  {model:20s}: [data not found]")


# ── drawing primitives ────────────────────────────────────────────────────────


def _draw_bar_with_scatter(
    fig: go.Figure,
    x_center: float,
    scores: list[float],
    model_label: str,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """
    Coloured bar (mean) with SEM error bar + jittered white dots.
    Works for both plain go.Figure() and make_subplots() figures.
    """
    vals = np.array(scores, dtype=float)
    mean_v = float(np.mean(vals))
    sem_v = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    color = _model_color(model_label)
    skw: dict[str, int] = {"row": row, "col": col} if row is not None else {}

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
            hovertemplate=(f"<b>{model_label}</b><br>SEM: {sem_v:.3f}<extra></extra>"),
            showlegend=False,
        ),
        **skw,
    )

    xs = x_center + RNG.uniform(
        -JITTER_SCALE * BAR_WIDTH, JITTER_SCALE * BAR_WIDTH, size=len(vals)
    )
    hover = [f"<b>{model_label}</b><br>r: {v:.3f}" for v in vals]
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
        ),
        **skw,
    )


def _axis_style(models: list[str], x_centers: list[float]) -> tuple[dict, dict]:
    """Return (xaxis_kwargs, yaxis_kwargs) dicts."""
    x_range = [x_centers[0] - SPACING * 0.75, x_centers[-1] + SPACING * 0.75]
    # Convert full model names to short labels
    short_labels = [MODEL_SHORT_LABELS.get(m, m) for m in models]
    xax = dict(
        tickvals=x_centers,
        ticktext=short_labels,
        tickfont=dict(size=32),
        showgrid=False,
        zeroline=False,
        range=x_range,
    )
    yax = dict(
        range=[-0.1, 1.18],
        gridcolor="#e8e8e8",
        tickfont=dict(size=24),
        gridwidth=1,
        zeroline=True,
        zerolinecolor="#cccccc",
        zerolinewidth=1,
    )
    return xax, yax


# ── figure builders ───────────────────────────────────────────────────────────


def build_ratings_figure(testset: str, runs: list[dict]) -> go.Figure:
    """Single-panel ratings plot: bar + scatter per model."""
    runs_by_model = {r["model"]: r for r in runs}
    models = MODEL_ORDER
    x_centers = [i * SPACING for i in range(len(models))]

    fig = go.Figure()
    for i, model in enumerate(models):
        run = runs_by_model.get(model)
        if run is None:
            continue
        rdir = resolve(run["run_dir"])
        try:
            scores = load_subject_pearsonrs(rdir)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue
        _draw_bar_with_scatter(fig, x_centers[i], scores, run["model"])

    xax, yax = _axis_style(models, x_centers)
    fig.update_layout(
        barmode="overlay",
        xaxis=xax,
        yaxis=yax,
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        margin=dict(l=60, r=20, t=3, b=25),
    )
    return fig


def build_rr_figure(testset: str, runs: list[dict]) -> go.Figure:
    """
    RR plot for one testset. Controlled by RR_METRIC at the top of the file.
    One or two panels; bar + scatter per model.
    """
    runs_by_model = {r["model"]: r for r in runs}
    models = MODEL_ORDER
    x_centers = [i * SPACING for i in range(len(models))]

    if RR_METRIC == "both":
        show_metrics = ["pearsonr_vs_human", "pairwise_pearsonr"]
    else:
        show_metrics = [RR_METRIC]

    n_panels = len(show_metrics)
    fig = make_subplots(
        rows=1,
        cols=n_panels,
        horizontal_spacing=0.14,
    )

    for panel_idx, metric_key in enumerate(show_metrics):
        col = panel_idx + 1
        for i, model in enumerate(models):
            run = runs_by_model.get(model)
            if run is None:
                continue
            rdir = resolve(run["run_dir"])
            try:
                rr_data = load_rr_pearsonrs(rdir)
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")
                continue
            scores = rr_data.get(metric_key, [])
            if not scores:
                print(f"  WARNING: no {metric_key} data in {run['run_dir']}")
                continue
            _draw_bar_with_scatter(
                fig, x_centers[i], scores, run["model"], row=1, col=col
            )

        xax, yax = _axis_style(models, x_centers)
        x_key = "xaxis" if col == 1 else f"xaxis{col}"
        y_key = "yaxis" if col == 1 else f"yaxis{col}"
        fig.update_layout(**{x_key: xax, y_key: yax})

    fig.update_layout(
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        margin=dict(l=60, r=20, t=3, b=25),
    )
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

RR_TESTSETS = {"rr_alice10", "rr_memsearch10", "rr_monthiversary6"}


def main() -> None:
    by_testset: dict[str, list[dict]] = {}
    for run in RUNS:
        ts = run.get("testset", "unknown")
        by_testset.setdefault(ts, []).append(run)

    for testset, runs in by_testset.items():
        print(f"\nBuilding plot: {testset}  ({len(runs)} run(s))")

        is_rr = testset in RR_TESTSETS

        # Print statistics
        print_pearsonr_statistics(testset, runs, is_rr)

        # Build figure
        fig = (
            build_rr_figure(testset, runs)
            if is_rr
            else build_ratings_figure(testset, runs)
        )

        html_out = OUTPUT_DIR / f"pearsonr_{testset}.html"
        png_out = OUTPUT_DIR / f"pearsonr_{testset}.png"
        fig.write_html(str(html_out))
        fig.write_image(str(png_out))
        print(f"\n  -> {html_out}")
        print(f"  -> {png_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
