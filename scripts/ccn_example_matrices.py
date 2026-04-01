"""
plot_example_matrices.py

For each of three datasets (alice10, memsearch10, monthiversary6), finds the
recall whose Pearson r (Claude Haiku vs human) is closest to the dataset mean,
then produces three plots for that example:
  1. Human matrix
  2. Model (Haiku) matrix
  3. Overlay (TP / FP / FN / TN colour coded)

Axes convention:
  x  – recall segments  (0 → n, left to right)
  y  – story segments   (0 → m, top to bottom; origin top-left)

All plots share the same colour conventions and are saved as PNG + HTML.
"""

import pickle
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.stats import pearsonr as scipy_pearsonr

# ── config ────────────────────────────────────────────────────────────────────

DATA_ROOT = Path("data/eval")
OUTPUT_DIR = Path("data/eval/plots/matrices")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "alice10": {
        "model_dir": "20260309_194145-cyoa_alice10-anthropic-m_claude-haiku-4-5-seed_42",
    },
    "memsearch10": {
        "model_dir": "20260310_010740-memsearch10-anthropic-m_claude-haiku-4-5-seed_42",
    },
    "monthiversary6": {
        "model_dir": "20260309_215727-cyoa_monthiversary6-anthropic-m_claude-haiku-4-5-seed_42",
    },
}

# ── overlay colour map ────────────────────────────────────────────────────────
# TP = human=1 & model=1   dark green
# TN = human=0 & model=0   light grey
# FP = human=0 & model=1   red   (model fires, human didn't)
# FN = human=1 & model=0   blue  (human fires, model missed)
OVERLAY_COLORS = {
    "TP": "#2d6a4f",  # dark green
    "TN": "#f0f0f0",  # light grey
    "FP": "#e63946",  # red
    "FN": "#457b9d",  # steel blue
}

# single-matrix colourscale: white → dark
MATRIX_COLORSCALE = [[0, "#ffffff"], [1, "#1a1a2e"]]

# fixed plot size (same as comparison plots)
PLOT_WIDTH = 400
PLOT_HEIGHT = 400

TITLE_FONT = dict(size=13)

# ── helpers ───────────────────────────────────────────────────────────────────


def resolve(d: str) -> Path:
    p = Path(d)
    return p if p.is_absolute() else DATA_ROOT / p


def _pearsonr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(scipy_pearsonr(a, b)[0])


def load_matrices(model_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return (matrices_model, matrices_human) for a run directory."""
    model_pkl = model_dir / "recall_matrices_model.pkl"
    human_pkls = [
        p for p in model_dir.glob("recall_matrices_*.pkl") if "model" not in p.name
    ]
    if not model_pkl.exists():
        raise FileNotFoundError(f"Model pickle not found: {model_pkl}")
    if not human_pkls:
        raise FileNotFoundError(f"Human pickle not found in {model_dir}")

    with open(model_pkl, "rb") as f:
        matrices_model = pickle.load(f)
    with open(human_pkls[0], "rb") as f:
        matrices_human = pickle.load(f)

    return matrices_model, matrices_human


def find_closest_to_mean(
    matrices_model: list[np.ndarray],
    matrices_human: list[np.ndarray],
) -> tuple[int, float, float]:
    """
    Compute per-subject Pearson r, then return (index, subject_r, mean_r)
    for the subject whose r is closest to the dataset mean.
    Skips all-zero human matrices.
    """
    pearsonrs = []
    valid_indices = []

    for i, (rm_m, rm_h) in enumerate(zip(matrices_model, matrices_human)):
        rm_h_flat = rm_h.flatten()
        rm_m_flat = rm_m.flatten()
        if (rm_h_flat == 0).all():
            continue
        r = _pearsonr_safe(rm_h_flat, rm_m_flat)
        pearsonrs.append(r)
        valid_indices.append(i)

    mean_r = float(np.mean(pearsonrs))
    dists = [abs(r - mean_r) for r in pearsonrs]
    best = int(np.argmin(dists))
    return valid_indices[best], pearsonrs[best], mean_r


# ── single-matrix heatmap ─────────────────────────────────────────────────────


def _heatmap_figure(
    matrix: np.ndarray,
    title: str,
    colorscale: list,
    zmin: float = 0,
    zmax: float = 1,
) -> go.Figure:
    """
    Plain heatmap. x = recall segments, y = story segments (top-down).
    matrix shape: (n_story_segs, n_recall_segs)
    """
    n_story, n_recall = matrix.shape

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            showscale=False,
            xgap=0.5,
            ygap=0.5,
        )
    )
    fig.update_layout(
        title=dict(text=title, font=TITLE_FONT, x=0.5),
        xaxis=dict(
            title="Recall segment",
            tickmode="linear",
            tick0=0,
            dtick=max(1, n_recall // 10),
            showgrid=False,
        ),
        yaxis=dict(
            title="Story segment",
            tickmode="linear",
            tick0=0,
            dtick=max(1, n_story // 10),
            showgrid=False,
            autorange="reversed",  # 0 at top, m at bottom
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        margin=dict(l=70, r=30, t=65, b=60),
    )
    return fig


# ── overlay heatmap ───────────────────────────────────────────────────────────


def _overlay_figure(
    rm_human: np.ndarray,
    rm_model: np.ndarray,
    title: str,
) -> go.Figure:
    """
    Overlay matrix with four categories:
      0 = TN (grey)  1 = FN (blue)  2 = FP (red)  3 = TP (green)
    x = recall segments, y = story segments (top-down).
    """
    h = (rm_human > 0).astype(int)
    m = (rm_model > 0).astype(int)

    overlay = np.zeros_like(h, dtype=int)
    overlay[(h == 0) & (m == 0)] = 0  # TN
    overlay[(h == 1) & (m == 0)] = 1  # FN
    overlay[(h == 0) & (m == 1)] = 2  # FP
    overlay[(h == 1) & (m == 1)] = 3  # TP

    n_story, n_recall = h.shape

    colorscale = [
        [0 / 4, OVERLAY_COLORS["TN"]],
        [1 / 4, OVERLAY_COLORS["TN"]],
        [1 / 4, OVERLAY_COLORS["FN"]],
        [2 / 4, OVERLAY_COLORS["FN"]],
        [2 / 4, OVERLAY_COLORS["FP"]],
        [3 / 4, OVERLAY_COLORS["FP"]],
        [3 / 4, OVERLAY_COLORS["TP"]],
        [4 / 4, OVERLAY_COLORS["TP"]],
    ]

    fig = go.Figure(
        go.Heatmap(
            z=overlay,
            colorscale=colorscale,
            zmin=-0.5,
            zmax=3.5,
            showscale=False,
            xgap=0.5,
            ygap=0.5,
        )
    )

    # legend as invisible scatter traces
    legend_items = [
        ("TP", OVERLAY_COLORS["TP"]),
        ("TN", OVERLAY_COLORS["TN"]),
        ("FP", OVERLAY_COLORS["FP"]),
        ("FN", OVERLAY_COLORS["FN"]),
    ]
    for label, color in legend_items:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color, symbol="square"),
                name=label,
                showlegend=True,
            )
        )

    fig.update_layout(
        title=dict(text=title, font=TITLE_FONT, x=0.5),
        xaxis=dict(
            title="Recall segment",
            tickmode="linear",
            tick0=0,
            dtick=max(1, n_recall // 10),
            showgrid=False,
        ),
        yaxis=dict(
            title="Story segment",
            tickmode="linear",
            tick0=0,
            dtick=max(1, n_story // 10),
            showgrid=False,
            autorange="reversed",
        ),
        legend=dict(
            orientation="v",
            x=1.02,
            y=0.5,
            font=dict(size=11),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        margin=dict(l=70, r=80, t=65, b=60),
    )
    return fig


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    for dataset_name, cfg in DATASETS.items():
        print(f"\nProcessing {dataset_name}...")
        model_dir = resolve(cfg["model_dir"])

        matrices_model, matrices_human = load_matrices(model_dir)
        idx, subject_r, mean_r = find_closest_to_mean(matrices_model, matrices_human)

        rm_model = matrices_model[idx]
        rm_human = matrices_human[idx]

        print(f"  dataset mean r = {mean_r:.3f}")
        print(f"  selected subject {idx}, r = {subject_r:.3f}")
        print(f"  matrix shape: {rm_human.shape}  " f"(story segs x recall segs)")

        base_title = (
            f"{dataset_name} | subject {idx} | "
            f"r = {subject_r:.3f} (dataset mean = {mean_r:.3f})"
        )

        # 1. human matrix
        fig_human = _heatmap_figure(
            rm_human,
            title=f"Human \u2013 {base_title}",
            colorscale=MATRIX_COLORSCALE,
        )
        # 2. model matrix
        fig_model = _heatmap_figure(
            rm_model,
            title=f"Claude Haiku \u2013 {base_title}",
            colorscale=MATRIX_COLORSCALE,
        )
        # 3. overlay
        fig_overlay = _overlay_figure(
            rm_human,
            rm_model,
            title=f"Overlay \u2013 {base_title}",
        )

        for tag, fig in [
            ("human", fig_human),
            ("model", fig_model),
            ("overlay", fig_overlay),
        ]:
            html_out = OUTPUT_DIR / f"{dataset_name}_{tag}.html"
            png_out = OUTPUT_DIR / f"{dataset_name}_{tag}.png"
            fig.write_html(str(html_out))
            fig.write_image(str(png_out))
            print(f"  -> {png_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
