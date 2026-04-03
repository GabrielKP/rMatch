import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from recall_matrix.load import (
    load_cyoa_recall_matrix_human_binary,
    load_cyoa_story_recall_segments,
    load_ratings_dict,
    load_story_recall_segments,
)
from recall_matrix.utils import ratings_single_sub_to_matrix
from scipy.stats import pearsonr
from sklearn.metrics import precision_score, recall_score


def accuracy(array_1: np.ndarray, array_2: np.ndarray) -> float:
    return np.sum(array_1 == array_2) / len(array_1)


def f1_per_subject(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── config ───────────────────────────────────────────────────────────────────
RUNS = [
    {
        "run_dir": Path(
            "data/eval/20260309_215727-cyoa_monthiversary6-anthropic-m_claude-haiku-4-5-seed_42"
        ),
        "testset": "cyoa_monthiversary6",
        "story_names": [
            "monthiversary_3",
            "monthiversary_4",
            "monthiversary_10",
            "monthiversary_14",
            "monthiversary_19",
            "monthiversary_23",
            "monthiversary_25",
        ],
    },
]

# {
#         "run_dir": Path(
#             "data/eval/20260310_010740-memsearch10-anthropic-m_claude-haiku-4-5-seed_42" # noqa: E501
#         ),
#         "testset": "memsearch10",
#         "story_names": [
#             "breadland",
#             "ednora",
#             "from_dad_to_son",
#             "heartstrings",
#             "hollow",
#             "i_love_death",
#             "ichthys",
#             "laundry",
#             "mismatched",
#             "mop",
#         ],
#     },

# {
#         "run_dir": Path(
#             "data/eval/20260227_224618-cyoa_alice10-openai-m_gpt-5.2-seed_42"
#         ),
#         "testset": "cyoa_alice10",
#         "story_names": [
#             "alice_2",
#             "alice_3",
#             "alice_4",
#             "alice_5",
#             "alice_6",
#             "alice_7",
#             "alice_8",
#             "alice_11",
#             "alice_12",
#             "alice_13",
#         ],
#     },

#    {
#         "run_dir": Path(
#             "data/eval/20260309_194145-cyoa_alice10-anthropic-m_claude-haiku-4-5-seed_42" # noqa: E501
#         ),
#         "testset": "cyoa_alice10",
#         "story_names": [
#             "alice_2",
#             "alice_3",
#             "alice_4",
#             "alice_5",
#             "alice_6",
#             "alice_7",
#             "alice_8",
#             "alice_11",
#             "alice_12",
#             "alice_13",
#         ],
#     },


METRICS = ["precision", "recall", "f1", "accuracy", "weighted_accuracy", "pearsonr"]


# ── data loading helpers ──────────────────────────────────────────────────────
def load_story_recall_segments_for_testset(testset, story_names):
    segments = []
    if testset.startswith("cyoa"):
        raw = load_cyoa_story_recall_segments(story_names=story_names)
        for story_name, sub_id, story_segs, recall_segs in raw:
            segments.append((story_name, sub_id, story_segs, recall_segs))
    elif testset.startswith("memsearch"):
        for story_name in story_names:
            single, _, _ = load_story_recall_segments(
                story_name=story_name,
                story_segmentation_method="seg_c",
                recall_segmentation_method="sentences",
            )
            for sub_id, story_segs, recall_segs in single:
                segments.append((story_name, sub_id, story_segs, recall_segs))
    return segments


def load_human_matrices_for_testset(testset, story_names, story_recall_segments):
    if testset.startswith("cyoa"):
        matrices = {}
        for story_name, sub_id, _, _ in story_recall_segments:
            rm = load_cyoa_recall_matrix_human_binary(
                story_name=story_name, sub_id=sub_id
            )
            matrices[(story_name, sub_id)] = rm
    elif testset.startswith("memsearch"):
        ratings_dicts = defaultdict(dict)
        for story_name in story_names:
            ratings_dict = load_ratings_dict(
                story_name=story_name,
                rater_name="human",
                story_segmentation_method="seg_c",
                recall_segmentation_method="sentences",
            )
            n = ratings_dict["n_story_segments"]
            for sub_id, single_sub_ratings in ratings_dict["ratings"].items():
                ratings_dicts[story_name][sub_id] = ratings_single_sub_to_matrix(
                    single_sub_ratings, n
                )
        matrices = {
            (story_name, sub_id): ratings_dicts[story_name][sub_id]
            for story_name, sub_id, _, _ in story_recall_segments
            if sub_id in ratings_dicts[story_name]
        }
    return matrices


# ── metrics computation ───────────────────────────────────────────────────────
def compute_metrics_per_subject(run_dir, testset, story_names):
    with open(run_dir / "recall_matrices_model.pkl", "rb") as f:
        matrices_model = pickle.load(f)

    story_recall_segments = load_story_recall_segments_for_testset(testset, story_names)
    human_matrices = load_human_matrices_for_testset(
        testset, story_names, story_recall_segments
    )

    filtered = [
        (story_name, sub_id, story_segs, recall_segs)
        for story_name, sub_id, story_segs, recall_segs in story_recall_segments
        if not (human_matrices.get((story_name, sub_id), np.zeros((1, 1))) == 0).all()
        and (story_name, sub_id) in human_matrices
    ]

    assert len(filtered) == len(matrices_model), (
        f"Mismatch: {len(filtered)} subjects vs {len(matrices_model)} model matrices"
    )

    records = []
    for idx, (story_name, sub_id, _, _) in enumerate(filtered):
        rm_model = matrices_model[idx]
        rm_human = (human_matrices[(story_name, sub_id)] > 0).astype(int)

        rm_model_flat = rm_model.flatten()
        rm_human_flat = rm_human.flatten()

        if (rm_model == 0).all():
            prec = rec = f1_score = pearsonr_score = acc = weighted_accuracy = 0.0
        else:
            prec = precision_score(rm_human_flat, rm_model_flat, zero_division=0)
            rec = recall_score(rm_human_flat, rm_model_flat, zero_division=0)
            f1_score = f1_per_subject(prec, rec)
            pearsonr_score = float(pearsonr(rm_human_flat, rm_model_flat)[0])  # type: ignore
            acc = accuracy(rm_human_flat, rm_model_flat)
            weighted_accuracy = prec * rec

        records.append(
            {
                "story_name": story_name,
                "sub_id": sub_id,
                "precision": prec,
                "recall": rec,
                "f1": f1_score,
                "accuracy": acc,
                "weighted_accuracy": weighted_accuracy,
                "pearsonr": pearsonr_score,
            }
        )

    return records


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_run(records, run_label, output_path):
    story_names = sorted(set(r["story_name"] for r in records))

    # assign a color per story using plotly's qualitative palette
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
    ]
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(story_names)}

    spacing = 1.5
    jitter_scale = 0.15
    rng = np.random.default_rng(42)

    fig = go.Figure()

    # one trace per story (for legend grouping)
    for story in story_names:
        story_records = [r for r in records if r["story_name"] == story]
        xs, ys, hover_texts = [], [], []

        for m_idx, metric in enumerate(METRICS):
            x_center = m_idx * spacing
            for r in story_records:
                x = x_center + rng.uniform(-jitter_scale, jitter_scale)
                xs.append(x)
                ys.append(r[metric])
                hover_texts.append(
                    f"sub: {r['sub_id']}<br>story: {r['story_name']}"
                    "<br>{metric}: {r[metric]:.3f}"
                )

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=story,
                marker=dict(color=color_map[story], size=7, opacity=0.75),
                text=hover_texts,
                hoverinfo="text",
            )
        )

    # overlay mean ± SEM per metric as a separate non-legend trace
    for m_idx, metric in enumerate(METRICS):
        x_center = m_idx * spacing
        vals = np.array([r[metric] for r in records])
        mean_val = np.mean(vals)
        sem_val = np.std(vals, ddof=1) / np.sqrt(len(vals))

        # mean line
        fig.add_shape(
            type="line",
            x0=x_center - 0.25,
            x1=x_center + 0.25,
            y0=mean_val,
            y1=mean_val,
            line=dict(color="black", width=2.5),
        )
        # SEM error bar
        fig.add_shape(
            type="line",
            x0=x_center,
            x1=x_center,
            y0=mean_val - sem_val,
            y1=mean_val + sem_val,
            line=dict(color="black", width=1.5),
        )
        # SEM caps
        for y_cap in [mean_val - sem_val, mean_val + sem_val]:
            fig.add_shape(
                type="line",
                x0=x_center - 0.1,
                x1=x_center + 0.1,
                y0=y_cap,
                y1=y_cap,
                line=dict(color="black", width=1.5),
            )
        # mean annotation
        fig.add_annotation(
            x=x_center,
            y=mean_val,
            text=f"{mean_val:.2f}",
            showarrow=False,
            yshift=12,
            font=dict(size=10, color="black"),
        )

    fig.update_layout(
        title=dict(text=run_label, font=dict(size=13)),
        xaxis=dict(
            tickvals=[i * spacing for i in range(len(METRICS))],
            ticktext=METRICS,
            tickfont=dict(size=12),
            showgrid=False,
        ),
        yaxis=dict(
            range=[-0.1, 1.1],
            title="Score",
            gridcolor="lightgrey",
            gridwidth=1,
        ),
        legend=dict(
            title="Story",
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0.8)",
        ),
        plot_bgcolor="white",
        width=1100,
        height=600,
    )

    fig.write_html(str(output_path))
    png_path = output_path.with_suffix(".png")
    fig.write_image(str(png_path))
    print(f"  Saved to {output_path} and {png_path}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for run in RUNS:
        run_dir = run["run_dir"]
        print(f"Processing {run_dir.name}...")
        records = compute_metrics_per_subject(
            run_dir=run_dir,
            testset=run["testset"],
            story_names=run["story_names"],
        )
        output_path = run_dir / "scatter_metrics.html"
        plot_run(records, run_label=run_dir.name, output_path=output_path)
