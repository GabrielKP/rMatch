import argparse

from recall_matrix.plot_single_sub import plot_single_sub
from recall_matrix.raters import RaterHuggingFace, RaterOpenAI, RaterReranker


def rate_binary(
    rater_name: str,
    story_name: str,
    story_segmentation_method: str | None = None,
    recall_segmentation_method: str | None = None,
    output_scores: bool = False,
    sub_ids: list[str] | None = None,
    # shared parameters
    model_name: str | None = None,
    device: str | None = None,
    # reranker specific parameters
    reranker_threshold: float | None = None,
    top_k: int = 5,
):
    """Rate whether a recall segment refers to a story segment.

    Parameters
    ----------
    rater_name: str
        Name of the rater to use.
    story_name: str
        Name of the story to rate.
    story_segmentation_method: str | None
        Method to segment story into parts. Will try to auto-select if None.
    recall_segmentation_method: str | None
        Method to segment recall into parts. Will try to auto-select if None.
    output_scores: bool
        Whether to output the scores of the story segments for each recall segment.
        If True, the output will additionaly contain the score for each story segment.
    sub_ids: list[str] | None
        The subject ids to rate. If None, will rate all subjects.
    model_name: str
        [reranker] Name of the model to use for the reranker.
    reranker_threshold: float
        [reranker] Threshold above which a story-segment score counts as recalled.
    top_k: int
        [reranker] Number of top k story segments to consider for each recall segment.
    device: str | None
        [reranker] Device to use for the reranker. If None, will be autoselected.
    """
    if rater_name == "reranker":
        rater = RaterReranker(model_name=model_name, device=device)
    elif rater_name == "openai":
        rater = RaterOpenAI(model_name=model_name)
    elif rater_name == "huggingface":
        rater = RaterHuggingFace(model_name=model_name)
    else:
        raise ValueError(f"Invalid argument: {rater_name=}")

    output_dict = rater.rate(
        story_name=story_name,
        story_segmentation_method=story_segmentation_method,
        recall_segmentation_method=recall_segmentation_method,
        sub_ids=sub_ids,
        output_scores=output_scores,
        reranker_threshold=reranker_threshold,  # type: ignore
        top_k=top_k,  # type: ignore
        device=device,  # type: ignore
    )
    output_path = rater.save_to_json(output_dict)

    # plot a single sub
    if not output_dict["output_scores"]:
        sub_id = list(output_dict["ratings"].keys())[0]
        plot_single_sub(sub_id=sub_id, paths_ratings=[output_path])


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-r",
        "--rater_name",
        choices=["reranker", "openai", "huggingface"],
        default="reranker",
        help="Name of the rater to use. Default is 'reranker'.",
    )
    args.add_argument(
        "-s",
        "--story_name",
        type=str,
        default="pieman",
        help="Name of the story to rate.",
    )
    args.add_argument(
        "-ssm",
        "--story_segmentation_method",
        type=str,
        default=None,
        help=(
            "Method to segment story into parts."
            " Will try to auto-select if none is provided."
        ),
    )
    args.add_argument(
        "-rsm",
        "--recall_segmentation_method",
        type=str,
        default=None,
        help=(
            "The method to use for recall segmentation."
            "Will auto-select if none is provided."
        ),
    )
    args.add_argument(
        "--output_scores",
        action="store_true",
        default=False,
        help=(
            "Whether to output the scores of the story segments for each"
            " recall segment. If True, the output will additionaly contain the"
            " score for each story segment."
        ),
    )
    args.add_argument(
        "--sub_ids",
        "--sub-ids",
        type=str,
        default=None,
        nargs="+",
        help="The subject ids to rate. If None, will rate all subjects.",
    )
    args.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="BAAI/bge-reranker-v2-m3",
        help=(
            "[reranker, openai, huggingface] Name of the model to use for the reranker."
        ),
    )
    args.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "[reranker, huggingface] Device to use for the reranker."
            "If None, will be autoselected."
        ),
    )
    args.add_argument(
        "-rt",
        "--reranker_threshold",
        type=float,
        default=0.09,
        help=(
            "[reranker] Threshold above which a story-segment score counts as recalled."
        ),
    )
    args.add_argument(
        "-tk",
        "--top_k",
        type=int,
        default=5,
        help=(
            "[reranker] Number of top k story segments to consider"
            " for each recall segment."
        ),
    )

    args = args.parse_args()
    rate_binary(
        rater_name=args.rater_name,
        story_name=args.story_name,
        story_segmentation_method=args.story_segmentation_method,
        recall_segmentation_method=args.recall_segmentation_method,
        model_name=args.model_name,
        output_scores=args.output_scores,
        sub_ids=args.sub_ids,
        reranker_threshold=args.reranker_threshold,
        top_k=args.top_k,
        device=args.device,
    )
