import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rmatch import get_logger, print_config
from rmatch.load import load_story_recall_segments
from rmatch.utils import get_param_str

log = get_logger(__name__)


class Matcher:
    def __init__(self):
        self.additional_metadata = dict()
        self.usage_metrics = dict
        self.estimated_usage_metrics = dict
        self.dry_run = bool

    def get_usage(self) -> dict | None:
        return None

    def _estimate_tokens(self, query: str) -> int:
        raise NotImplementedError("Subclasses must implement this method")

    def _calculate_cost(self, in_tokens: int, out_tokens: int) -> float:
        raise NotImplementedError("Subclasses must implement this method")

    def match(
        self,
        story_name: str,
        story_segmentation_method: str | None = None,
        recall_segmentation_method: str | None = None,
        sub_ids: list[str] | None = None,
        output_scores: bool = False,
        story_recall_segments: list[tuple[str, list[str], list[str]]] | None = None,
        **kwargs,
    ) -> dict:
        """Return each recall segment with its referenced story segments.

        Parameters
        ----------
        story_name: str
            name of the story
        story_segmentation_method: str | None
            method used to segment the story
        recall_segmentation_method: str | None
            method used to segment the recall
        sub_ids: list[str]
            list of subject ids. If None, all subjects will be rated.

        Returns
        -------
        output_dict: dict
            dict with metadata and the ratings:

            {
                "matcher_name": "...",
                "story_name": "...",
                "story_segmentation_method": "...",
                "recall_segmentation_method": "...",
                "n_story_segments": ...,
                "output_scores": ...,
                **kwargs, # additional metadata from specific matcher
                "ratings": {
                    "sub-id-1": single_subject_ratings,
                    "sub-id-2": single_subject_ratings,
                    ...
                }
            }
            See compute_ratings_single_sub for the format of the single_subject_ratings.
        """
        # load
        if story_recall_segments is None:
            loaded = load_story_recall_segments(
                story_name=story_name,
                story_segmentation_method=story_segmentation_method,
                recall_segmentation_method=recall_segmentation_method,
                sub_ids=sub_ids,
            )
            story_recall_segments = loaded[0]
            story_segmentation_method = loaded[1]
            recall_segmentation_method = loaded[2]
        elif story_segmentation_method is None or recall_segmentation_method is None:
            raise ValueError(
                "When story_recall_segments is provided, "
                "story_segmentation_method and recall_segmentation_method must be set."
            )

        output_dict = {
            "matcher_name": self.matcher_name,
            "story_name": story_name,
            "story_segmentation_method": story_segmentation_method,
            "recall_segmentation_method": recall_segmentation_method,
            "n_story_segments": len(story_recall_segments[0][1]),
            "output_scores": output_scores,
            **kwargs,
        }

        # rate
        log.info("Beginning rating")
        sub_ids_for_print = [row[0] for row in story_recall_segments]
        print_config(output_dict, sub_ids=sub_ids_for_print)  # type: ignore
        ratings = self.compute_ratings(
            story_recall_segments=story_recall_segments,
            output_scores=output_scores,
            **kwargs,
        )
        output_dict["ratings"] = ratings

        return output_dict

    def compute_ratings_single_sub(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        output_scores: bool = False,
    ) -> list[tuple[int, list[int] | list[tuple[int, float]]]]:
        """Outputs single subject recall ratings as a list

        Parameters
        ----------
        story_segments: list[str]
            list of story segments
        recall_segments: list[str]
            list of recall segments
        output_scores: bool
            whether to output scores

        Returns
        -------
        single_subject_ratings: list[tuple[int, list[tuple[int, float]] | list[int]]]
            list of recall segment ids and their referenced story segment ids or scores:
            if output_scores is False:
                [
                    (recall_segment_id_1, [story_segment_id_x, story_segment_id_y, ..]),
                    (recall_segment_id_2, [story_segment_id_z, ...]),
                    ...
                ]
            else:
                [
                    (recall_segment_id_1, [
                        (story_segment_id_x, score_x),
                        (story_segment_id_y, score_y),
                        ...
                    ]),
                    (recall_segment_id_2, [story_segment_id_z, score_z, ...]),
                    ...
                ]
        """
        raise NotImplementedError("Subclasses must implement this method")

    def compute_ratings(
        self,
        story_recall_segments: list[tuple[str, list[str], list[str]]],
        output_scores: bool = False,
        **kwargs: dict,
    ) -> dict[str, list[tuple[int, list[int] | list[tuple[int, float]]]]]:
        """Outputs ratings as dict.


        Parameters
        ----------
        story_recall_segments: list[tuple[str, list[str], list[str]]]
            list of story recall segments
        output_scores: bool
            whether to output scores

        Returns
        -------
        ratings: dict
            dict of single_subject_ratings:
            If output_scores is False:
                {
                    "sub-id-1": [
                        (
                            recall_segment_id_1,
                            [
                                story_segment_id_x,
                                story_segment_id_y,
                                ...,
                            ]
                        ),
                        (
                            recall_segment_id_2,
                            [
                                story_segment_id_z,
                                story_segment_id_a,
                                ...,
                            ]
                        ),
                        ...
                    ],
                    ...
                }

            If output_scores is True:
                {
                    "sub-id-1": [
                        (
                            recall_segment_id_1,
                            [
                                (story_segment_id_x, score_x),
                                (story_segment_id_y, score_y),
                                ...
                            ]
                        ),
                        (
                            recall_segment_id_2,
                            [
                                (story_segment_id_z, score_z),
                                (story_segment_id_a, score_a),
                                ...,
                            ]
                        ),
                        ...
                    ],
                    ...
                }
        """

        ratings = dict()
        for sub_id, story_segments, recall_segments in tqdm(
            story_recall_segments, desc="(rating subs)", position=0
        ):
            ratings[sub_id] = self.compute_ratings_single_sub(
                story_segments=story_segments,
                recall_segments=recall_segments,
                output_scores=output_scores,
                **kwargs,
            )

        return ratings

    def save_to_json(self, output_dict: dict, output_path: Path | None = None) -> Path:
        """Save the output dict to a json file.

        If output_path is None, writes under.
        Otherwise writes to the given file path.
        """

        # convert potential numpy values into python native values
        # (json cannot handle numpy values)
        output_scores = output_dict["output_scores"]
        ratings_converted = dict()
        for sub_id, single_sub_ratings in output_dict["ratings"].items():
            ratings_converted[sub_id] = list()
            for recall_segment_id, story_segment_ids in single_sub_ratings:
                story_segment_ids_converted = list()
                if output_scores:
                    for story_segment_id, score in story_segment_ids:
                        if isinstance(story_segment_id, np.generic):
                            story_segment_id = story_segment_id.item()

                        if isinstance(score, np.generic):
                            score = score.item()

                        story_segment_ids_converted.append((story_segment_id, score))

                else:
                    for story_segment_id in story_segment_ids:
                        if isinstance(story_segment_id, np.generic):
                            story_segment_id = story_segment_id.item()

                        story_segment_ids_converted.append(story_segment_id)

                ratings_converted[sub_id].append(
                    (recall_segment_id, story_segment_ids_converted)
                )
        output_dict["ratings"] = ratings_converted

        story_name = output_dict["story_name"]
        param_str = get_param_str(output_dict)
        if output_path is None:
            final_path = (
                Path("data")
                / "stories-and-recalls"
                / story_name
                / "ratings"
                / f"{param_str}.json"
            )
        else:
            final_path = Path(output_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with open(final_path, "w") as f_out:
            f_out.write(json.dumps(output_dict) + "\n")
        log.info(f"Saved ratings to {final_path}")

        return final_path

    @property
    def matcher_name(self) -> str:
        return self._matcher_name

    @matcher_name.setter
    def matcher_name(self, value: str):
        self._matcher_name = value
