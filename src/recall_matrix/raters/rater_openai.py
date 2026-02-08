import re

from openai import OpenAI
from tqdm import tqdm

from recall_matrix import ENV, get_logger
from recall_matrix.raters.rater import Rater

log = get_logger(__name__)


class RaterOpenAI(Rater):
    def __init__(self, model_name: str | None = None):
        self.rater_name = "openai"

        if model_name is None:
            self.model_name = "gpt-4.1"
            log.info(f"Initializing model to default: {self.model_name}")
        else:
            self.model_name = model_name
            log.info(f"Initializing model: {self.model_name}")

        self.client = OpenAI(api_key=ENV["OPENAI_API_KEY"])

    def build_message(
        self, recall_segment: str, story: str, story_segments: str
    ) -> str:
        message = f"""This is the original text:
{story}

It can be broken down into the following independent pieces of information:
{story_segments}

Here is a single clause from a participant's recall of the story:
"{recall_segment}"

Which of the numbered information pieces above are expressed by this recall clause?
List all applicable numbers. If none apply, return the string "<NONE>".

Return ONLY a set of numbers in <>, for example:
<3, 7, 12>
"""
        return message

    def format_story_segments(self, story_segments: list[str]) -> str:
        return "\n".join(
            f"{i+1}. {seg.strip()}" for i, seg in enumerate(story_segments)
        )

    def format_story(self, story_segments: list[str]) -> str:
        return " ".join(seg.strip() for seg in story_segments)

    def parse(self, raw: str) -> set[int]:
        raw = raw.strip()

        if "<NONE>" in raw:
            return set()

        match = re.search(r"<([0-9,\s]+)>", raw)
        if not match:
            log.warning(f"failed to parse model output: {raw}")
            return set()

        parsed_set = {
            int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()
        }

        return parsed_set

    def compute_ratings_single_sub(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        output_scores: bool = False,
    ) -> list[tuple[int, list[int]]]:
        if output_scores:
            raise NotImplementedError(
                "OpenAI rater current does not support output_scores = True"
            )

        story = self.format_story(story_segments)
        story_segs_formatted = self.format_story_segments(story_segments)
        ratings: list[tuple[int, list[int]]] = []

        for idx, recall_seg in enumerate(
            tqdm(recall_segments, desc="(rating recall segments)", position=1)
        ):
            query = self.build_message(recall_seg, story, story_segs_formatted)
            response = self.client.responses.create(model=self.model_name, input=query)

            parsed_response = self.parse(response.output_text)
            story_indices = sorted(idx - 1 for idx in parsed_response)
            ratings.append((idx, story_indices))

        return ratings
