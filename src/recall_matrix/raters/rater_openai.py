from recall_matrix.raters.rater import Rater


class RaterOpenAI(Rater):
    def __init__(self, model_name: str | None = None):
        self.rater_name = "openai"

        if model_name is None:
            # default model!
            pass

    def single_subject_ratings(
        self, story_segments: list[str], recall_segments: list[str]
    ) -> list[tuple[int, list[int] | list[tuple[int, float]]]]:
        raise NotImplementedError("Dhruva!")
