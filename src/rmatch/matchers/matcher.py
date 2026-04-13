from collections import defaultdict

from rmatch import get_logger, matchlist_type

log = get_logger(__name__)


class Matcher:
    _registry: dict[str, type["Matcher"]] = {}
    max_retries: int

    def __init__(self, **kwargs) -> None:
        # match_key -> list over recall idx[ list over attempts [ (prompt, response) ] ]
        self.prompt_response_log: dict[
            str, list[list[tuple[str, str, set[int] | None]]]
        ] = defaultdict(list)

    def _log_prompt_response(
        self,
        match_key: str,
        idx_recall: int,
        prompt: str,
        response: str,
        parsed_response: set[int] | None,
    ) -> None:
        while len(self.prompt_response_log[match_key]) <= idx_recall:
            self.prompt_response_log[match_key].append(list())
        self.prompt_response_log[match_key][idx_recall].append(
            (prompt, response, parsed_response)
        )

    def __init_subclass__(cls, matcher_name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if matcher_name is not None:
            Matcher._registry[matcher_name] = cls

    def __new__(cls, *args, **kwargs):
        # check for Matcher, such that upon direct instantiation of
        # subclasses we don't do anything.
        if cls is Matcher:
            name = kwargs.get("matcher_name")
            if name is None:
                raise TypeError(
                    "matcher_name is required when instantiating Matcher directly"
                )
            subcls = Matcher._registry.get(name)
            if subcls is None:
                available = ", ".join(sorted(Matcher._registry))
                raise ValueError(f"Unknown matcher: {name!r}. Available: {available}")
            return object.__new__(subcls)
        return super().__new__(cls)

    @property
    def matcher_name(self) -> str:
        return self._matcher_name

    @matcher_name.setter
    def matcher_name(self, value: str):
        self._matcher_name = value

    def get_usage(self) -> dict | None:
        return None

    def _estimate_tokens(self, query: str) -> int:
        raise NotImplementedError("Subclasses must implement this method")

    def _calculate_cost(self, in_tokens: int, out_tokens: int) -> float:
        raise NotImplementedError("Subclasses must implement this method")

    def match(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        match_key: str | None = None,
    ) -> matchlist_type:
        """Return each recall segment with its referenced story segments.

        Implementations should call ``_log_prompt_response`` after each model
        interaction (prompt and raw response text) for debugging and optional export.

        Parameters
        ----------
        story_segments: list[str]
            list of story segments
        recall_segments: list[str]
            list of recall segments
        match_key: str
            a string to identify story & recall segments. Used to save prompts and
            Responses to the prompt_response_log.

        Returns
        -------
        match_list: list[tuple[int, list[int]]]
            list of matches:
            [
                (recall_segment_id_1, [story_segment_id_x, story_segment_id_y, ..]),
                (recall_segment_id_2, [story_segment_id_z, ...]),
                ...
            ]
        """
        raise NotImplementedError("Subclasses must implement this method")
