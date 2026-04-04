from rmatch import get_logger

log = get_logger(__name__)


class Matcher:
    _registry: dict[str, type["Matcher"]] = {}

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
    ) -> list[tuple[int, list[int]]]:
        """Return each recall segment with its referenced story segments.

        Parameters
        ----------
        story_segments: list[str]
            list of story segments
        recall_segments: list[str]
            list of recall segments

        Returns
        -------
        matches: list[tuple[int, list[int]]]
            list of matches:
            [
                (recall_segment_id_1, [story_segment_id_x, story_segment_id_y, ..]),
                (recall_segment_id_2, [story_segment_id_z, ...]),
                ...
            ]
        """
        raise NotImplementedError("Subclasses must implement this method")
