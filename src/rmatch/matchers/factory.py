from rmatch.matchers.matcher_anthropic import MatcherAnthropic
from rmatch.matchers.matcher_base import MatcherBase
from rmatch.matchers.matcher_cuda import MatcherCuda
from rmatch.matchers.matcher_huggingface import MatcherHuggingFace
from rmatch.matchers.matcher_mac import MatcherMac
from rmatch.matchers.matcher_openai import MatcherOpenAI

__all__ = ["get_matcher"]


def get_matcher(name: str, **kwargs) -> MatcherBase:
    cls = MatcherBase._registry.get(name)
    if cls is None:
        available = ", ".join(sorted(MatcherBase._registry))
        raise ValueError(f"Unknown matcher: {name!r}. Available: {available}")
    return cls(**kwargs)
