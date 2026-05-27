import json
import os
import re
from typing import cast

import pandas as pd

from rmatch import get_logger

log = get_logger(__name__)
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


class SegmenterCuda:
    def __init__(
        self,
        model_name: str | None = None,
        max_new_tokens: int | None = None,
        max_model_len: str | int | None = None,
        api_key: str | None = None,
        # gpu params, see https://docs.vllm.ai/en/v0.8.0/api/offline_inference/llm.html#vllm.LLM
        tensor_parallel_size: int | None = None,  # number of GPUs to shard across
        gpu_memory_utilization: float = 0.90,
        max_retries: int = 10,
        granularity: str = "idea",
    ):
        if model_name is None:
            self.model_name = "google/gemma-4-31B-it"
        else:
            self.model_name = model_name

        log.info(f"Initializing vLLM model with CUDA: {self.model_name}")
        self.max_new_tokens = max_new_tokens or 4096
        self.max_retries = max_retries or 10
        log.info(f"Max new tokens: {self.max_new_tokens}")
        log.info(f"Max retries: {self.max_retries}")

        if granularity == "idea" or granularity == "event":
            self.granularity = granularity
        else:
            raise ValueError("Granularity must be 'idea' or 'event'")
        log.info(f"Setting granularity: {self.granularity}")

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "SegmenterCuda requires vllm. "
                "Install rMatch with it: pip install rMatch[cuda]"
            )

        if tensor_parallel_size is None:
            import torch

            tensor_parallel_size = max(1, torch.cuda.device_count())
            log.info(
                f"GPUs (tensor_parallel_size): {tensor_parallel_size} (Auto-detected)"
            )
        else:
            log.info(f"GPUs (tensor_parallel_size): {tensor_parallel_size}")

        log.info(f"GPU memory utilization: {gpu_memory_utilization}")
        if max_model_len == "auto":
            self.max_model_len = None
        else:
            self.max_model_len = max_model_len or 90000
        log.info(f"Max model length: {self.max_model_len}")

        hf_token = api_key or os.environ.get("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token  # vLLM picks this up automatically

        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=self.max_model_len,  # adjust to your context needs
            trust_remote_code=True,  # needed for Gemma-4
        )

        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # type: ignore

        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=0.0,  # greedy; set higher for sampling
        )

    def _build_prompt(self, transcript: str, attempt: int = 0) -> str:
        match attempt:
            case 0:
                attempt_str = ""
            case 1:
                attempt_str = "\nTry hard to keep the verbatim text.\n"
            case 2:
                attempt_str = "\nBe extra careful to keep the verbatim text.\n"
            case 3:
                attempt_str = (
                    "\nIn past attempts, you made mistakes, try to avoid them.\n"
                )
            case _:
                attempt_str = f"KEEP THE VERBATIM TEXT. ATTEMPT {attempt}.\n"

        if self.granularity == "event":
            return f"""
I have a transcript of someone describing movies they watched.
I need you to split the text into events. Do not change or remove any words from the text
Use the following key points to complete the task:
1. An event is an ongoing coherent situation.
2. When segmenting, focus on natural event boundaries.
3. Verify the segmented text is word-for-word identical to the original.
    a. Even if you think that a word was duplicated (for example the same word is repeated twice), keep both instances in
    b. The provided clause segmentation should be 100% identical to the original transcription and the text words

Output ONLY a JSON array where each element is an event. No preamble, no explanation, no markdown code blocks. Format:
["Event 1 text here", "Event 2 text here", "Event 3 text here"]

Example input: "I watched a really interesting documentary about ocean life. It was fascinating and educational and I learned so much about marine biology and ecosystems. After watching that, I went to the grocery store to buy some lemons. They were at a good price too! When I came home, I made a tart with them."

Example output:
["I watched a really interesting documentary about ocean life. It was fascinating and educational and I learned so much about marine biology and ecosystems.", "After watching that, I went to the grocery store to buy some lemons. They were at a good price too!", "When I came home, I made a tart with them."]
{attempt_str}
Here is the transcript to segment:
{transcript}
"""  # noqa: E501
        return f"""
I have a transcript of someone describing movies they watched. Follow these steps:

1. Split the text into linguistic clauses. Do not change or remove any words from the text
2. When segmenting, focus on punctuation, topic shifts, and new noun-verb units (natural breaks)
    a. DO NOT add any punctuation or fix any grammatical mistakes in the transcript, it should be exactly preserved
    b. Maintain the exact verbatim text in the original transcript
    c. DO NOT change capitalization, preserve letter cases
3. Be sure to leave phrases such as "oh my god" intact and in their own events
4. Segment into clauses where every new finite verb with its own (even implied) subject starts a new clause; coordinators that launch a fresh verb = split; keep fillers intact; preserve exact wording
5. Include lead in words (like "Well,") with the clause that immediately follows it
6. Make sure that filler or transition words ("and", "but", "oh") are not their own clauses
7. Verify the segmented text is word-for-word identical to the original
    a. Even if you think that a word was duplicated (for example the same word is repeated twice), keep both instances in
    b. The provided clause segmentation should be 100% identical to the original transcription and the text words

CRITICAL NOTE: Independent clauses that represent a single thought should be kept together.
For instance, “He said that he was sad” is a single thought. Do NOT break it up into two clauses (i.e., “He said” and “that he was sad”).
As another example, “I think the time jumped” is a single thought. Do NOT break it up into two clauses (i.e., “I think” and “the time jumped”).

Output ONLY a JSON array where each element is a segment. No preamble, no explanation, no markdown code blocks. Format:
["segment 1 text here", "segment 2 text here", "segment 3 text here"]

Example input: "I watched a really interesting documentary about ocean life. It was fascinating and educational and I learned so much about marine biology and ecosystems."

Example output:
["I watched a really interesting documentary about ocean life.", "It was fascinating and educational", "and I learned so much about marine biology and ecosystems."]
{attempt_str}
Here is the transcript to segment:
{transcript}
"""  # noqa: E501

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """Convert messages list to single prompt string using model's template."""
        return cast(
            str,
            self.tokenizer.apply_chat_template(
                messages,  # type: ignore[arg-type]
                tokenize=False,
                add_generation_prompt=True,
            ),
        )

    def _validate_segments(
        self, transcript: str, segments: list[str]
    ) -> tuple[bool, str | None]:
        SLACK = 2
        transcript = re.sub(r"\s+", " ", transcript)
        cursor = 0
        for i, seg in enumerate(segments):
            seg = re.sub(r"\s+", " ", seg)
            idx = transcript.find(seg, cursor, cursor + len(seg) + SLACK)
            if idx == -1:
                snippet_start = max(0, cursor - 50)
                snippet_end = min(len(transcript), cursor + 50)
                context = transcript[snippet_start:snippet_end]

                return False, (
                    f"segment {i} not found verbatim\n"
                    f"SEGMENT:\n{repr(seg)}\n"
                    f"CONTEXT AROUND CURSOR:\n{repr(context)}"
                )
            cursor = idx + len(seg)
        return True, None

    def _validate_segments_advanced_recovery(
        self, transcript: str, segments: list[str]
    ) -> tuple[bool, dict[int, tuple[str, str]]]:
        SLACK = 2
        transcript = re.sub(r"\s+", " ", transcript).strip()
        cursor = 0
        failures = {}

        i = 0
        while i < len(segments):
            seg = re.sub(r"\s+", " ", segments[i]).strip()
            idx = transcript.find(seg, cursor, cursor + len(seg) + SLACK)
            # fail case
            if idx == -1:
                # check for simple letter case failure
                idx_case_insensitive = transcript.lower().find(
                    seg.lower(), cursor, cursor + len(seg) + SLACK
                )
                if idx_case_insensitive != -1:
                    segments[i] = transcript[
                        idx_case_insensitive : idx_case_insensitive + len(seg)
                    ]
                    cursor = idx_ci + len(seg)
                    i += 1
                    continue

                snippet_start = max(0, cursor - 50)
                snippet_end = min(len(transcript), cursor + 50)
                context = transcript[snippet_start:snippet_end]

                failures[i] = (
                    (
                        f"segment {i} not found verbatim\n"
                        f"SEGMENT:\n{repr(seg)}\n"
                        f"CONTEXT AROUND CURSOR:\n{repr(context)}"
                    ),
                    "invalid",
                )

                j = i + 1  # search for next seg
                recovered = False
                while j < len(segments):
                    recovery_seg = re.sub(r"\s+", " ", segments[j]).strip()
                    recovery_idx = transcript.find(recovery_seg, cursor)

                    # recovered case
                    if recovery_idx != -1:
                        for k in range(i + 1, j):
                            skipped_seg = re.sub(r"\s+", " ", segments[k]).strip()
                            failures[k] = (
                                (
                                    f"segment {k} skipped during recovery (between {i} and {j})\n"
                                    f"SEGMENT:\n{repr(skipped_seg)}"
                                ),
                                "skipped",
                            )
                        cursor = recovery_idx + len(recovery_seg)
                        i = j + 1
                        recovered = True
                        break
                    j += 1

                if not recovered:
                    for k in range(i + 1, len(segments)):
                        unreachable_seg = re.sub(r"\s+", " ", segments[k]).strip()
                        failures[k] = (
                            (
                                f"segment {k} unreachable after failure at {i}\n"
                                f"SEGMENT:\n{repr(unreachable_seg)}"
                            ),
                            "unreachable",
                        )
                    break

            # success case, goto next segment + advance cursor
            else:
                cursor = idx + len(seg)
                i += 1

        remaining = transcript[cursor:].strip()
        if remaining and len(remaining) > SLACK:
            failures[len(segments)] = (
                (
                    "original transcript not fully covered\n"
                    f"REMAINDER:\n{repr(remaining[:200])}"
                ),
                "transcripts not fully covered",
            )
        valid = len(failures) == 0
        return valid, failures

    def segment(
        self,
        transcript: str,
    ) -> pd.DataFrame | None:
        return self.segment_batch([transcript])[0]

    def segment_batch(
        self, transcripts: list[str], labels: list[str] | None = None
    ) -> list[pd.DataFrame | None]:
        log.info(f"Prepping {len(transcripts)} transcripts")
        if labels is not None and len(labels) != len(transcripts):
            log.warning("Sufficient labels not provided, defaulting to indices")
            labels = None

        labels_processed = [
            labels[i] if labels is not None else f"Transcript #{i+1}"
            for i in range(len(transcripts))
        ]

        results: list[pd.DataFrame | None] = [None for _ in range(len(transcripts))]
        indices_pending = list(range(len(transcripts)))

        for attempt in range(self.max_retries):
            if not indices_pending:
                break
            prompts = list()
            for idx_pending in indices_pending:
                prompt = self._build_prompt(transcripts[idx_pending], attempt=attempt)

                formatted_prompt = self._apply_chat_template(
                    [{"role": "user", "content": prompt}]
                )
                prompts.append(formatted_prompt)

            log.info(f"Segmenting {len(indices_pending)} transcripts")
            outputs = self.llm.generate(prompts, self.sampling_params)

            still_pending: list[int] = list()
            for idx_pending, output in zip(indices_pending, outputs):
                # parse response
                response_text = output.outputs[0].text

                try:
                    cleaned = response_text.strip()
                    if cleaned.startswith("```"):
                        cleaned = cleaned.split("```")[1]
                        if cleaned.startswith("json"):
                            cleaned = cleaned[4:]
                        cleaned = cleaned.strip()

                    segments = json.loads(cleaned)

                    if not isinstance(segments, list):
                        log.warning(
                            f"{labels_processed[idx_pending]}: Output was not a JSON array"
                        )
                        still_pending.append(idx_pending)
                        continue

                    # try to validate for verbatim preservation
                    valid, failures = self._validate_segments_advanced_recovery(
                        transcripts[idx_pending], segments
                    )

                    # build dataframe
                    segment_data = []
                    for i, seg in enumerate(segments):
                        is_failed = i in failures
                        error = failures.get(i, None)

                        segment_data.append(
                            {
                                "idx": i + 1,
                                "segment": seg,
                                "failed": is_failed,
                                "error_type": error[1] if error is not None else error,
                                "error_msg": error[0] if error is not None else error,
                            }
                        )
                    # add remainder errors to results df
                    remainder_error = failures.get(len(segments), None)
                    if remainder_error is not None:
                        segment_data.append(
                            {
                                "idx": len(segments) + 1,
                                "segment": None,
                                "failed": True,
                                "error_type": remainder_error[1],
                                "error_msg": remainder_error[0],
                            }
                        )
                    results[idx_pending] = pd.DataFrame(segment_data)

                    if not valid:
                        # validation failed
                        log.warning(
                            f"{RED}[VALIDATION FAILED]{RESET} {labels_processed[idx_pending]}\n"
                        )
                        for _, error_msg in failures.items():
                            log.warning(f"{YELLOW}{error_msg}{RESET}")
                        still_pending.append(idx_pending)
                        continue

                except (json.JSONDecodeError, ValueError) as e:
                    log.error(f"{labels_processed[idx_pending]}: Failed to parse - {e}")
                    still_pending.append(idx_pending)
                    continue

            if len(still_pending) > 0:
                log.info(
                    f"Attempt {attempt + 1}/{self.max_retries}: "
                    f"{len(still_pending)}/{len(indices_pending)}"
                    " transcripts need retry"
                )
            indices_pending = still_pending

        for idx_pending in indices_pending:
            log.warning(
                f"All {self.max_retries} attempts failed for transcript {idx_pending}"
            )
        return results
