from typing import cast

from rmatch import get_logger

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Literal

log = get_logger(__name__)


class SegmenterVLLM:
    def __init__(
        self,
        model_name: str | None = None,
        max_new_tokens: int | None = None,
        max_model_len: int | None = None,
        api_key: str | None = None,
        prompt: str | None = None,
        # gpu params, seehttps://docs.vllm.ai/en/v0.8.0/api/offline_inference/llm.html#vllm.LLM
        tensor_parallel_size: int | None = None,  # number of GPUs to shard across
        gpu_memory_utilization: float = 0.90,
    ):
        if model_name is None:
            self.model_name = "google/gemma-4-E4B-it"
        else:
            self.model_name = model_name
        
        log.info(f"Initializing vLLM model: {self.model_name}")
        self.max_new_tokens = max_new_tokens or 2048
        log.info(f"Max new tokens: {self.max_new_tokens}")
    
        import os

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vllm is required for vllm segmentation. "
                "Install with: pip install rMatch[cuda]"
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

    def segment(
        self,
        transcript: str,
        return_raw: bool = False,
    ) -> list[str] | str:
        
        prompt = f"""I have a transcript of someone describing movies they watched. Follow these steps:

1. Split the text into sentences
2. For sentences longer than 20 words, split them at natural breaks (punctuation, topic shifts, noun-verb units)
3. Verify the segmented text is word-for-word identical to the original

Output ONLY a JSON array where each element is a segment. No preamble, no explanation, no markdown code blocks. Format:
["segment 1 text here", "segment 2 text here", "segment 3 text here"]

Example input: "I watched a really interesting documentary about ocean life. It was fascinating and educational and I learned so much about marine biology and ecosystems."

Example output:
["I watched a really interesting documentary about ocean life.", "It was fascinating and educational", "and I learned so much about marine biology and ecosystems."]

Here is the transcript to segment:
{transcript}
"""

        # Apply chat template once upfront
        formatted_prompt = self._apply_chat_template(
            [{"role": "user", "content": prompt}]
        )

        print("Segmenting...")
        response = self.llm.generate(
            [formatted_prompt],
            self.sampling_params,
        )

        response_text = response[0].outputs[0].text

        if return_raw:
            return response_text
        
        try:
            cleaned = response_text.strip()
            if cleaned.startswith('```'):
                cleaned = cleaned.split('```')[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
            
            segments = json.loads(cleaned)

            if not isinstance(segments, list):
                raise ValueError("Model output was not a JSON array")
            
            return segments
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse model output as JSON: {e}")
            print(f"Raw output: {response_text}")
            raise

    def segment_batch(
            self,
            transcripts: list[str],
    ) -> list[list[str]]:
        log.info(f"Prepping {len(transcripts)} transcripts for batch processing...")
        prompts = []
        for t in transcripts:
            prompt = f"""I have a transcript of someone describing movies they watched. Follow these steps:

1. Split the text into sentences
2. For sentences longer than 20 words, split them at natural breaks (punctuation, topic shifts, noun-verb units)
3. Verify the segmented text is word-for-word identical to the original

Output ONLY a JSON array where each element is a segment. No preamble, no explanation, no markdown code blocks. Format:
["segment 1 text here", "segment 2 text here", "segment 3 text here"]

Example input: "I watched a really interesting documentary about ocean life. It was fascinating and educational and I learned so much about marine biology and ecosystems."

Example output:
["I watched a really interesting documentary about ocean life.", "It was fascinating and educational", "and I learned so much about marine biology and ecosystems."]

Here is the transcript to segment:
{t}
"""
            # Apply chat template once upfront
            formatted_prompt = self._apply_chat_template(
                [{"role": "user", "content": prompt}]
            )
            prompts.append(formatted_prompt)
        
        log.info(f"Segmenting {len(transcripts)} transcripts (parallel)...")
        outputs = self.llm.generate(prompts, self.sampling_params)

        all_segs = []
        for i, output in enumerate(outputs):
            response_text = output.outputs[0].text

            try:
                cleaned = response_text.strip()
                if cleaned.startswith('```'):
                    cleaned = cleaned.split('```')[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                    cleaned = cleaned.strip()
                
                segments = json.loads(cleaned)

                if not isinstance(segments, list):
                    log.warning(f"Transcript {i}: Output was not a JSON array, using empty list")
                    all_segs.append([])
                else:
                    all_segs.append(segments)
                
            except (json.JSONDecodeError, ValueError) as e:
                log.error(f"Transcript {i}: Failed to parse - {e}")
                all_segs.append([])  # Empty list on failure
        
        return all_segs
    
def segs_to_df(segments: list[str]) -> pd.DataFrame:
    data = {
        "recall_segment": list(range(1, len(segments) + 1)),
        "story_segment": [''] * len(segments),
        "text": segments,
    }

    return pd.DataFrame(data)

def process_directory(
        input_dir: str,
        output_dir: str,
        model_name: str | None = None,
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float = 0.90,
        use_batch: bool = True,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {output_path}")

    transcript_files = list(input_path.glob("*.txt"))

    if not transcript_files:
        log.warning("No transcripts found")
        return
    
    log.info(f"Found {len(transcript_files)} transcript files")

    segmenter = SegmenterVLLM(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    if use_batch:
        log.info("Using batch mode")
        transcripts = []
        for file_path in transcript_files:
            with open(file_path, 'r', encoding="utf-8") as f:
                transcripts.append(f.read())
        
        all_segments = segmenter.segment_batch(transcripts)

        for file_path, segments in zip(transcript_files, all_segments):
            output_file = output_path / f"{file_path.stem}.csv"
            df = segs_to_df(segments)
            df.to_csv(output_file, index=False, encoding="utf-8")
            log.info(f"Saved: {output_file} ({len(segments)} segments)")
    else:
        log.info("Using sequential mode")
        for i, file_path in enumerate(transcript_files, 1):
            log.info(f"Processing {i}/{len(transcript_files)}: {file_path.name}")
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
            
            try:
                segments = segmenter.segment(transcript)
            except Exception as e:
                log.error(f"Failed to segment {file_path.name}: {e}")
                segments = []
            
            output_file = output_path / f"{file_path.stem}.csv"
            df = segs_to_df(segments)
            df.to_csv(output_file, index=False, encoding="utf-8")
            log.info(f"Saved: {output_file} ({len(segments)} segments)")

    log.info("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment transcripts w/ vLLM"
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
    )
    
    parser.add_argument(
        "--no-batch",
        action="store_true",
    )

    args = parser.parse_args()
    if args.output_dir is None:
        output_dir = os.path.join(args.input_dir, "segmented")
    else:
        output_dir = args.output_dir
    
    process_directory(
        input_dir=args.input_dir,
        output_dir=output_dir,
        model_name=args.model,
        use_batch=not args.no_batch,
    )
