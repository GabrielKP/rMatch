import os
import re
from pathlib import Path

import torch
import transformers
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import BitsAndBytesConfig

# from vllm import LLM, SamplingParams
from recall_matrix.raters.rater import Rater


def create_pipeline(model_id, batch_size=8):
    # Use BitsAndBytes for quantization to reduce memory usage and speed up inference
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load the tokenizer and set padding token
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Create the text generation pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={
            "quantization_config": quant_config,
            "dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        device_map="auto",
        batch_size=batch_size,
    )

    return pipeline


# Define a function to evaluate LLM output
def eval_LLM_output(pipeline, prompts):
    # Define the message to be sent to the model.
    all_messages = [[{"role": "user", "content": p}] for p in prompts]

    # Generate the output
    outputs = pipeline(
        all_messages,
        max_new_tokens=1024,
    )

    # Return the output
    return [output[0]["generated_text"][-1]["content"] for output in outputs]


def vllm_initialise_engine(model_id):
    pass


def read_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content


def number_and_split_statements(statements_ls):
    numbered_statements = []
    for idx, statement in enumerate(statements_ls, start=1):
        numbered_statements.append(f"{idx}. {statement}")
    return "\n".join(numbered_statements)


def initialise_prompt(raw_segmentation, recall_statement, prompt_path):
    # Assemble them into required formats for injection into prompt
    data_dict = {
        "narrative": " ".join(raw_segmentation),
        "segmentation": number_and_split_statements(raw_segmentation),
        "alternative": recall_statement,
    }

    # Read the prompt template
    template = read_txt(prompt_path)

    prompt_out = template.format(**data_dict)
    return prompt_out, data_dict


def parse_events_from_output(raw: str) -> list[int]:
    raw = raw.strip()
    if "<>" in raw:
        return []
    match = re.search(r"<([0-9,\s]+)>", raw)
    if not match:
        return []  # silent fail?
    parsed_list = [
        int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()
    ]
    return [y - 1 for y in parsed_list]


def hf_token_login():
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    token = token if token else os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("Error: HF_TOKEN not found in .env file or in environment variables.")
        quit()


### Only necessary if we want to save the prompts and outputs for debugging
"""
def dump_to_txt(output_path, string):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f_out:
        f_out.write(string)


def add_to_txt(output_path, string):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f_out:
        f_out.write(string)


def extract_date():
    time_ls = time.ctime(time.time()).split()
    time_ls2 = time_ls[1:3] + [time_ls[4]]
    return "_".join(time_ls2)


def extract_date_time():
    time_ls = time.ctime(time.time()).split()
    time_ls2 = time_ls[1:3] + [time_ls[4]] + [time_ls[3]]
    return "_".join(time_ls2)
"""
###


class RaterHuggingFace(Rater):
    def __init__(self, model_name: str | None = None):
        self.rater_name = "huggingface"

        if model_name is None:
            self.model_name = "meta-llama/Llama-3.1-70B-Instruct"

        hf_token_login()

        self.pipeline = create_pipeline(model_name, batch_size=4)

    def compute_ratings_single_sub(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        output_scores: bool = False,
        **kwargs,
    ) -> list[tuple[int, list[int] | list[tuple[int, float]]]]:
        """
        output_txt_path = (
            Path("data")
            / "stories-and-recalls"
            / args.story_name
            / "debug"
            / "Llama-3.1-8B-Instruct"
        )

        dump_to_txt(
            Path(
                output_txt_path
                / f"{subject_ids[subj_no - 1]}-output_prompt-{date_today}.txt"
            ),
            f"Subject #{subj_no} - {extract_date_time().replace('_', ' ')}\n\n\n",
        )
        date_today = extract_date()
        """

        prompt_path = Path("data") / "prompts" / "rate_incontext_prompt.txt"

        parsed_events = []
        all_prompts = []
        for single_recall_segment in recall_segments:
            prompt, __ = initialise_prompt(
                story_segments, single_recall_segment, prompt_path
            )
            all_prompts.append(prompt)

        model_output = eval_LLM_output(self.pipeline, all_prompts)
        for i, output in enumerate(model_output):
            parsed_single_recall = parse_events_from_output(output)
            parsed_events.append([i, parsed_single_recall])

            ### Only necessary if we want to save the prompts and outputs for debugging
            """
            if output_txt_path is not None:
                add_to_txt(
                    Path(
                        output_txt_path
                        / f"{extract_date_time().replace(':', '-')}-output_prompt.txt"
                    ),
                    f"{extract_date_time().replace('_', ' ')}\n"
                    f"Prompt #{current_segment + 1}:\n"
                    f"{prompt}\n\n"
                    f"Model output:\n"
                    f"{model_output}\n\n\n",
                )
            """

        return parsed_events
