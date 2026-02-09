# Import dependencies
import argparse
import ast
import json
import re
import time
from pathlib import Path

import torch
import transformers

# Import functions
from recall_matrix.load import load_story_recall_segments


# Define a function to number and split statements
def number_and_split_statements(statements_ls):
    numbered_statements = []
    for idx, statement in enumerate(statements_ls, start=1):
        numbered_statements.append(f"{idx}. {statement}")
    return "\n".join(numbered_statements)


# Define a function to create the pipeline
def create_pipeline(model_id):
    # Create the text generation pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"dtype": torch.bfloat16},
        device_map="auto",
    )

    return pipeline


# Define a function to evaluate LLM output
def eval_LLM_output(pipeline, prompt):
    # Define the message to be sent to the model.
    messages = [
        {"role": "user", "content": prompt},
    ]

    # Generate the output
    outputs = pipeline(
        messages,
        max_new_tokens=8192,
    )

    # Return the output
    return outputs[0]["generated_text"][-1]["content"]


def read_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content


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


def validate_and_store_results(
    output_path, incontext_output_dict, overwrite_existing, store_results
):
    if store_results:
        # If the file exists and we are not overwriting, update existing results
        if output_path.exists() and not overwrite_existing:
            print(
                f"Results already exist at {output_path}. Updating additional results."
            )
            with open(output_path, "r") as f_in:
                existing_dict = json.load(f_in)
            existing_dict["ratings"].update(incontext_output_dict["ratings"])
            with open(output_path, "w") as f_out:
                f_out.write(json.dumps(existing_dict))
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f_out:
                f_out.write(json.dumps(incontext_output_dict))
            if overwrite_existing:
                print(f"Overwrote existing results at {output_path}.")
    return incontext_output_dict


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

    return parsed_list


def rate_incontext(
    story_name: str,
    model_id: str,
    story_segment_method: str,
    recall_segment_method: str,
    subjects: list | None = [],
    prompt_path: str | None = None,
    output_txt_path: str | None = None,
    store_results: bool = True,
    overwrite_existing: bool = False,
):
    # Define some variables for later use
    subject_ids = [f"sub-{str(i).zfill(3)}" for i in subjects]
    ratings_dict = {}

    # Aggregate information to dump to .json
    param_str = (
        f"model-{model_id}_rsm-{recall_segment_method}_ssm-{story_segment_method}_"
    )

    json_output_path = (
        Path("data")
        / "stories-and-recalls"
        / story_name
        / "ratings"
        / f"{param_str}.json"
    )

    # Load the story recall segments
    story_and_recall_segments, _, _ = load_story_recall_segments(
        story_name=story_name,
        story_segment_method=story_segment_method,
        recall_segment_method=recall_segment_method,
        sub_ids=subject_ids,
    )

    # Identify subjects that have already been calculated
    if json_output_path.exists() and not overwrite_existing:
        with open(json_output_path, "r") as f_in:
            existing_dict = json.load(f_in)
        completed_subjects = set(existing_dict["ratings"].keys())
        subjects = [
            subj
            for subj in subjects
            if f"sub-{str(subj).zfill(3)}" not in completed_subjects
        ]
        print(
            f"Skipping {len(completed_subjects)} subjects"
            " already present in existing results."
        )
        print(
            f"The following subjects will be processed: "
            f"{', '.join([f'sub-{str(subj).zfill(3)}' for subj in subjects])}"
        )

    # Aggregate information and dump to .json
    ## reranker_threshold, top_k, output_scores all removed.
    incontext_output_dict = {
        "param_str": param_str,
        "story_name": story_name,
        "story_segment_method": story_segment_method,
        "recall_segment_method": recall_segment_method,
        "model_id": model_id,
    }
    date_today = extract_date()

    # Handle each subject in turn
    pipeline = create_pipeline(model_id)
    for subj_no in subjects:
        print(f"\n\n\nProcessing subject {subj_no}...\n\n\n")

        # Pull out the recall and original story
        _, story_segments, recall_segments = story_and_recall_segments[subj_no - 1]

        dump_to_txt(
            Path(
                output_txt_path
                / f"{subject_ids[subj_no - 1]}-output_prompt-{date_today}.txt"
            ),
            f"Subject #{subj_no} - {extract_date_time().replace('_', ' ')}\n\n\n",
        )

        parsed_events = []
        current_segment = 0
        for single_recall_segment in recall_segments:
            print(f"Processing segment {current_segment + 1}... for subject #{subj_no}")

            prompt, __ = initialise_prompt(
                story_segments, single_recall_segment, prompt_path
            )

            # Get the model output
            model_output = eval_LLM_output(pipeline, prompt)
            parsed_single_recall = parse_events_from_output(model_output)
            parsed_events.append([current_segment, parsed_single_recall])

            if output_txt_path is not None:
                add_to_txt(
                    Path(
                        output_txt_path
                        / f"{subject_ids[subj_no - 1]}-output_prompt-{date_today}.txt"
                    ),
                    f"""{extract_date_time().replace("_", " ")}\n
                    Prompt #{current_segment + 1}:\n
                    {prompt}\n\n
                    Model output:\n
                    {model_output}\n\n\n""",
                )

            current_segment += 1
            print("\n\n\n")

        ratings_dict[subject_ids[subj_no - 1]] = parsed_events
        incontext_output_dict["ratings"] = ratings_dict
        incontext_output_dict = validate_and_store_results(
            json_output_path, incontext_output_dict, overwrite_existing, store_results
        )

    return incontext_output_dict


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-s", "--story_name", type=str, default="pieman")
    args.add_argument(
        "-ssm", "--story_segment_method", type=str, default=None
    )  # sentences_corrected
    args.add_argument(
        "-rsm", "--recall_segment_method", type=str, default=None
    )  # sentences
    args.add_argument(
        "-m", "--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
    )
    args.add_argument("-fs", "--first_subject", type=int, default=1)
    args.add_argument("-ls", "--last_subject", type=int, default=3)
    args.add_argument(
        "-pp",
        "--prompt_path",
        type=str,
        default="data/prompts/rate_incontext_prompt.txt",
    )
    args.add_argument("-dsr", "--dont_save_results", action="store_false", default=True)
    args.add_argument("-o", "--overwrite_existing", action="store_true", default=False)
    args = args.parse_args()

    prompt_dump_path = Path("data") / "stories-and-recalls" / args.story_name / "debug"

    rate_incontext(
        story_name=args.story_name,
        model_id=args.model_id,
        story_segment_method=args.story_segment_method,
        recall_segment_method=args.recall_segment_method,
        subjects=list(range(args.first_subject, args.last_subject + 1)),
        prompt_path=args.prompt_path,
        output_txt_path=prompt_dump_path,
        store_results=args.dont_save_results,
        overwrite_existing=args.overwrite_existing,
    )

    # uv run src/recall_matrix/rate_incontext.py -ssm sentences_corrected -rsm sentences -fs 1 -ls 3 -o # noqa: E501
