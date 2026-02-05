# Import dependencies
import transformers
import torch
import ast
import argparse
import json
from pathlib import Path

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

def initialise_prompt(raw_recall, raw_segmentation, prompt_path):
    # Assemble them into required formats for injection into prompt
    data_dict = {
        "narrative": " ".join(raw_segmentation),
        "segmentation": number_and_split_statements(raw_segmentation),
        "recall": number_and_split_statements(raw_recall)
    }

    # Read the prompt template
    template = read_txt(prompt_path)

    prompt_out = template.format(**data_dict)
    return data_dict, prompt_out

def parse_events_from_second_output(model_output):
    # Split the output into lines
    lines = model_output.split("\n")
    
    # Initialize a list to hold the parsed events
    parsed_events = []
    
    # Iterate through each line and extract numbered events
    count = 1
    for line in lines:
        line = line.strip()
        if line == "" or not (line[0].isdigit() and ':' in line):
            continue

        line_split_ls = line.split(":")
        line_split_ls[1] = line_split_ls[1].split("]")[0] + "]"
        for i in range(len(line_split_ls)):
            try:
                line_split_ls[i] = ast.literal_eval(line_split_ls[i].strip())
            except ValueError:
                print("ValueError occurred.")
                print(line_split_ls[i].strip(), "\n")

        parsed_events.append(line_split_ls)
    
    return parsed_events


def rate_incontext(
    story_name: str,
    model_id: str,
    story_segment_method: str,
    recall_segment_method: str,
    total_subjects: int,
    first_prompt_path: str | None = None,
    second_prompt_path: str | None = None,
    store_results: bool = True,
    overwrite_existing: bool = False
):
    # Define some variables for later use
    subjects = [i for i in range(1, total_subjects+1)]
    subject_ids = [f"sub-{str(i).zfill(3)}" for i in subjects]
    ratings_dict = {}

    # Aggregate information to dump to .json
    param_str = (
        f"model-{model_id}_"
        f"rsm-{recall_segment_method}_"
        f"ssm-{story_segment_method}_"
    )

    output_path = (
        Path("data")
        / "stories-and-recalls"
        / story_name
        / "ratings"
        / f"{param_str}.json"
    )

    # Load the story recall segments
    all_recall_ls = load_story_recall_segments(story_name=story_name, story_segment_method=story_segment_method, recall_segment_method=recall_segment_method, sub_ids=subject_ids)
    
    # Identify subjects that have already been calculated
    if output_path.exists() and not overwrite_existing:
        with open(output_path, "r") as f_in:
            existing_dict = json.load(f_in)
        completed_subjects = set(existing_dict["ratings"].keys())
        subject_ids = [sub_id for sub_id in subject_ids if sub_id not in completed_subjects]
        print(f"Skipping {len(completed_subjects)} subjects already present in existing results.")
        print(f"The following subjects will be processed: {subject_ids}")

    # Handle each subject in turn
    pipeline = create_pipeline(model_id)
    for subj_no in range(1, total_subjects+1):
        print(f"\n\n\nProcessing subject {subj_no}...\n\n\n")

        # Pull out the recall and original story
        raw_recall = all_recall_ls[0][subj_no-1][2]
        raw_segmentation = all_recall_ls[0][subj_no-1][1]

        # Format the first prompt and define the model ID
        data_dict, first_prompt = initialise_prompt(raw_recall, raw_segmentation, prompt_path=first_prompt_path)

        # Get the model output
        model_output = eval_LLM_output(pipeline, first_prompt)

        # Construct the second prompt
        prompt_extension = read_txt(second_prompt_path)
        second_prompt = first_prompt + "\n" + model_output + "\n" + prompt_extension

        # Get the second model output and parse it
        second_model_output = eval_LLM_output(second_prompt, model_id)
        parsed_events = parse_events_from_second_output(second_model_output)

        ratings_dict[subject_ids[subj_no-1]] = parsed_events

    # Aggregate information and dump to .json
    ## reranker_threshold, top_k, output_scores all removed.
    incontext_output_dict = {
        "param_str": param_str,
        "story_name": story_name,
        "story_segment_method": story_segment_method,
        "recall_segment_method": recall_segment_method,
        "model_id": model_id,
        "ratings": ratings_dict
    }

    if store_results:
        # If the file exists and we are not overwriting, update existing results
        if output_path.exists() and not overwrite_existing:
            print(f"Results already exist at {output_path}. Updating additional results.")
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

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-s", "--story_name", type=str, default="pieman")
    args.add_argument("-ssm", "--story_segment_method", type=str, default=None)  # sentences_corrected
    args.add_argument("-rsm", "--recall_segment_method", type=str, default=None)  # sentences
    args.add_argument("-m", "--model_id", type=str, default="meta-llama/Llama-3.1-70B-Instruct")
    args.add_argument("-ts", "--total_subjects", type=int, default=3)
    args.add_argument("-fpp", "--first_prompt_path", type=str, default="data/prompts/rate_incontext_prompt.txt")
    args.add_argument("-spp", "--second_prompt_path", type=str, default="data/prompts/rate_incontext_prompt_extended.txt")
    args.add_argument("-dsr", "--dont_save_results", action="store_false", default=True)
    args.add_argument("-o", "--overwrite_existing", action="store_true", default=False)
    args = args.parse_args()

    rate_incontext(
        story_name= args.story_name,
        model_id= args.model_id,
        story_segment_method= args.story_segment_method,
        recall_segment_method= args.recall_segment_method,
        total_subjects= args.total_subjects,
        first_prompt_path= args.first_prompt_path,
        second_prompt_path= args.second_prompt_path,
        store_results= args.dont_save_results,
    )

    # uv run src/recall_matrix/rate_incontext.py -ssm sentences_corrected -rsm sentences -ts 2 -o