#!/usr/bin/env python3
"""
Tokenize story transcripts and participant recalls using NLTK sentence tokenizer.
Outputs CSV files with sentence segmentation to a fresh directory structure.
"""

import shutil
from pathlib import Path

import nltk
import pandas as pd

# Download the punkt tokenizer if not already available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download("punkt")


def clean_text(text):
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("”", '"')
    text = text.replace("“", '"')
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "--")
    text = text.replace("\u2026", "...")

    return text


def tokenize_story(input_file, output_file, story_name):
    """
    Tokenize a story file and save as CSV with columns: segment, text

    Args:
        input_file: Path to the story.txt file
        output_file: Path to save transcript-storyname.csv
        story_name: Name of the story
    """
    print(f"Tokenizing story: {input_file}")

    # Read the story text
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    text = clean_text(text)
    # Tokenize into sentences
    sentences = nltk.sent_tokenize(text)

    # Create DataFrame
    df = pd.DataFrame(
        {"story_segment": range(1, len(sentences) + 1), "text": sentences}
    )

    # Save to CSV
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"  → Created {output_file} with {len(sentences)} sentences")


def tokenize_recall(input_file, output_file, story_name, subject_id):
    """
    Tokenize a recall file and save as CSV with columns: segment, events, text

    Args:
        input_file: Path to the recall .txt file (e.g., sub01.txt)
        output_file: Path to save the tokenized CSV (e.g., sub-001-recall-storyname.csv)
        story_name: Name of the story
        subject_id: Subject identifier (e.g., "001")
    """
    print(f"Tokenizing recall: {input_file}")

    # Read the recall text
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    text = clean_text(text)
    # Tokenize into sentences
    sentences = nltk.sent_tokenize(text)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "recall_segment": range(1, len(sentences) + 1),
            "story_segments": "",  # Empty string for all rows
            "text": sentences,
            "notes": "",
        }
    )

    # Save to CSV
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"  → Created {output_file} with {len(sentences)} sentences")


def process_directory(root_dir):
    """
    Process the directory structure and tokenize all stories and recalls.

    Input structure:
        root_dir/
            story1/
                transcripts/
                    story.txt
                recalls/
                    sub01.txt
                    sub02.txt
                    ...
            story2/
                ...

    Output structure:
        flashfiction_processed/
            story1/
                transcript-story1.csv
                recalls/
                    sub-001-recall-story1.csv
                    sub-002-recall-story1.csv
                    ...
            story2/
                ...
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Directory '{root_dir}' does not exist!")
        return

    # Create output directory at the same level as input directory
    output_root = root_path.parent / "flashfiction_processed"

    # Remove output directory if it exists, then create fresh
    if output_root.exists():
        print(f"Removing existing output directory: {output_root}")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True)
    print(f"Created fresh output directory: {output_root}\n")

    # Find all story directories
    story_dirs = [d for d in root_path.iterdir() if d.is_dir()]

    if not story_dirs:
        print(f"No story directories found in {root_dir}")
        return

    print(f"Found {len(story_dirs)} story directories\n")
    print("=" * 60)

    for story_dir in sorted(story_dirs):
        story_name = story_dir.name
        print(f"\nProcessing: {story_name}")
        print("-" * 60)

        # Create story output directory
        story_output_dir = output_root / story_name
        story_output_dir.mkdir(parents=True, exist_ok=True)

        # Process transcripts/story.txt
        transcripts_dir = story_dir / "transcripts"
        if transcripts_dir.exists():
            story_file = transcripts_dir / "story.txt"
            if story_file.exists():
                # Output file: transcript-storyname.csv (directly in story folder)
                output_file = story_output_dir / f"transcript-{story_name}.csv"
                tokenize_story(story_file, output_file, story_name)
            else:
                print(f"  ⚠ No story.txt found in {transcripts_dir}")
        else:
            print(f"  ⚠ No transcripts/ directory found")

        # Process recalls/sub*.txt
        recalls_dir = story_dir / "recalls"
        if recalls_dir.exists():
            recall_files = sorted(recalls_dir.glob("sub*.txt"))

            if recall_files:
                # Create recalls output directory
                recalls_output_dir = story_output_dir / "recalls"
                recalls_output_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n  Found {len(recall_files)} recall files:")
                for recall_file in recall_files:
                    # Extract subject number from filename (e.g., sub01.txt -> 001)
                    subject_id = (
                        recall_file.stem.replace("sub", "").replace("-", "").zfill(3)
                    )

                    # Create output filename: sub-XXX-recall-storyname.csv
                    output_file = (
                        recalls_output_dir / f"sub-{subject_id}-recall-{story_name}.csv"
                    )
                    tokenize_recall(recall_file, output_file, story_name, subject_id)
            else:
                print(f"  ⚠ No sub*.txt files found in {recalls_dir}")
        else:
            print(f"  ⚠ No recalls/ directory found")

    print("\n" + "=" * 60)
    print(f"✓ Processing complete!")
    print(f"✓ Output saved to: {output_root}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python tokenize_stories.py <root_directory>")
        print("\nExample:")
        print("  python tokenize_stories.py ./stories")
        print("\nExpected INPUT directory structure:")
        print("  root_directory/")
        print("    story_name/")
        print("      transcripts/")
        print("        story.txt")
        print("      recalls/")
        print("        sub01.txt")
        print("        sub02.txt")
        print("        ...")
        print("\nOUTPUT directory structure:")
        print("  flashfiction_processed/  (created at same level as input)")
        print("    story_name/")
        print("      transcript-story_name.csv")
        print("      recalls/")
        print("        sub-001-recall-story_name.csv")
        print("        sub-002-recall-story_name.csv")
        print("        ...")
        sys.exit(1)

    root_directory = sys.argv[1]
    process_directory(root_directory)
