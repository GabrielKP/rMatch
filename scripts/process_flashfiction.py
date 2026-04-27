#!/usr/bin/env python3
"""
Tokenize story transcripts and participant recalls using NLTK sentence tokenizer.
Outputs CSV files with sentence segmentation.
"""

from pathlib import Path

import nltk
import pandas as pd

# Download the punkt tokenizer if not already available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download("punkt")


def tokenize_story(input_file, output_file):
    """
    Tokenize a story file and save as CSV with columns: segment, text

    Args:
        input_file: Path to the story.txt file
        output_file: Path to save story_nltk.csv
    """
    print(f"Tokenizing story: {input_file}")

    # Read the story text
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Tokenize into sentences
    sentences = nltk.sent_tokenize(text)

    # Create DataFrame
    df = pd.DataFrame({"segment": range(1, len(sentences) + 1), "text": sentences})

    # Save to CSV
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"  → Created {output_file} with {len(sentences)} sentences")


def tokenize_recall(input_file, output_file):
    """
    Tokenize a recall file and save as CSV with columns: segment, events, text

    Args:
        input_file: Path to the recall .txt file (e.g., sub01.txt)
        output_file: Path to save the tokenized CSV (e.g., sub01.csv)
    """
    print(f"Tokenizing recall: {input_file}")

    # Read the recall text
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Tokenize into sentences
    sentences = nltk.sent_tokenize(text)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "segment": range(1, len(sentences) + 1),
            "events": "",  # Empty string for all rows
            "text": sentences,
        }
    )

    # Save to CSV
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"  → Created {output_file} with {len(sentences)} sentences")


def process_directory(root_dir):
    """
    Process the directory structure and tokenize all stories and recalls.

    Expected structure:
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
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Directory '{root_dir}' does not exist!")
        return

    # Find all story directories
    story_dirs = [d for d in root_path.iterdir() if d.is_dir()]

    if not story_dirs:
        print(f"No story directories found in {root_dir}")
        return

    print(f"\nFound {len(story_dirs)} story directories\n")
    print("=" * 60)

    for story_dir in sorted(story_dirs):
        print(f"\nProcessing: {story_dir.name}")
        print("-" * 60)

        # Process transcripts/story.txt
        transcripts_dir = story_dir / "transcripts"
        if transcripts_dir.exists():
            story_file = transcripts_dir / "story.txt"
            if story_file.exists():
                output_file = transcripts_dir / "story_nltk.csv"
                tokenize_story(story_file, output_file)
            else:
                print(f"  ⚠ No story.txt found in {transcripts_dir}")
        else:
            print(f"  ⚠ No transcripts/ directory found")

        # Process recalls/sub*.txt
        recalls_dir = story_dir / "recalls"
        if recalls_dir.exists():
            recall_files = sorted(recalls_dir.glob("sub*.txt"))

            if recall_files:
                print(f"\n  Found {len(recall_files)} recall files:")
                for recall_file in recall_files:
                    # Create output filename: sub01.txt -> sub01.csv
                    output_file = recalls_dir / f"{recall_file.stem}.csv"
                    tokenize_recall(recall_file, output_file)
            else:
                print(f"  ⚠ No sub*.txt files found in {recalls_dir}")
        else:
            print(f"  ⚠ No recalls/ directory found")

    print("\n" + "=" * 60)
    print("✓ Processing complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python tokenize_stories.py <root_directory>")
        print("\nExample:")
        print("  python tokenize_stories.py ./stories")
        print("\nExpected directory structure:")
        print("  root_directory/")
        print("    story_name/")
        print("      transcripts/")
        print("        story.txt")
        print("      recalls/")
        print("        sub01.txt")
        print("        sub02.txt")
        print("        ...")
        sys.exit(1)

    root_directory = sys.argv[1]
    process_directory(root_directory)
