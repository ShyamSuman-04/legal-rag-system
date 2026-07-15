"""
generate_llm_answers.py

Runs the complete RAG pipeline on every question in the Golden Set
and generates a CSV containing:

Question
Ground Truth Answer
LLM Generated Answer

Output:
evaluation/llm_gen_csv.csv

Author: Shyam Suman
"""

import sys
import time
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# Project Root
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


# ------------------------------------------------------------
# Import RAG Pipeline
# ------------------------------------------------------------
from rag.rag_pipeline import RAGPipeline


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
GOLDEN_SET_PATH = PROJECT_ROOT / "data" / "golden_set" / "golden_set.csv"

OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "llm_gen_csv.csv"


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
TOTAL_QUESTIONS = 199

START_INDEX = 148      # Question 29
END_INDEX = 190

DELAY_SECONDS = 60


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():

    print("=" * 80)
    print("US TAX & LEGAL RAG SYSTEM")
    print("Golden Set LLM Answer Generation")
    print("=" * 80)

    if not GOLDEN_SET_PATH.exists():
        raise FileNotFoundError(
            f"Golden Set not found:\n{GOLDEN_SET_PATH}"
        )

    # --------------------------------------------------------
    # Read Golden Set
    # --------------------------------------------------------

    golden_df = pd.read_csv(GOLDEN_SET_PATH)

    golden_df = golden_df.head(TOTAL_QUESTIONS)

    # --------------------------------------------------------
    # Resume Support
    # --------------------------------------------------------

    if OUTPUT_PATH.exists():

        output_df = pd.read_csv(OUTPUT_PATH)

        completed = len(output_df)

        print(f"\nExisting output found.")
        print(f"Completed rows : {completed}")

    else:

        output_df = pd.DataFrame(
            columns=[
                "Question",
                "Ground Truth Answer",
                "LLM Generated Answer",
            ]
        )

        completed = 0

    # --------------------------------------------------------
    # Load Pipeline ONLY ONCE
    # --------------------------------------------------------

    print("\nLoading RAG Pipeline...\n")

    pipeline = RAGPipeline(debug=False)

    print("\nPipeline Loaded Successfully.\n")

    # --------------------------------------------------------
    # Loop
    # --------------------------------------------------------

    start = max(completed, START_INDEX)
    for index in range(start, END_INDEX):

        row = golden_df.iloc[index]

        question = str(row["Question"]).strip()

        ground_truth = str(row["Ground Truth Answer"]).strip()

        print("=" * 80)
        print(f"[{index+1}/{len(golden_df)}]")
        print(question)
        print("=" * 80)

        try:

            response = pipeline.answer_question(question)

            llm_answer = response["answer"]

        except Exception as e:

            print(f"ERROR : {e}")

            llm_answer = f"ERROR : {e}"

        # ----------------------------------------------------
        # Save immediately
        # ----------------------------------------------------

        output_df.loc[len(output_df)] = [
            question,
            ground_truth,
            llm_answer,
        ]

        output_df.to_csv(
            OUTPUT_PATH,
            index=False,
        )

        print("\nSaved.")

        # ----------------------------------------------------
        # Wait
        # ----------------------------------------------------

        if index != END_INDEX - 1:

            print(f"\nSleeping for {DELAY_SECONDS} seconds...\n")

            time.sleep(DELAY_SECONDS)

    print("\n")
    print("=" * 80)
    print("COMPLETED")
    print("=" * 80)
    print(f"Saved to:\n{OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()