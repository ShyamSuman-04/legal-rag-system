"""
judge_answers.py

Evaluates LLM generated answers against Ground Truth answers.

Input:
evaluation/llm_gen_csv.csv

Output:
evaluation/evaluation_csv.csv

Adds:
- Document Retrieval Score
- Faithfulness Score

Author : Shyam Suman
"""

import os
import re
import time
from pathlib import Path

import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_ROOT / "evaluation" / "llm_gen_csv.csv"

OUTPUT_CSV = PROJECT_ROOT / "evaluation" / "evaluation_csv.csv"

ENV_PATH = PROJECT_ROOT / ".env"

# ----------------------------------------------------------
# Settings
# ----------------------------------------------------------

DELAY_SECONDS = 6

# ----------------------------------------------------------
# Load API Key
# ----------------------------------------------------------

load_dotenv(ENV_PATH)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "openai/gpt-oss-20b"

# ----------------------------------------------------------
# System Prompt
# ----------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG)
system developed for the US Tax & Legal domain.

You will receive TWO ANSWERS.

==============================================================
GROUND TRUTH ANSWER
==============================================================

...

==============================================================
LLM GENERATED ANSWER
==============================================================

...

Evaluate ONLY the following TWO criteria.

Criterion 1

Do ALL document names and page numbers referenced in the
LLM Generated Answer correctly match the document names and
page numbers present in the Ground Truth Answer?

Answer:
YES
or
NO

Criterion 2

Is the LLM Generated Answer factually consistent with
the Ground Truth Answer?

Ignore wording differences.

Treat paraphrasing as correct.

Only answer NO if the generated answer:

• contradicts facts
• invents legal information
• changes legal meaning
• omits critical legal meaning

Ignore:

• formatting
• grammar
• writing style
• sentence order

Rules

Never explain.

Never justify.

Never output markdown.

Never output JSON.

Never output punctuation.

Never output anything except exactly TWO WORDS.

Valid outputs are ONLY:

YES YES

YES NO

NO YES

NO NO
"""

# ----------------------------------------------------------
# Parser
# ----------------------------------------------------------


def parse_response(text: str):
    """
    Converts any Groq response into:

    YES YES
    YES NO
    NO YES
    NO NO
    """

    text = text.upper()

    text = re.sub(r"[^A-Z]", " ", text)

    words = text.split()

    words = [w for w in words if w in ("YES", "NO")]

    if len(words) >= 2:
        return words[0], words[1]

    if len(words) == 1:
        return words[0], ""

    return "", ""


# ----------------------------------------------------------
# Judge
# ----------------------------------------------------------


def evaluate(ground_truth, generated):

    user_prompt = f"""
==============================================================
GROUND TRUTH ANSWER
==============================================================

{ground_truth}


==============================================================
LLM GENERATED ANSWER
==============================================================

{generated}
"""

    completion = client.chat.completions.create(

        model=MODEL,

        temperature=0,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    reply = completion.choices[0].message.content.strip()

    return parse_response(reply)


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------


def main():

    print("=" * 80)
    print("LLM JUDGE")
    print("=" * 80)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    input_df = pd.read_csv(INPUT_CSV)

    # ------------------------------------------------------

    if OUTPUT_CSV.exists():

        output_df = pd.read_csv(OUTPUT_CSV)

        completed = len(output_df)

        print(f"Resuming from row {completed + 1}")

    else:

        output_df = pd.DataFrame(columns=[
            "Question",
            "Ground Truth Answer",
            "LLM Generated Answer",
            "Document Retrieval Score",
            "Faithfulness Score",
        ])

        completed = 0

    # ------------------------------------------------------

    total = len(input_df)

    for index in range(completed, total):

        row = input_df.iloc[index]

        question = row["Question"]

        gt = row["Ground Truth Answer"]

        llm = row["LLM Generated Answer"]

        print("=" * 80)
        print(f"[{index+1}/{total}]")
        print(question)
        print("=" * 80)

        try:

            retrieval_score, faithfulness_score = evaluate(
                gt,
                llm,
            )

        except Exception as e:

            print(e)

            retrieval_score = ""

            faithfulness_score = ""

        output_df.loc[len(output_df)] = [

            question,

            gt,

            llm,

            retrieval_score,

            faithfulness_score,
        ]

        output_df.to_csv(

            OUTPUT_CSV,

            index=False,

        )

        print(

            f"Retrieval : {retrieval_score} | "
            f"Faithfulness : {faithfulness_score}"

        )

        print("Saved.")

        if index != total - 1:

            print(f"Sleeping {DELAY_SECONDS} seconds...\n")

            time.sleep(DELAY_SECONDS)

    print("\nDone.")

    print(f"\nSaved to:\n{OUTPUT_CSV}")


# ----------------------------------------------------------

if __name__ == "__main__":

    main()