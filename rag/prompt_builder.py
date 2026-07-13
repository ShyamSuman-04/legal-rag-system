"""
prompt_builder.py

Builds the final prompt that is sent to the LLM.

Responsibilities
----------------
1. Load the system prompt.
2. Format retrieved legal documents into context.
3. Respect maximum context size.
4. Build the final prompt for the LLM.

Author: Shyam Suman
Project: US Tax & Legal RAG System
"""

import logging
from typing import Dict, List

from config import SYSTEM_PROMPT_FILE, MAX_CONTEXT_CHARACTERS

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Document Type Mapping
# ---------------------------------------------------------------------

DOCUMENT_TYPE_DISPLAY = {
    "acts": "Acts",
    "judgments": "Court Judgment",
    "tax": "Tax Document",
    "pov": "Point-of-View Document",
}


# ---------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------

class PromptBuilder:
    """
    Builds the final prompt sent to the LLM.

    Pipeline
    --------
        System Prompt
                +
        User Question
                +
        Retrieved Context
                =
          Final LLM Prompt
    """

    def __init__(self):
        self.system_prompt = self.load_system_prompt()

        logger.info("=" * 65)
        logger.info("Prompt Builder Initialized")
        logger.info(
            "Max Context Characters : %d",
            MAX_CONTEXT_CHARACTERS,
        )
        logger.info("=" * 65)

    # -----------------------------------------------------------------
    # System Prompt
    # -----------------------------------------------------------------

    def load_system_prompt(self) -> str:
        """
        Loads the system prompt from system_prompt.txt.

        Returns
        -------
        str
            System prompt.
        """
        if not SYSTEM_PROMPT_FILE.exists():
            raise FileNotFoundError(
                f"System prompt not found:\n{SYSTEM_PROMPT_FILE}"
            )

        logger.info(
            "Loading system prompt from\n%s",
            SYSTEM_PROMPT_FILE,
        )

        with open(
            SYSTEM_PROMPT_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            prompt = file.read().strip()

        logger.info("System prompt loaded successfully.")
        return prompt

    # -----------------------------------------------------------------
    # Format One Retrieved Document
    # -----------------------------------------------------------------

    def format_document(self, chunk: Dict) -> str:
        """
        Converts one retrieved chunk into a readable
        document block for the LLM.

        Parameters
        ----------
        chunk : Dict

        Returns
        -------
        str
        """
        document_name = chunk.get(
            "document_name",
            "Unknown Document",
        )

        # Normalize document type for display
        raw_type = chunk.get("document_type", "").lower()
        document_type = DOCUMENT_TYPE_DISPLAY.get(
            raw_type,
            chunk.get("document_type", "Unknown"),
        )

        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")

        # Handle single-page documents
        if page_start is not None and page_end is not None:
            if page_start == page_end:
                pages = str(page_start)
            else:
                pages = f"{page_start}-{page_end}"
        else:
            pages = "-"

        text = chunk.get("text", "").strip()

        block = (
            "\n"
            + "=" * 70
            + "\n\n"
            + f"Document : {document_name}\n\n"
            + f"Type : {document_type}\n\n"
            + f"Pages : {pages}\n\n"
            + "Text:\n\n"
            + text
            + "\n\n"
            + "=" * 70
            + "\n"
        )
        return block

    # -----------------------------------------------------------------
    # Build Context
    # -----------------------------------------------------------------

    def build_context(self, chunks: List[Dict]) -> str:
        """
        Builds the context section from reranked chunks.

        Context size is limited using
        MAX_CONTEXT_CHARACTERS.

        Parameters
        ----------
        chunks : List[Dict]

        Returns
        -------
        str
        """
        logger.info("=" * 65)
        logger.info("Building Context")
        logger.info(
            "Retrieved Chunks : %d",
            len(chunks),
        )

        context_blocks = []
        current_size = 0
        included_chunks = 0

        seen_chunk_ids = set()
        duplicate_chunks = 0

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id in seen_chunk_ids:
                duplicate_chunks += 1
                continue

            seen_chunk_ids.add(chunk_id)

            block = self.format_document(chunk)
            block_size = len(block)

            if current_size + block_size > MAX_CONTEXT_CHARACTERS:
                logger.info(
                    "Maximum context size reached."
                )
                break

            context_blocks.append(block)
            current_size += block_size
            included_chunks += 1

        context = "".join(context_blocks)
    
        logger.info(
            "Chunks Included : %d",
            included_chunks,
        )
        logger.info(
            "Duplicate Chunks : %d",
            duplicate_chunks,
        )
        logger.info(
            "Context Characters : %d",
            len(context),
        )
        logger.info("=" * 65)

        return context

    # -----------------------------------------------------------------
    # Build Final Prompt
    # -----------------------------------------------------------------

    def build_prompt(
        self,
        question: str,
        chunks: List[Dict],
    ) -> Dict:
        """
        Builds the complete prompt that will be sent to the LLM.

        Parameters
        ----------
        question : str
            User question.
        chunks : List[Dict]
            Reranked retrieval results.

        Returns
        -------
        Dict
            {
                "question": str,
                "prompt": str,
                "context": str,
                "context_documents": List[Dict]
            }
        """
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        logger.info("=" * 65)
        logger.info("Building Final Prompt")
        logger.info("Question : %s", question)

        context = self.build_context(chunks)

        prompt = (
            f"{self.system_prompt}\n\n"
            f"{'=' * 70}\n\n"
            "QUESTION\n\n"
            f"{question}\n\n"
            f"{'=' * 70}\n\n"
            "CONTEXT\n\n"
            f"{context}\n"
            f"{'=' * 70}\n\n"
            "Answer:\n\n"
            "Use only the supplied context. End your response with a 'References' section.\n\n"
        )

        logger.info(
            "Prompt Length : %d characters",
            len(prompt),
        )
        logger.info("=" * 65)



        return {
            "question": question,
            "prompt": prompt,
            "context": context,
            "context_documents": self.extract_context_documents(chunks),
        }

    # -----------------------------------------------------------------
    # Context Document Metadata
    # -----------------------------------------------------------------

    def extract_context_documents(self, chunks: List[Dict]) -> List[Dict]:
        """
        Extracts lightweight citation information.
        These metadata are NOT sent to the LLM.
        They are returned so that the UI,
        evaluation pipeline, or logging layer
        can easily display which documents
        were supplied as context.
        """
        documents = []
        for chunk in chunks:
            documents.append(
                {
                    "document_name": chunk.get(
                        "document_name",
                        "",
                    ),
                    "document_type": chunk.get(
                        "document_type",
                        "",
                    ),
                    "page_start": chunk.get(
                        "page_start",
                    ),
                    "page_end": chunk.get(
                        "page_end",
                    ),
                }
            )
        return documents

    # -----------------------------------------------------------------
    # Prompt Preview
    # -----------------------------------------------------------------

        # -----------------------------------------------------------------
    # Prompt Preview
    # -----------------------------------------------------------------

    def preview_prompt(
        self,
        prompt_data: Dict,
        show_system_prompt: bool = True,
    ) -> None:
        """
        Displays the prompt that will be sent to the LLM.

        By default, only the Question, Context, and Answer sections
        are displayed since those are the parts that usually need
        debugging.

        Set show_system_prompt=True if you also want to inspect the
        loaded system prompt.
        """

        print("\n")
        print("=" * 90)
        print("LLM INPUT PREVIEW")
        print("=" * 90)

        # ---------------------------------------------------------
        # Optional System Prompt
        # ---------------------------------------------------------

        if show_system_prompt:

            print("\nSYSTEM PROMPT\n")
            print("-" * 90)
            print(self.system_prompt)
            print("-" * 90)

        # ---------------------------------------------------------
        # Question
        # ---------------------------------------------------------

        print("\nQUESTION\n")
        print("-" * 90)
        print(prompt_data["question"])

        # ---------------------------------------------------------
        # Context
        # ---------------------------------------------------------

        print("\nCONTEXT\n")
        print("-" * 90)

        print(prompt_data["context"])

        # ---------------------------------------------------------
        # Answer
        # ---------------------------------------------------------

        print("-" * 90)
        print("\nANSWER\n")
        print("Use only the supplied context.")
        print("End your response with a 'References' section.")

        # ---------------------------------------------------------
        # Statistics
        # ---------------------------------------------------------

        print("\n")
        print("=" * 90)
        print("PROMPT STATISTICS")
        print("=" * 90)

        print(f"Prompt Characters : {len(prompt_data['prompt']):,}")
        print(f"Context Characters: {len(prompt_data['context']):,}")
        print(
            f"Documents Used    : "
            f"{len(prompt_data['context_documents'])}"
        )

        print("=" * 90)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """
    Local test for PromptBuilder.

    Workflow
    --------
        Question
            ↓
        Reranker
            ↓
        Prompt Builder
            ↓
        Prompt Preview
    """
    try:
        print("\n")
        print("=" * 80)
        print("US TAX & LEGAL RAG SYSTEM")
        print("Prompt Builder")
        print("=" * 80)
        print("Type 'exit' or 'quit' to stop.\n")

        # Import here to avoid circular dependency if needed
        from retrieval.reranker import Reranker

        reranker = Reranker()
        prompt_builder = PromptBuilder()

        while True:
            question = input(
                "Enter question: "
            ).strip()
            if question.lower() in {
                "exit",
                "quit",
            }:
                print("\nExiting Prompt Builder...\n")
                break
            if not question:
                print(
                    "\nPlease enter a valid question.\n"
                )
                continue

            print("\nRetrieving relevant documents...\n")
            reranked_results = reranker.search(
                query=question
            )
            if not reranked_results:
                print(
                    "\nNo relevant documents found.\n"
                )
                continue

            prompt_data = prompt_builder.build_prompt(
                question=question,
                chunks=reranked_results,
            )

            prompt_builder.preview_prompt(prompt_data)


            print("\n")
            print("=" * 80)
            print("CONTEXT DOCUMENTS")
            print("=" * 80)
            for index, document in enumerate(
                prompt_data["context_documents"],
                start=1,
            ):
                print(
                    f"{index}. "
                    f"{document['document_name']} | "
                    f"{document['document_type']} | "
                    f"Pages "
                    f"{document['page_start']}-"
                    f"{document['page_end']}"
                )
            print("=" * 80)
            print(
                f"\nPrompt Length : "
                f"{len(prompt_data['prompt'])} characters\n"
            )

    except KeyboardInterrupt:
        print("\nInterrupted by user.\n")
    except Exception as error:
        logger.exception(error)
        print(
            "\nFailed to build prompt.\n"
        )


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()