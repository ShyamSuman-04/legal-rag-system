"""
rag_pipeline.py

End-to-end Retrieval-Augmented Generation (RAG) pipeline for the
US Tax & Legal RAG System.

This module is a thin orchestrator only - it does not reimplement
retrieval, reranking, or prompt construction. Each stage below is
delegated to the module that already owns it:

    Reranker       -> retrieval/reranker.py   (hybrid search + rerank)
    PromptBuilder  -> rag/prompt_builder.py   (context + prompt text)
    GroqClient     -> rag/groq_client.py      (LLM call)

Pipeline (what this file calls directly)
-----------------------------------------
User Question
        |
        v
Reranker.search()            <- hybrid search + cross-encoder rerank
        |
        v
PromptBuilder.build_prompt() <- formats reranked chunks into context
        |
        v
GroqClient.generate()        <- sends prompt_data["prompt"] to the LLM
        |
        v
Final Answer + References

Full internal retrieval chain (for reference - not called directly
by this file; all of this lives inside Reranker.search()):

    Question -> Hybrid Search (Keyword + Vector, RRF fusion)
             -> CrossEncoder Reranker -> reranked chunks

Author : Shyam Suman
Project : US Tax & Legal RAG System
"""

import logging
from typing import Dict, List

from retrieval.reranker import Reranker
from rag.prompt_builder import PromptBuilder
from rag.groq_client import GroqClient


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------

class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline.

    Responsibilities
    ----------------

    1. Retrieve relevant legal chunks.
    2. Rerank retrieved chunks.
    3. Build the final LLM prompt.
    4. Generate answer using Groq.
    5. Return answer together with retrieval metadata.
    """

    def __init__(self, debug: bool = False):
        """
        Parameters
        ----------
        debug : bool, default=False
            If True, the `prompt` and `context` fields are included
            in the response (useful for development). If False, they
            are omitted to keep the response lightweight.
        """
        self.debug = debug

        logger.info("=" * 65)
        logger.info("Initializing RAG Pipeline (debug=%s)", debug)
        logger.info("=" * 65)

        logger.info("Loading Reranker...")
        self.reranker = Reranker()

        logger.info("Loading Prompt Builder...")
        self.prompt_builder = PromptBuilder()

        logger.info("Loading Groq Client...")
        self.llm = GroqClient()

        logger.info("RAG Pipeline initialized successfully.")
        logger.info("=" * 65)

    # -----------------------------------------------------------------
    # Answer Question
    # -----------------------------------------------------------------

    def answer_question(
        self,
        question: str,
    ) -> Dict:
        """
        Executes the complete RAG pipeline.

        Parameters
        ----------
        question : str

        Returns
        -------
        Dict
            Always contains:
                "question"         : str
                "answer"           : str
                "references"       : List[Dict]  (deduplicated by doc+page)
                "model"            : str
                "prompt_tokens"    : int
                "completion_tokens": int
                "total_tokens"     : int
                "latency_seconds"  : float

            If debug=True (set during __init__), the following are also
            included:
                "prompt"           : str
                "context"          : str

        Note: this schema is identical whether or not any documents
        were retrieved, so print_response() (and any future caller,
        e.g. a FastAPI response model) never has to special-case a
        missing key depending on which path was taken.
        """
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        logger.info("=" * 65)
        logger.info("Running Complete RAG Pipeline")
        logger.info("Question : %s", question)
        logger.info("=" * 65)

        # -------------------------------------------------------------
        # Retrieval + Reranking
        # -------------------------------------------------------------
        logger.info("Retrieving relevant documents...")
        reranked_chunks = self.reranker.search(query=question)

        if not reranked_chunks:
            logger.warning("No relevant documents retrieved.")
            return self._build_error_response(
                question,
                "No relevant documents were found for the given question."
            )

        logger.info("Retrieved %d reranked chunks.", len(reranked_chunks))

        # -------------------------------------------------------------
        # Prompt Construction
        # -------------------------------------------------------------
        logger.info("Building final prompt...")
        prompt_data = self.prompt_builder.build_prompt(
            question=question,
            chunks=reranked_chunks,
        )
        logger.info("Prompt built successfully.")
        logger.info("Prompt Length : %d characters", len(prompt_data["prompt"]))

        # -------------------------------------------------------------
        # LLM Generation
        # -------------------------------------------------------------
        logger.info("Generating answer using Groq...")
        try:
            llm_response = self.llm.generate(prompt_data["prompt"])
        except Exception as e:
            logger.exception("LLM generation failed.")
            return self._build_error_response(
                question,
                f"LLM generation failed: {str(e)}"
            )

        # Safeguard against empty answer
        answer = llm_response["answer"].strip()
        if not answer:
            logger.warning("LLM returned an empty answer.")
            answer = "The language model returned an empty response."

        logger.info("LLM generation completed.")

        # -------------------------------------------------------------
        # Deduplicate references by (document_name, page_start, page_end)
        # -------------------------------------------------------------
        unique_refs = self._deduplicate_references(
            prompt_data["context_documents"]
        )

        # -------------------------------------------------------------
        # Build Final Response
        # -------------------------------------------------------------
        response = {
            "question": question,
            "answer": answer,  # <-- using the sanitized answer variable
            "references": unique_refs,
            "model": llm_response["model"],
            "prompt_tokens": llm_response["prompt_tokens"],
            "completion_tokens": llm_response["completion_tokens"],
            "total_tokens": llm_response["total_tokens"],
            "latency_seconds": llm_response["latency_seconds"],
        }

        # Include prompt & context only in debug mode
        if self.debug:
            response["prompt"] = prompt_data["prompt"]
            response["context"] = prompt_data["context"]

        logger.info("=" * 65)
        logger.info("RAG Pipeline completed successfully.")
        logger.info("=" * 65)

        return response

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _deduplicate_references(
        self,
        references: List[Dict]
    ) -> List[Dict]:
        """
        Remove duplicate references based on document name and page range.
        """
        seen = set()
        unique = []
        for ref in references:
            key = (
                ref.get("document_name"),
                ref.get("page_start"),
                ref.get("page_end"),
            )
            if key not in seen:
                seen.add(key)
                unique.append(ref)
        return unique

    def _build_error_response(
        self,
        question: str,
        error_message: str
    ) -> Dict:
        """
        Return a consistent error response with all expected keys.
        """
        response = {
            "question": question,
            "answer": error_message,
            "references": [],
            "model": self.llm.model if hasattr(self, "llm") else "unknown",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
        }
        if self.debug:
            response["prompt"] = ""
            response["context"] = ""
        return response

    # -----------------------------------------------------------------
    # Pretty Printing
    # -----------------------------------------------------------------

    def print_response(
        self,
        response: Dict,
    ) -> None:
        """
        Prints a formatted RAG response.
        """
        print("\n")
        print("=" * 90)
        print("QUESTION")
        print("=" * 90)
        print(response["question"])
        print("\n")

        print("=" * 90)
        print("ANSWER")
        print("=" * 90)
        print(response["answer"])
        print("\n")

        print("=" * 90)
        print("REFERENCES")
        print("=" * 90)

        references = response["references"]
        if not references:
            print("No references available.")
        else:
            for index, ref in enumerate(references, start=1):
                page_start = ref.get("page_start")
                page_end = ref.get("page_end")
                if page_start is not None and page_end is not None:
                    pages = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
                else:
                    pages = "-"
                print(
                    f"{index}. "
                    f"{ref['document_name']} | "
                    f"{ref['document_type']} | "
                    f"Pages {pages}"
                )

        print("\n")
        print("=" * 90)
        print("STATISTICS")
        print("=" * 90)
        print(f"Model               : {response['model']}")
        print(f"Prompt Tokens       : {response['prompt_tokens']}")
        print(f"Completion Tokens   : {response['completion_tokens']}")
        print(f"Total Tokens        : {response['total_tokens']}")
        print(f"Latency             : {response['latency_seconds']} sec")
        print(f"Retrieved Documents : {len(response['references'])}")
        print("=" * 90)

    # -----------------------------------------------------------------
    # Interactive CLI
    # -----------------------------------------------------------------

    def interactive_chat(self):
        """
        Interactive command-line interface for the complete
        Retrieval-Augmented Generation pipeline.
        """
        print("\n")
        print("=" * 80)
        print("US TAX & LEGAL RAG SYSTEM")
        print("Retrieval-Augmented Generation (RAG)")
        print("=" * 80)
        print("Type 'exit' or 'quit' to stop.\n")

        while True:
            try:
                question = input("Enter Question: ").strip()
                if question.lower() in {"exit", "quit"}:
                    print("\nExiting RAG Pipeline...\n")
                    break
                if not question:
                    print("\nPlease enter a valid question.\n")
                    continue

                response = self.answer_question(question)
                self.print_response(response)

            except KeyboardInterrupt:
                print("\nInterrupted by user.\n")
                break
            except Exception as error:
                logger.exception(error)
                print(f"\nError: {error}\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """
    Entry point for the complete RAG pipeline.
    """
    try:
        # Set debug=True here if you want to see prompt/context during development
        rag_pipeline = RAGPipeline(debug=False)
        rag_pipeline.interactive_chat()
    except KeyboardInterrupt:
        print("\nInterrupted by user.\n")
    except Exception as error:
        logger.exception(error)
        print("\nFailed to start RAG Pipeline.\n")


if __name__ == "__main__":
    main()