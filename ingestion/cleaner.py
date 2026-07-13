"""
===========================================================
Legal Text Cleaner

Purpose
-------
Clean extracted legal text while preserving legal citations.

Input
-----
pages.json

Output
------
clean_pages.json

Author : Shyam Suman
===========================================================
"""

import json
import logging
import re

from config import (
    PAGES_JSON,
    CLEAN_PAGES_JSON,
)

# =====================================================
# Logger
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


class LegalTextCleaner:

    """
    Performs rule-based cleaning of legal text.
    """

    def normalize_whitespace(self, text: str) -> str:
        """
        Remove unnecessary spaces.
        """

        text = re.sub(r"[ \t]+", " ", text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        text = re.sub(r"[ \t]+\n", "\n", text)

        return text.strip()

    def fix_hyphenation(self, text: str) -> str:
        """
        Join words broken across lines.

        Example

        busi-
        ness

        becomes

        business
        """

        return re.sub(r"(\w)-\s*\n\s*(\w)",r"\1\2",text,)

    def normalize_quotes(self, text: str) -> str:
        """
        Convert smart quotes into standard quotes.
        """

        text = text.replace("“", '"')
        text = text.replace("”", '"')
        text = text.replace("’", "'")
        text = text.replace("‘", "'")

        return text

    def remove_control_characters(self, text: str) -> str:
        """
        Remove invisible control characters.
        """

        return re.sub(
            r"[\x00-\x08\x0B\x0C\x0E-\x1F]",
            "",
            text,
        )

    def clean_text(self, text: str) -> str:
        """
        Complete cleaning pipeline.
        """

        text = self.fix_hyphenation(text)

        text = self.normalize_quotes(text)

        text = self.remove_control_characters(text)

        text = self.normalize_whitespace(text)

        return text


class CleanerPipeline:

    def __init__(self):

        self.cleaner = LegalTextCleaner()

        self.total_pages = 0

        self.cleaned_pages = []

        self.characters_before = 0

        self.characters_after = 0

    # -------------------------------------------------

    def load_pages(self):

        logger.info("Loading pages.json")

        with open(
            PAGES_JSON,
            "r",
            encoding="utf-8",
        ) as file:

            pages = json.load(file)

        self.total_pages = len(pages)

        logger.info(
            f"{self.total_pages} pages loaded."
        )

        return pages

    # -------------------------------------------------

    def clean_pages(self, pages):

        logger.info("Cleaning text...")

        for page in pages:

            #original_text = page["text"]

            original_text = page.get("text", "")

            if not isinstance(original_text, str):
                original_text = ""

            cleaned_text = self.cleaner.clean_text(
                original_text
            )

            self.characters_before += len(original_text)

            self.characters_after += len(cleaned_text)

            cleaned_page = page.copy()

            cleaned_page["text"] = cleaned_text

            self.cleaned_pages.append(cleaned_page)

    # -------------------------------------------------

    def save_pages(self):

        logger.info(
            "Saving clean_pages.json"
        )

        with open(
            CLEAN_PAGES_JSON,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.cleaned_pages,
                file,
                indent=4,
                ensure_ascii=False,
            )

    # -------------------------------------------------

    def print_summary(self):

        logger.info("=" * 60)

        logger.info("Cleaning Summary")

        logger.info("=" * 60)

        logger.info(
            f"Pages Cleaned : {self.total_pages}"
        )

        logger.info(
            f"Characters Before : {self.characters_before}"
        )

        logger.info(
            f"Characters After : {self.characters_after}"
        )

        logger.info(
            f"Output File : {CLEAN_PAGES_JSON}"
        )

        logger.info("=" * 60)

    # -------------------------------------------------

    def run(self):

        pages = self.load_pages()

        self.clean_pages(pages)

        self.save_pages()

        self.print_summary()


def main():

    pipeline = CleanerPipeline()

    pipeline.run()


if __name__ == "__main__":

    main()