"""
===========================================================
PDF Parser Module

Purpose
-------
Parse legal PDF documents while preserving page numbers
for future legal citations.

Workflow
--------
PDF
    ↓
Page-wise Text Extraction
    ↓
OCR (only if required)
    ↓
Page Metadata
    ↓
pages.json

Author : Shyam Suman
===========================================================
"""

import json
import uuid
import logging
from pathlib import Path
from typing import Dict, List

import fitz
import pytesseract

import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"D:\Tesseract-OCR\tesseract.exe"
)

from PIL import Image
from tqdm import tqdm

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PAGES_JSON,
    DOCUMENT_TYPES,
    OCR_TEXT_THRESHOLD,
)

from datetime import datetime

# =========================================================
# Configure Logger
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


class PDFParser:
    """
    Parses legal PDF documents page-by-page while preserving
    page numbers and document metadata.
    """

    def __init__(self):

        # ---------------- Statistics ---------------- #

        self.total_documents = 0
        self.total_pages = 0
        self.ocr_pages = 0
        self.failed_documents = 0

        # Stores parsed pages from every document
        self.pages_data: List[Dict] = []

        # Create processed directory if it doesn't exist
        PROCESSED_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =====================================================
    # OCR
    # =====================================================

    def perform_ocr(self, page: fitz.Page) -> str:
        """
        Perform OCR on a PDF page.

        Parameters
        ----------
        page : fitz.Page

        Returns
        -------
        str
            OCR extracted text.
        """

        try:

            pix = page.get_pixmap(dpi=300)

            image = Image.frombytes(
                "RGB",
                (pix.width, pix.height),
                pix.samples,
            )

            text = pytesseract.image_to_string(
                image,
                lang="eng",
            )

            self.ocr_pages += 1

            return text

        except Exception as e:

            logger.error(f"OCR failed : {e}")

            return ""

    # =====================================================
    # Extract Page Text
    # =====================================================

    def extract_page_text(self, page: fitz.Page) -> str:
        """
        Extract text from one PDF page.

        OCR is used only when the extracted text
        is below the configured threshold.
        """

        text = page.get_text("text") 

        if not text.strip():
            logger.info("Running OCR...")

            #logger.info(f"OCR → {pdf_path.name} Page {page_index+1}")

            text = self.perform_ocr(page)

        return text

    # =====================================================
    # Create Page Metadata
    # =====================================================

    def create_page_metadata(
        self,
        document_id: str,
        pdf_path: Path,
        document_type: str,
        page_number: int,
        page_id: str,
        processed_at: str,
        text: str,
    ) -> Dict:
        """
        Create metadata dictionary for a page.
        """

        return {

            "document_id": document_id,

            "document_name": pdf_path.stem,

            "document_type": document_type,

            "source_file": pdf_path.name,

            "relative_path": str(pdf_path),

            "page_number": page_number,

            "character_count": len(text),

            "word_count": len(text.split()),

            "page_id": page_id,

            "processed_at": processed_at,

            "text": text,

        }

    # =====================================================
    # Parse Single PDF
    # =====================================================

    def parse_single_pdf(
        self,
        pdf_path: Path,
        document_type: str,
    ) -> List[Dict]:
        """
        Parse one PDF document.

        Returns
        -------
        List[Dict]
            Page-wise extracted data.
        """

        logger.info(f"Parsing : {pdf_path.name}")

        pages = []

        try:

            with fitz.open(pdf_path) as document:

                document_id = str(uuid.uuid4())

                self.total_documents += 1

                for page_index, page in enumerate(document):

                    self.total_pages += 1

                    text = self.extract_page_text(page)

                    page_metadata = self.create_page_metadata(
                        document_id=document_id,
                        pdf_path=pdf_path,
                        document_type=document_type,
                        page_number=page_index + 1,
                        page_id = f"{document_id}_PAGE_{page_index+1:04d}",
                        processed_at=datetime.now().isoformat(),
                        text=text,
                    )

                    pages.append(page_metadata)

        except Exception as e:

            logger.error(
                f"Failed to process {pdf_path.name} : {e}"
            )

            self.failed_documents += 1

            return []

        return pages
    
    def parse_document_category(
        self,
        category_path: Path,
        document_type: str,
    ) -> None:
        """
        Parse every PDF inside one document category.

        Example
        -------
        data/raw/acts/
        data/raw/judgments/
        """

        if not category_path.exists():

            logger.warning(
                f"Directory not found : {category_path}"
            )

            return

        pdf_files = sorted(
            [
                file
                for file in category_path.iterdir()
                if file.is_file()
                and file.suffix.lower() == ".pdf"
            ]
        )

        logger.info(
            f"{document_type} : {len(pdf_files)} PDF(s) found."
        )

        for pdf_path in tqdm(
            pdf_files,
            desc=f"Processing {document_type}",
        ):

            pages = self.parse_single_pdf(
                pdf_path=pdf_path,
                document_type=document_type,
            )

            self.pages_data.extend(pages)

    # =====================================================
    # Parse Complete Dataset
    # =====================================================

    def parse_all_documents(self) -> None:
        """
        Parse every document category.
        """

        logger.info("=" * 60)
        logger.info("Starting PDF Parsing")
        logger.info("=" * 60)

        for document_type in DOCUMENT_TYPES:

            category_path = RAW_DATA_DIR / document_type

            self.parse_document_category(
                category_path=category_path,
                document_type=document_type,
            )

        logger.info("Parsing Complete.")

    # =====================================================
    # Save JSON
    # =====================================================

    def save_pages(self) -> None:
        """
        Save extracted pages to pages.json.
        """

        logger.info(
            f"Saving extracted pages to {PAGES_JSON}"
        )

        with open(
            PAGES_JSON,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.pages_data,
                file,
                indent=4,
                ensure_ascii=False,
            )

        logger.info(
            "pages.json created successfully."
        )

    # =====================================================
    # Summary
    # =====================================================

    def print_summary(self) -> None:
        """
        Print parser statistics.
        """

        logger.info("=" * 60)

        logger.info("PDF Parsing Summary")

        logger.info("=" * 60)

        logger.info(
            f"Documents Processed : {self.total_documents}"
        )

        logger.info(
            f"Pages Processed     : {self.total_pages}"
        )

        logger.info(
            f"OCR Pages           : {self.ocr_pages}"
        )

        logger.info(
            f"Failed Documents    : {self.failed_documents}"
        )

        logger.info(
            f"Total Extracted Pages : {len(self.pages_data)}"
        )

        logger.info("=" * 60)

def main():

    parser = PDFParser()

    parser.parse_all_documents()

    parser.save_pages()

    parser.print_summary()


if __name__ == "__main__":

    main()