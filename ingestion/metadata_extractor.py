"""
===========================================================
Legal Metadata Extractor

Purpose
-------
Enrich each chunk in chunks.json with structured, queryable
metadata extracted via regex (no AI / no LLM calls).

Design
------
Every chunk ends up with the SAME metadata schema
(unified schema), but only the keys that actually have
values are included (no "judges": null clutter on an Act
chunk, no "sections": [] clutter on a Judgment chunk).

Named entities (acts, cases, courts, judges, author, named
code references) are grouped under a nested "entities" key,
since they map naturally to Neo4j nodes. Structured legal
citations (usc_citations, cfr_citations, sections, articles,
chapters, titles, tax fields) stay flat, since they behave
more like citation-edges/properties than standalone nodes.

Keyword extraction uses corpus-level TF-IDF (pure Python,
no sklearn/spaCy dependency) computed once across all chunks,
so boilerplate terms that repeat in nearly every legal chunk
("federal", "secretary", "united") get down-weighted in favor
of terms distinctive to that specific chunk.

    chunks.json
         |
         v
    build_corpus_idf_scores()   <- one pass over all chunks
         |
         v
    extract_common_metadata()   <- dates, usc/cfr citations,
         |                          entities.acts, keywords
         v
    dispatch on document_type
         |
    -----------------------------------------------
    | acts | judgments | tax | pov |
    -----------------------------------------------
         |
         v
    merge (deep-merge "entities") + drop empty fields
         |
         v
    chunks_with_metadata.json

Deliberately out of scope for this module (by design, not
oversight):
  - Cross-entity relationship extraction (e.g. "Section 162"
    -> BELONGS_TO -> "Internal Revenue Code", or a judgment
    CITES a statute). That's graph-edge construction, not
    chunk metadata, and belongs in a separate
    relationship_extractor.py once this module is stable.
  - spaCy-based NER (ORG/PERSON/DATE) for judgments. Valuable
    later, but it's a real model dependency that isn't needed
    to hit the "must have" / "legal metadata" tiers.

Input
-----
chunks.json

Output
------
chunks_with_metadata.json

Author : Shyam Suman
===========================================================
"""

import json
import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional

from config import (CHUNKS_JSON, PROCESSED_DATA_DIR, CHUNKS_WITH_METADATA_JSON)

# -----------------------------------------------------
# NOTE: Add this line to config.py for consistency with
# the rest of the pipeline. Falling back to a local
# definition here so this module works standalone:
#
#   CHUNKS_WITH_METADATA_JSON = (
#       PROCESSED_DATA_DIR / "chunks_with_metadata.json"
#   )
# -----------------------------------------------------

#CHUNKS_WITH_METADATA_JSON = (PROCESSED_DATA_DIR / "chunks_with_metadata.json")

# =====================================================
# Logger
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# =====================================================
# Stopwords for lightweight keyword extraction
# (No spaCy / NLTK dependency needed)
# =====================================================

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "shall", "such",
    "under", "section", "act", "any", "may", "not", "are", "was",
    "were", "has", "have", "had", "from", "which", "these", "those",
    "each", "than", "then", "into", "upon", "other", "including",
    "including", "amended", "amendment", "shall", "will", "would",
    "there", "their", "its", "his", "her", "who", "whom", "been",
    "being", "also", "more", "most", "some", "all", "out", "over",
    "after", "before", "between", "during", "following", "pursuant",
    "provided", "made", "make", "makes", "date", "dated", "page",
    "public", "law", "united", "states", "congress",
}


class MetadataExtractor:
    """
    Collection of regex-based extraction methods for legal text.
    Each method is independently testable and returns either a
    list, a string, or None.
    """

    # -------------------------------------------------
    # Common / cross-document-type extractors
    # -------------------------------------------------

    def extract_usc_citations(self, text: str) -> List[str]:
        """
        Extract US Code citations.

        Examples matched:
            26 U.S.C. §162
            21 U.S.C. 379j-12(a)(1)(A)(ii)
        """

        pattern = (
            r"\d{1,3}\s+U\.S\.C\.?\s*"
            r"§{0,2}\s*"
            r"\d+[A-Za-z]*"
            r"(?:\s*[\u2013\-]\s*\d+)?"
            r"(?:\([A-Za-z0-9]+\))*"
        )

        matches = re.findall(pattern, text)

        return self._clean_and_dedupe(matches)

    def extract_cfr_citations(self, text: str) -> List[str]:
        """
        Extract Code of Federal Regulations citations.

        Example matched:
            26 CFR §1.162-1
        """

        pattern = (
            r"\d{1,3}\s+C\.?F\.?R\.?\s*"
            r"§{0,2}\s*"
            r"\d+(?:\.\d+)*"
            r"(?:\s*[\u2013\-]\s*\d+)?"
        )

        matches = re.findall(pattern, text)

        return self._clean_and_dedupe(matches)

    def extract_dates(self, text: str) -> List[str]:
        """
        Extract dates such as "Sept. 30, 2023" or "January 5, 2024".
        """

        months = (
            r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|May|Jun\.?|Jul\.?|Aug\.?|"
            r"Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?|"
            r"January|February|March|April|June|July|"
            r"August|September|October|November|December"
        )

        pattern = rf"\b(?:{months})\s+\d{{1,2}},?\s+\d{{4}}\b"

        matches = re.findall(pattern, text)

        return self._clean_and_dedupe(matches)

    def extract_act_names(self, text: str) -> List[str]:
        """
        Extract named Acts, e.g. "Federal Food, Drug, and Cosmetic
        Act" or "Homeland Security Act of 2002".
        """

        pattern = (
            r"\b[A-Z][a-zA-Z]*(?:,)?"
            r"(?:\s+(?:[A-Z][a-zA-Z]*|and|of|the|for)(?:,)?){0,9}"
            r"\s+Act\b(?:\s+of\s+\d{4})?"
        )

        matches = re.findall(pattern, text)

        cleaned = [match.strip(" ,") for match in matches]

        cleaned = [
            match for match in cleaned
            if not self._is_generic_act_reference(match)
        ]

        return self._clean_and_dedupe(cleaned)

    def _is_generic_act_reference(self, act_name: str) -> bool:
        """
        Filters out generic, non-named references like "This Act",
        "That Act", or "Congress An Act" that the regex can pick up
        from ordinary phrases (e.g. "An Act Making appropriations...")
        rather than an actual named statute.
        """

        generic_lead_words = {"this", "that", "the", "congress", "such", "said"}

        first_word = act_name.split()[0].lower() if act_name.split() else ""

        # Require at least 2 words before "Act" for a real statute name
        word_count_before_act = len(act_name.split()) - 1

        return (
            first_word in generic_lead_words
            or word_count_before_act < 2
        )

    def extract_code_references(self, text: str) -> List[str]:
        """
        Extract named legal codes/regulations that are not "Acts"
        in the statute-name sense but are still important legal
        sources, e.g. "Internal Revenue Code", "IRC", "Treasury
        Regulations", "Code of Federal Regulations".
        """

        known_codes = [
            "Internal Revenue Code",
            "Treasury Regulations",
            "Code of Federal Regulations",
            "Federal Register",
            "IRC",
        ]

        found = [
            code for code in known_codes
            if re.search(r"\b" + re.escape(code) + r"\b", text)
        ]

        return self._clean_and_dedupe(found)

    def _tokenize_for_keywords(self, text: str) -> List[str]:
        """
        Shared tokenizer for keyword extraction: lowercase,
        alphabetic tokens of 4+ letters, minus stopwords. Excludes
        digits entirely, so citation numbers never enter the
        keyword pool.
        """

        words = re.findall(r"[A-Za-z]{4,}", text.lower())

        return [word for word in words if word not in STOPWORDS]

    def _strip_known_matches(self, text: str, matches: List[str]) -> str:
        """
        Remove already-extracted citation/date substrings from a
        copy of the text before keyword tokenization, so recurring
        citation fragments (e.g. "Federal", "Drug") don't dominate
        the keyword list just because they're part of a citation
        that's already captured elsewhere in the metadata.
        """

        cleaned = text

        for match in matches:
            cleaned = cleaned.replace(match, " ")

        return cleaned

    def extract_keywords(
        self,
        text: str,
        idf_scores: Optional[Dict[str, float]] = None,
        top_n: int = 8,
    ) -> List[str]:
        """
        Keyword extraction based on term frequency, optionally
        weighted by corpus-level inverse document frequency
        (TF-IDF) when `idf_scores` is supplied by the pipeline.

        Without idf_scores (e.g. when testing this method in
        isolation on a single chunk), falls back to plain term
        frequency. With idf_scores (the normal pipeline path),
        common legal boilerplate ("federal", "secretary", "united")
        gets down-weighted because it appears in almost every
        chunk, surfacing terms that are actually distinctive to
        this chunk.
        """

        tokens = self._tokenize_for_keywords(text)

        if not tokens:
            return []

        term_counts = Counter(tokens)

        if idf_scores:

            scored_terms = {
                term: count * idf_scores.get(term, 1.0)
                for term, count in term_counts.items()
            }

            top_terms = sorted(
                scored_terms.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:top_n]

            return [term for term, _ in top_terms]

        top_terms = term_counts.most_common(top_n)

        return [term for term, _ in top_terms]

    # -------------------------------------------------
    # Acts-specific extractors
    # -------------------------------------------------

    def extract_sections(self, text: str) -> List[str]:
        """
        Extract section headings ("SEC. 2303."), inline section
        references ("Section 740(a)(1)(A)(ii)"), and section-symbol
        citations ("§162", "§ 162", "§§162-163").
        """

        heading_pattern = r"SEC\.\s*(\d+[A-Za-z]*)"

        reference_pattern = (
            r"[Ss]ection\s+(\d+[A-Za-z]*(?:\([A-Za-z0-9]+\))*)"
        )

        symbol_pattern = (
            r"§{1,2}\s*(\d+[A-Za-z]*(?:\s*[\u2013\-]\s*\d+[A-Za-z]*)?)"
        )

        matches = (
            re.findall(heading_pattern, text)
            + re.findall(reference_pattern, text)
            + re.findall(symbol_pattern, text)
        )

        return self._clean_and_dedupe(matches)

    def extract_articles(self, text: str) -> List[str]:
        """
        Extract article references, e.g. "Article 5" / "Article IV".
        """

        pattern = r"Article\s+([IVXLCDM]+|\d+)"

        matches = re.findall(pattern, text)

        return self._clean_and_dedupe(matches)

    def extract_chapters(self, text: str) -> List[str]:
        """
        Extract chapter references, e.g. "CHAPTER III".
        """

        pattern = r"CHAPTER\s+([IVXLCDM]+|\d+)"

        matches = re.findall(pattern, text)

        return self._clean_and_dedupe(matches)

    def extract_titles(self, text: str) -> List[str]:
        """
        Extract title references, e.g. "TITLE III".
        """

        pattern = r"TITLE\s+([IVXLCDM]+)\b"

        matches = re.findall(pattern, text)

        return self._clean_and_dedupe(matches)

    # -------------------------------------------------
    # Judgment-specific extractors
    # -------------------------------------------------

    def extract_case_names(self, text: str) -> List[str]:
        """
        Extract case names in the "Party v. Party" format,
        e.g. "Smith v. Jones" or "Welch v. Helvering".
        """

        pattern = (
            r"\b[A-Z][A-Za-z.&'\-]*(?:\s+[A-Z][A-Za-z.&'\-]*){0,4}"
            r"\s+v\.?\s+"
            r"[A-Z][A-Za-z.&'\-]*(?:\s+[A-Z][A-Za-z.&'\-]*){0,4}\b"
        )

        matches = re.findall(pattern, text)

        cleaned = [self._strip_case_name_leadin(match) for match in matches]

        return self._clean_and_dedupe(cleaned)

    def _strip_case_name_leadin(self, case_name: str) -> str:
        """
        The case-name regex is greedy about capitalized words, so it
        can accidentally swallow a preceding capitalized lead-in
        word (e.g. "In Welch v. Helvering" instead of just
        "Welch v. Helvering"). Strip known lead-ins from the front.
        """

        lead_ins = {
            "In", "See", "Cf", "Accordingly", "However", "Moreover",
            "Therefore", "Compare", "Cited", "Citing", "Following",
        }

        words = case_name.split()

        while words and words[0].rstrip(".,") in lead_ins:
            words = words[1:]

        return " ".join(words)

    def extract_court_names(self, text: str) -> List[str]:
        """
        Extract known court names using a fixed reference list.

        Matches longest form first (e.g. "United States Supreme
        Court" rather than the truncated "Supreme Court") by trying
        candidates longest-first and suppressing any match whose
        span is already covered by a longer match found earlier.
        """

        known_courts = [
            "United States Supreme Court",
            "U.S. Supreme Court",
            "Supreme Court",
            "United States Court of Appeals",
            "U.S. Court of Appeals",
            "Court of Appeals",
            "United States District Court",
            "U.S. District Court",
            "District Court",
            "United States Tax Court",
            "U.S. Tax Court",
            "Tax Court",
            "United States Court of Federal Claims",
            "Court of Federal Claims",
            "United States Bankruptcy Court",
            "Bankruptcy Court",
            "Circuit Court",
            "Federal Circuit",
        ]

        known_courts_by_length = sorted(
            known_courts, key=len, reverse=True
        )

        claimed_spans: List[tuple] = []

        found: List[str] = []

        for court in known_courts_by_length:

            for match in re.finditer(re.escape(court), text, flags=re.IGNORECASE):

                start, end = match.span()

                already_covered = any(
                    start >= span_start and end <= span_end
                    for span_start, span_end in claimed_spans
                )

                if already_covered:
                    continue

                claimed_spans.append((start, end))

                found.append(court)

        return self._clean_and_dedupe(found)

    def extract_judge_names(self, text: str) -> List[str]:
        """
        Extract judge / justice names, e.g. "Justice Roberts".
        """

        pattern = r"\b(?:Judge|Justice)\s+([A-Z][A-Za-z.]+(?:\s+[A-Z][A-Za-z.]+)?)"

        matches = re.findall(pattern, text)

        return self._clean_and_dedupe(matches)

    # -------------------------------------------------
    # Tax-specific extractors
    # -------------------------------------------------

    def extract_irs_publications(self, text: str) -> List[str]:
        """
        Extract IRS publication references, e.g. "Publication 17".
        """

        pattern = r"Publication\s+\d+[A-Za-z\-]*"

        matches = re.findall(pattern, text)

        return self._clean_and_dedupe(matches)

    def extract_tax_forms(self, text: str) -> List[str]:
        """
        Extract tax form / schedule references, e.g. "Schedule C",
        "Form 1040".
        """

        pattern = r"\b(?:Form\s+\d+[A-Za-z\-]*|Schedule\s+[A-Z])\b"

        matches = re.findall(pattern, text)

        return self._clean_and_dedupe(matches)

    def extract_tax_years(self, text: str) -> List[str]:
        """
        Extract tax/fiscal year references, e.g. "fiscal year 2024".
        """

        pattern = r"(?:tax|fiscal)\s+year\s+(\d{4})"

        matches = re.findall(pattern, text, flags=re.IGNORECASE)

        return self._clean_and_dedupe(matches)

    # -------------------------------------------------
    # POV-specific extractors
    # -------------------------------------------------

    def extract_author(self, text: str) -> Optional[str]:
        """
        Best-effort author extraction for commentary / POV
        documents, e.g. "By John Doe" or "Author: John Doe".
        """

        pattern = r"(?:By|Author:)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})"

        match = re.search(pattern, text)

        if match:
            return match.group(1).strip()

        return None

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _clean_and_dedupe(self, items: List[str]) -> List[str]:
        """
        Strip whitespace, drop empties, remove duplicates while
        preserving order.
        """

        seen = set()

        cleaned = []

        for item in items:

            value = re.sub(r"\s+", " ", item).strip()

            if not value or value in seen:
                continue

            seen.add(value)

            cleaned.append(value)

        return cleaned

    # -------------------------------------------------
    # Common metadata (applies to every document type)
    # -------------------------------------------------

    def extract_common_metadata(
        self,
        text: str,
        idf_scores: Optional[Dict[str, float]] = None,
    ) -> Dict:

        usc_citations = self.extract_usc_citations(text)

        cfr_citations = self.extract_cfr_citations(text)

        dates = self.extract_dates(text)

        act_names = self.extract_act_names(text)

        code_references = self.extract_code_references(text)

        # Strip already-extracted citation/date substrings before
        # tokenizing for keywords, so recurring citation fragments
        # don't crowd out genuinely distinctive terms (Issue 5).
        text_for_keywords = self._strip_known_matches(
            text,
            usc_citations + cfr_citations + dates,
        )

        keywords = self.extract_keywords(
            text_for_keywords,
            idf_scores=idf_scores,
        )

        return {
            "usc_citations": usc_citations,
            "cfr_citations": cfr_citations,
            "dates": dates,
            "keywords": keywords,
            "entities": {
                "acts": act_names,
                "code_references": code_references,
            },
        }

    # -------------------------------------------------
    # Document-type-specific metadata
    # -------------------------------------------------

    def extract_act_metadata(self, text: str) -> Dict:

        return {
            "sections": self.extract_sections(text),
            "articles": self.extract_articles(text),
            "chapters": self.extract_chapters(text),
            "titles": self.extract_titles(text),
        }

    def extract_judgment_metadata(self, text: str) -> Dict:

        return {
            "sections": self.extract_sections(text),
            "entities": {
                "cases": self.extract_case_names(text),
                "courts": self.extract_court_names(text),
                "judges": self.extract_judge_names(text),
            },
        }

    def extract_tax_metadata(self, text: str) -> Dict:

        return {
            "irs_publications": self.extract_irs_publications(text),
            "tax_forms": self.extract_tax_forms(text),
            "tax_years": self.extract_tax_years(text),
            "sections": self.extract_sections(text),
        }

    def extract_pov_metadata(self, text: str) -> Dict:

        return {
            "sections": self.extract_sections(text),
            "entities": {
                "cases": self.extract_case_names(text),
                "author": self.extract_author(text),
            },
        }

    # -------------------------------------------------
    # Dispatcher : combine common + type-specific metadata
    # -------------------------------------------------

    def extract_all_metadata(
        self,
        chunk: Dict,
        idf_scores: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Full metadata extraction pipeline for a single chunk.
        Combines common metadata with document-type-specific
        metadata, then drops empty fields to keep the JSON clean.
        """

        text = chunk.get("text", "")

        document_type = chunk.get("document_type", "")

        common_metadata = self.extract_common_metadata(
            text,
            idf_scores=idf_scores,
        )

        type_extractors = {
            "acts": self.extract_act_metadata,
            "judgments": self.extract_judgment_metadata,
            "tax": self.extract_tax_metadata,
            "pov": self.extract_pov_metadata,
        }

        extractor_fn = type_extractors.get(document_type)

        specific_metadata = extractor_fn(text) if extractor_fn else {}

        merged_metadata = self._merge_metadata(
            common_metadata,
            specific_metadata,
        )

        return self._drop_empty_fields(merged_metadata)

    def _merge_metadata(self, common: Dict, specific: Dict) -> Dict:
        """
        Merge type-specific metadata into common metadata. The
        "entities" key is deep-merged (common entities like "acts"
        and "code_references" are preserved alongside type-specific
        entities like "cases" or "courts"); every other key is a
        plain overwrite/add.
        """

        merged = dict(common)

        for key, value in specific.items():

            if key == "entities" and isinstance(merged.get("entities"), dict):
                merged["entities"] = {**merged["entities"], **value}
            else:
                merged[key] = value

        return merged

    def _drop_empty_fields(self, metadata: Dict) -> Dict:
        """
        Remove keys whose value is None, an empty list, or an
        empty string, so unrelated document types don't carry
        clutter fields. Recurses one level into the nested
        "entities" dict, and drops "entities" entirely if nothing
        inside it survives.
        """

        cleaned = {}

        for key, value in metadata.items():

            if key == "entities" and isinstance(value, dict):

                nested_cleaned = {
                    nested_key: nested_value
                    for nested_key, nested_value in value.items()
                    if not self._is_empty(nested_value)
                }

                if nested_cleaned:
                    cleaned["entities"] = nested_cleaned

                continue

            if self._is_empty(value):
                continue

            cleaned[key] = value

        return cleaned

    def _is_empty(self, value) -> bool:

        if value is None:
            return True

        if isinstance(value, (list, str)) and len(value) == 0:
            return True

        return False


class MetadataPipeline:
    """
    Orchestrates loading chunks, enriching them with metadata,
    saving the result, and printing summary statistics.
    """

    def __init__(self):

        self.extractor = MetadataExtractor()

        self.total_chunks = 0

        self.enriched_chunks: List[Dict] = []

        self.field_coverage: Counter = Counter()

    # -------------------------------------------------

    def load_chunks(self) -> List[Dict]:

        logger.info("Loading chunks.json")

        with open(
            CHUNKS_JSON,
            "r",
            encoding="utf-8",
        ) as file:

            chunks = json.load(file)

        self.total_chunks = len(chunks)

        logger.info(f"{self.total_chunks} chunks loaded.")

        return chunks

    # -------------------------------------------------

    def build_corpus_idf_scores(
        self,
        chunks: List[Dict],
    ) -> Optional[Dict[str, float]]:
        """
        Compute corpus-level inverse-document-frequency scores so
        keyword extraction can down-weight terms that appear in
        almost every chunk (e.g. "federal", "secretary", "united")
        and surface terms that are actually distinctive to a given
        chunk. Skipped for trivially small corpora, where document
        frequency is meaningless.
        """

        if len(chunks) < 2:
            return None

        logger.info("Building corpus-level IDF scores for keyword scoring")

        document_frequencies: Counter = Counter()

        for chunk in chunks:

            tokens = set(
                self.extractor._tokenize_for_keywords(chunk.get("text", ""))
            )

            document_frequencies.update(tokens)

        total_documents = len(chunks)

        idf_scores = {
            term: math.log((total_documents + 1) / (doc_freq + 1)) + 1
            for term, doc_freq in document_frequencies.items()
        }

        return idf_scores

    # -------------------------------------------------

    def enrich_chunks(
        self,
        chunks: List[Dict],
        idf_scores: Optional[Dict[str, float]] = None,
    ):

        logger.info("Extracting metadata...")

        for chunk in chunks:

            try:
                metadata = self.extractor.extract_all_metadata(
                    chunk,
                    idf_scores=idf_scores,
                )

            except Exception as error:

                logger.error(
                    f"Failed to extract metadata for chunk "
                    f"'{chunk.get('chunk_id')}': {error}"
                )
                metadata = {}

            enriched_chunk = chunk.copy()

            enriched_chunk["metadata"] = metadata

            self.enriched_chunks.append(enriched_chunk)

            for field, value in metadata.items():

                if field == "entities" and isinstance(value, dict):
                    for nested_field in value.keys():
                        self.field_coverage[f"entities.{nested_field}"] += 1
                    continue

                self.field_coverage[field] += 1

    # -------------------------------------------------

    def save_chunks(self):

        logger.info("Saving chunks_with_metadata.json")

        with open(
            CHUNKS_WITH_METADATA_JSON,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.enriched_chunks,
                file,
                indent=4,
                ensure_ascii=False,
            )

    # -------------------------------------------------

    def print_summary(self):

        logger.info("=" * 60)

        logger.info("Metadata Extraction Summary")

        logger.info("=" * 60)

        logger.info(f"Total Chunks : {self.total_chunks}")

        logger.info(f"Output File  : {CHUNKS_WITH_METADATA_JSON}")

        logger.info("-" * 60)

        logger.info("Field coverage (chunks containing each field):")

        for field, count in self.field_coverage.most_common():

            percentage = (
                (count / self.total_chunks) * 100
                if self.total_chunks
                else 0
            )

            logger.info(
                f"  {field:20s}: {count:5d} chunks ({percentage:.1f}%)"
            )

        logger.info("=" * 60)

    # -------------------------------------------------

    def run(self):

        chunks = self.load_chunks()

        idf_scores = self.build_corpus_idf_scores(chunks)

        self.enrich_chunks(chunks, idf_scores=idf_scores)

        self.save_chunks()

        self.print_summary()


def main():

    pipeline = MetadataPipeline()

    pipeline.run()


if __name__ == "__main__":

    main()