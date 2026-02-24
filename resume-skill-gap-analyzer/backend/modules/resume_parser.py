"""
=============================================================================
 Resume Parser Module
=============================================================================
 Role in the pipeline:
   This is the FIRST stage of the analysis pipeline. It takes in a resume
   file (PDF or TXT), extracts the raw text, and then identifies technical
   skills by matching against the master skills list.

 Techniques used:
   - PyMuPDF (fitz) for PDF text extraction
   - spaCy NLP for noun-chunk and named-entity extraction
   - Regex word-boundary matching for precise skill detection
=============================================================================
"""

import re
from typing import Dict, List

import fitz  # PyMuPDF — high-performance PDF text extraction
import spacy


class ResumeParser:
    """Parses resume files and extracts technical skills from the text."""

    def __init__(self) -> None:
        """Initialize the parser by loading the spaCy English NLP model."""
        # Load the small English model — good balance of speed and accuracy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("   [ResumeParser] spaCy model loaded successfully.")
        except OSError:
            # If the model isn't installed, warn the user
            print("   [ResumeParser] WARNING: spaCy model 'en_core_web_sm' not found.")
            print("   Run: python -m spacy download en_core_web_sm")
            self.nlp = None

    # -----------------------------------------------------------------
    #  PDF Text Extraction
    # -----------------------------------------------------------------
    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """
        Extract raw text from a PDF file using PyMuPDF.

        Args:
            file_bytes: The raw bytes of the uploaded PDF file.

        Returns:
            A single string containing all text from every page.
        """
        text = ""
        try:
            # Open the PDF from an in-memory byte stream
            doc = fitz.open(stream=file_bytes, filetype="pdf")

            # Iterate through every page and accumulate text
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                text += page_text + "\n"

            doc.close()
            print(f"   [ResumeParser] Extracted text from PDF ({len(doc)} pages).")

        except Exception as e:
            print(f"   [ResumeParser] ERROR extracting PDF text: {e}")
            text = ""

        return text.strip()

    # -----------------------------------------------------------------
    #  Plain Text Extraction
    # -----------------------------------------------------------------
    def extract_text_from_txt(self, file_bytes: bytes) -> str:
        """
        Decode plain-text resume content from raw bytes.

        Args:
            file_bytes: The raw bytes of the uploaded TXT file.

        Returns:
            The decoded text string.
        """
        try:
            # Try UTF-8 first, fall back to latin-1 for broader compatibility
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")

        print(f"   [ResumeParser] Extracted text from TXT ({len(text)} characters).")
        return text.strip()

    # -----------------------------------------------------------------
    #  Skill Extraction (Core NLP + Regex Logic)
    # -----------------------------------------------------------------
    def extract_skills(self, text: str, skills_master: Dict[str, List[str]]) -> List[str]:
        """
        Identify technical skills mentioned in the resume text.

        Strategy:
          1. Flatten the skills_master dict into a single searchable list.
          2. Use regex word-boundary matching to find each skill in the text.
             This prevents partial matches (e.g., "R" inside "React").
          3. Use spaCy to extract noun chunks and named entities as
             supplementary signals for skill detection.
          4. Deduplicate and sort the final list alphabetically.

        Args:
            text:          The raw resume text (already extracted).
            skills_master: The master skills dictionary keyed by category.

        Returns:
            A sorted, deduplicated list of skill names found in the text.
        """
        found_skills = set()

        # Flatten the master skills dict into one list
        all_skills = []
        for category, skills in skills_master.items():
            all_skills.extend(skills)

        # Lowercase version of the text for case-insensitive matching
        text_lower = text.lower()

        # --- Regex-based skill matching ---
        for skill in all_skills:
            # Build a regex pattern with word boundaries
            # re.escape handles special characters like "C++", "C#", "Node.js"
            pattern = r"\b" + re.escape(skill.lower()) + r"\b"

            # Special handling for skills with special chars that break \b
            if skill in ("C++", "C#"):
                pattern = re.escape(skill.lower())

            if re.search(pattern, text_lower):
                found_skills.add(skill)

        # --- spaCy NLP-based extraction (supplementary) ---
        if self.nlp is not None:
            doc = self.nlp(text)

            # Extract noun chunks (e.g., "machine learning", "data science")
            noun_chunks = [chunk.text.lower().strip() for chunk in doc.noun_chunks]

            # Extract named entities (e.g., "Python", "AWS")
            entities = [ent.text.lower().strip() for ent in doc.ents]

            # Check if any master skill appears in noun chunks or entities
            for skill in all_skills:
                skill_lower = skill.lower()
                if skill_lower in noun_chunks or skill_lower in entities:
                    found_skills.add(skill)

        # Sort alphabetically for consistent output
        result = sorted(found_skills)
        print(f"   [ResumeParser] Found {len(result)} skills in resume text.")
        return result

    # -----------------------------------------------------------------
    #  Main Parse Method (Entry Point)
    # -----------------------------------------------------------------
    def parse(
        self,
        file_bytes: bytes,
        filename: str,
        skills_master: Dict[str, List[str]],
    ) -> Dict:
        """
        Parse a resume file and extract skills — main entry point.

        Routes to the correct text extractor based on file extension,
        then runs skill extraction on the resulting text.

        Args:
            file_bytes:    Raw bytes of the uploaded file.
            filename:      Original filename (used to determine file type).
            skills_master: The master skills dictionary.

        Returns:
            A dict containing:
              - raw_text:         The full extracted text
              - extracted_skills: List of identified skill names
              - skill_count:      Number of skills found
        """
        print(f"\n{'='*60}")
        print(f"   [ResumeParser] Parsing file: {filename}")
        print(f"{'='*60}")

        # Route to the appropriate extractor based on file extension
        if filename.lower().endswith(".pdf"):
            raw_text = self.extract_text_from_pdf(file_bytes)
        elif filename.lower().endswith(".txt"):
            raw_text = self.extract_text_from_txt(file_bytes)
        else:
            # Unsupported format — return empty results
            print(f"   [ResumeParser] ERROR: Unsupported file type: {filename}")
            return {
                "raw_text": "",
                "extracted_skills": [],
                "skill_count": 0,
            }

        # Handle empty extraction (corrupted file, etc.)
        if not raw_text:
            print("   [ResumeParser] WARNING: No text extracted from file.")
            return {
                "raw_text": "",
                "extracted_skills": [],
                "skill_count": 0,
            }

        # Run skill extraction on the raw text
        extracted_skills = self.extract_skills(raw_text, skills_master)

        return {
            "raw_text": raw_text,
            "extracted_skills": extracted_skills,
            "skill_count": len(extracted_skills),
        }
