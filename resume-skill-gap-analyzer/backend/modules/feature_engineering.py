"""
=============================================================================
 Feature Engineering Module
=============================================================================
 Role in the pipeline:
   This is the THIRD stage. It transforms the raw skill lists (from the
   resume parser and GitHub analyzer) into structured feature vectors that
   can be fed into the ML models.

 Why feature engineering matters:
   ML models can't understand raw text — they need numerical features.
   We create binary (0/1) features indicating whether each skill was
   found in the resume, on GitHub, or both.

 Improvements over previous version:
   - 4 features instead of 2: in_resume, in_github, both_confirmed, is_required
   - Sentence-transformers for semantic similarity (with TF-IDF fallback)
   - Skill overlap statistics for richer reporting
=============================================================================
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngineer:
    """Transforms raw skill data into structured features for ML models."""

    def __init__(self) -> None:
        """
        Initialize the feature engineer.
        Tries loading sentence-transformers for semantic similarity.
        Falls back to TF-IDF if not available.
        """
        self.use_transformers = False
        self.st_model = None

        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_transformers = True
            logger.info("[FeatureEngineer] Using sentence-transformers "
                        "for semantic similarity")
        except Exception:
            logger.info("[FeatureEngineer] Using TF-IDF "
                        "(sentence-transformers not available)")

        logger.info("[FeatureEngineer] Initialized.")

    # -----------------------------------------------------------------
    #  Build a Full Feature Vector
    # -----------------------------------------------------------------
    def build_feature_vector(
        self,
        claimed_skills: List[str],
        demonstrated_skills: List[str],
        required_skills: List[str],
        nice_to_have_skills: List[str],
    ) -> pd.DataFrame:
        """
        Build a comprehensive feature vector from all skill sources.

        For each unique skill across all lists, we create three binary features:
          - claimed_{skill}:      1 if found in the resume
          - demonstrated_{skill}: 1 if found on GitHub
          - combined_{skill}:     1 if found in either source

        This gives the ML model a rich view of the candidate's skill profile.

        Args:
            claimed_skills:      Skills extracted from the resume.
            demonstrated_skills: Skills found on GitHub.
            required_skills:     Skills required for the target role.
            nice_to_have_skills: Nice-to-have skills for the target role.

        Returns:
            A pandas DataFrame with one row and many binary feature columns.
        """
        # Get the union of all skills to define our feature space
        all_skills = set(claimed_skills + demonstrated_skills + required_skills + nice_to_have_skills)

        # Build feature dictionary — each skill gets three binary indicators
        features: Dict[str, int] = {}
        for skill in sorted(all_skills):
            # Sanitize skill name for use as a column name
            safe_name = skill.lower().replace(" ", "_").replace("/", "_").replace(".", "_")

            # Is this skill claimed in the resume?
            features[f"claimed_{safe_name}"] = 1 if skill in claimed_skills else 0

            # Is this skill demonstrated on GitHub?
            features[f"demonstrated_{safe_name}"] = 1 if skill in demonstrated_skills else 0

            # Is this skill present in either source?
            features[f"combined_{safe_name}"] = (
                1 if (skill in claimed_skills or skill in demonstrated_skills) else 0
            )

        # Create a single-row DataFrame from the feature dictionary
        df = pd.DataFrame([features])
        logger.debug(f"[FeatureEngineer] Built feature vector with {len(df.columns)} features.")
        return df

    # -----------------------------------------------------------------
    #  Create Skill Matrix (for ML Model Input)
    # -----------------------------------------------------------------
    def create_skill_matrix(
        self,
        claimed_skills: List[str],
        demonstrated_skills: List[str],
        required_skills: List[str],
        nice_to_have_skills: List[str],
    ) -> pd.DataFrame:
        """
        Create a skill-by-skill matrix used for ML prediction and analysis.

        Each row represents one skill from the target role's requirements.
        Columns indicate:
          - skill_name:       The skill identifier
          - category:         "required" or "nice_to_have"
          - in_resume:        1 if found in resume, 0 otherwise
          - in_github:        1 if found on GitHub, 0 otherwise
          - combined:         1 if found in either source
          - both_confirmed:   1 if found in BOTH sources (strongest evidence)
          - is_required:      1 if skill is required for the role

        This matrix is what gets fed into the ML model for predictions
        and is also used directly for gap analysis.

        Args:
            claimed_skills:      Skills from the resume.
            demonstrated_skills: Skills from GitHub.
            required_skills:     Skills required for the role.
            nice_to_have_skills: Nice-to-have skills for the role.

        Returns:
            A pandas DataFrame with one row per skill.
        """
        rows = []

        # Process required skills first
        for skill in required_skills:
            in_resume = 1 if skill in claimed_skills else 0
            in_github = 1 if skill in demonstrated_skills else 0
            rows.append({
                "skill_name": skill,
                "category": "required",
                "in_resume": in_resume,
                "in_github": in_github,
                "combined": 1 if (in_resume or in_github) else 0,
                "both_confirmed": 1 if (in_resume and in_github) else 0,
                "is_required": 1,
            })

        # Then process nice-to-have skills
        for skill in nice_to_have_skills:
            in_resume = 1 if skill in claimed_skills else 0
            in_github = 1 if skill in demonstrated_skills else 0
            rows.append({
                "skill_name": skill,
                "category": "nice_to_have",
                "in_resume": in_resume,
                "in_github": in_github,
                "combined": 1 if (in_resume or in_github) else 0,
                "both_confirmed": 1 if (in_resume and in_github) else 0,
                "is_required": 0,
            })

        # Build the DataFrame from the list of row dicts
        df = pd.DataFrame(rows)
        logger.debug(f"[FeatureEngineer] Created skill matrix with {len(df)} skills "
                     f"(4 features per skill).")
        return df

    # -----------------------------------------------------------------
    #  Encode for ML Model
    # -----------------------------------------------------------------
    def encode_for_model(
        self,
        skill_matrix: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract the feature matrix (X) and label vector (y) from the skill matrix.

        X contains the input features the model uses for prediction:
          - in_resume       (binary) — skill claimed in resume
          - in_github       (binary) — skill demonstrated on GitHub
          - both_confirmed  (binary) — confirmed by both sources
          - is_required     (binary) — required for the target role

        y is the target label:
          - combined (1 if the skill is present in either source)

        Args:
            skill_matrix: The DataFrame from create_skill_matrix().

        Returns:
            A tuple of (X, y) where X is a DataFrame and y is a Series.
        """
        # Features: 4-dimensional feature vector per skill
        X = skill_matrix[["in_resume", "in_github",
                          "both_confirmed", "is_required"]].copy()

        # Label: whether the skill is "present" (found in at least one source)
        y = skill_matrix["combined"].copy()

        logger.debug(f"[FeatureEngineer] Encoded {len(X)} samples for model (4 features).")
        return X, y

    # -----------------------------------------------------------------
    #  Semantic Similarity
    # -----------------------------------------------------------------
    def compute_semantic_similarity(
        self,
        resume_text: str,
        job_description: str,
    ) -> float:
        """
        Compute semantic similarity between resume text and job description.

        If sentence-transformers is available:
          - Encode both texts with all-MiniLM-L6-v2
          - Compute cosine similarity
          - Return float 0-1

        Else (TF-IDF fallback):
          - Vectorize with TF-IDF
          - Compute cosine similarity
          - Return float 0-1

        Always returns 0.5 on any failure (neutral score).
        """
        try:
            if self.use_transformers and self.st_model is not None:
                # Sentence-transformers path
                embeddings = self.st_model.encode(
                    [resume_text, job_description],
                    convert_to_numpy=True
                )
                # Cosine similarity between the two embeddings
                from numpy.linalg import norm
                cos_sim = float(
                    np.dot(embeddings[0], embeddings[1]) /
                    (norm(embeddings[0]) * norm(embeddings[1]) + 1e-8)
                )
                return max(0.0, min(1.0, cos_sim))
            else:
                # TF-IDF fallback
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity

                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(
                    [resume_text, job_description]
                )
                sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                return float(sim[0][0])

        except Exception as e:
            logger.warning(f"Semantic similarity failed: {e}")
            return 0.5

    # -----------------------------------------------------------------
    #  Skill Overlap Statistics
    # -----------------------------------------------------------------
    def get_skill_overlap_stats(
        self,
        claimed_skills: List[str],
        demonstrated_skills: List[str],
        required_skills: List[str],
    ) -> dict:
        """
        Returns statistics about skill overlap between
        resume, github, and job requirements.
        Used for richer reporting.

        Returns:
            {
              resume_coverage:   % of required skills in resume,
              github_coverage:   % of required skills in github,
              combined_coverage: % covered by either,
              overlap_rate:      % of resume skills also in github,
              unique_to_resume:  count,
              unique_to_github:  count,
            }
        """
        claimed_set = set(claimed_skills)
        demonstrated_set = set(demonstrated_skills)
        required_set = set(required_skills)

        total_required = max(len(required_set), 1)

        resume_in_required = claimed_set & required_set
        github_in_required = demonstrated_set & required_set
        combined_in_required = (claimed_set | demonstrated_set) & required_set

        # Overlap between resume and github skills (across all skills)
        all_candidate_skills = claimed_set | demonstrated_set
        overlap = claimed_set & demonstrated_set

        return {
            "resume_coverage": round(len(resume_in_required) / total_required * 100, 1),
            "github_coverage": round(len(github_in_required) / total_required * 100, 1),
            "combined_coverage": round(len(combined_in_required) / total_required * 100, 1),
            "overlap_rate": round(
                len(overlap) / max(len(claimed_set), 1) * 100, 1
            ),
            "unique_to_resume": len(claimed_set - demonstrated_set),
            "unique_to_github": len(demonstrated_set - claimed_set),
        }
