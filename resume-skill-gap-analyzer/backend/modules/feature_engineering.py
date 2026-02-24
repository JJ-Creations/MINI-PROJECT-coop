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
=============================================================================
"""

from typing import Dict, List, Tuple

import pandas as pd


class FeatureEngineer:
    """Transforms raw skill data into structured features for ML models."""

    def __init__(self) -> None:
        """Initialize the feature engineer."""
        print("   [FeatureEngineer] Initialized.")

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
        print(f"   [FeatureEngineer] Built feature vector with {len(df.columns)} features.")
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
          - skill_name:  The skill identifier
          - category:    "required" or "nice_to_have"
          - in_resume:   1 if found in resume, 0 otherwise
          - in_github:   1 if found on GitHub, 0 otherwise
          - combined:    1 if found in either source

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
            rows.append({
                "skill_name": skill,
                "category": "required",
                "in_resume": 1 if skill in claimed_skills else 0,
                "in_github": 1 if skill in demonstrated_skills else 0,
                "combined": 1 if (skill in claimed_skills or skill in demonstrated_skills) else 0,
            })

        # Then process nice-to-have skills
        for skill in nice_to_have_skills:
            rows.append({
                "skill_name": skill,
                "category": "nice_to_have",
                "in_resume": 1 if skill in claimed_skills else 0,
                "in_github": 1 if skill in demonstrated_skills else 0,
                "combined": 1 if (skill in claimed_skills or skill in demonstrated_skills) else 0,
            })

        # Build the DataFrame from the list of row dicts
        df = pd.DataFrame(rows)
        print(f"   [FeatureEngineer] Created skill matrix with {len(df)} skills.")
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
          - in_resume (binary)
          - in_github (binary)

        y is the target label:
          - combined (1 if the skill is present in either source)

        This encoding is used for both training (with synthetic data)
        and prediction (with real candidate data).

        Args:
            skill_matrix: The DataFrame from create_skill_matrix().

        Returns:
            A tuple of (X, y) where X is a DataFrame and y is a Series.
        """
        # Features: whether the skill is in the resume and/or on GitHub
        X = skill_matrix[["in_resume", "in_github"]].copy()

        # Label: whether the skill is "present" (found in at least one source)
        y = skill_matrix["combined"].copy()

        print(f"   [FeatureEngineer] Encoded {len(X)} samples for model (2 features).")
        return X, y
