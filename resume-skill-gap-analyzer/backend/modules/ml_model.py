"""
=============================================================================
 Machine Learning Model Module
=============================================================================
 Role in the pipeline:
   This is the FOURTH stage. It uses two ML classifiers to predict whether
   a candidate truly "has" each skill based on resume and GitHub evidence.

 Models used:
   1. Logistic Regression (with StandardScaler) — a linear model that
      outputs calibrated probabilities. Good for understanding feature
      importance via coefficients.

   2. Decision Tree (max_depth=4) — a non-linear model that creates
      interpretable if/then rules. Good for explaining decisions.

 Why synthetic data?
   In a real system, we'd have labeled data from recruiters. For this
   demo/prototype, we generate synthetic training data that captures
   the core pattern: if a skill appears in resume OR GitHub, it's
   likely a real skill of the candidate.
=============================================================================
"""

import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


class SkillGapMLModel:
    """Trains and runs ML models for skill presence prediction."""

    def __init__(self) -> None:
        """
        Initialize both ML models.

        Logistic Regression:
          - Wrapped in a Pipeline with StandardScaler for feature normalization
          - max_iter=200 to ensure convergence on small datasets
          - random_state=42 for reproducible results

        Decision Tree:
          - max_depth=4 to prevent overfitting (keeps the tree interpretable)
          - random_state=42 for reproducibility
        """
        # Logistic Regression pipeline: scale features → classify
        self.lr_pipeline = Pipeline([
            ("scaler", StandardScaler()),        # Normalizes features to zero mean, unit variance
            ("classifier", LogisticRegression(
                random_state=42,
                max_iter=200,
            )),
        ])

        # Decision Tree classifier — deliberately shallow for interpretability
        self.dt_model = DecisionTreeClassifier(
            max_depth=4,
            random_state=42,
        )

        # Track training accuracy for reporting
        self.lr_accuracy: float = 0.0
        self.dt_accuracy: float = 0.0
        self.is_trained: bool = False

        print("   [MLModel] Initialized Logistic Regression + Decision Tree models.")

    # -----------------------------------------------------------------
    #  Generate Synthetic Training Data
    # -----------------------------------------------------------------
    def generate_training_data(self, all_skills_master: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic training data for the ML models.

        Why synthetic data?
          In a production system, we would use labeled data from actual
          hiring decisions. For this demo, we simulate the relationship
          between skill evidence (resume/GitHub) and true skill presence.

        Data generation logic:
          - 200 samples total
          - Each sample has two features: in_resume (0/1), in_github (0/1)
          - Label (skill_present): follows these rules:
            * Both sources say yes → always 1 (strong evidence)
            * Only resume says yes → 1 with 85% probability (resumes can exaggerate)
            * Only GitHub says yes → 1 with 90% probability (code doesn't lie)
            * Neither source → always 0 (no evidence)
          - This captures the real-world intuition that GitHub evidence
            is slightly more reliable than resume claims alone.

        Args:
            all_skills_master: The full skills master dict (used for sizing).

        Returns:
            A tuple (X, y) of training features and labels.
        """
        random.seed(42)  # Reproducible results
        np.random.seed(42)

        n_samples = 200
        data = []

        for _ in range(n_samples):
            in_resume = random.randint(0, 1)
            in_github = random.randint(0, 1)

            # Determine the label based on evidence combination
            if in_resume == 1 and in_github == 1:
                # Both sources confirm — strong evidence, always present
                skill_present = 1
            elif in_resume == 1 and in_github == 0:
                # Only resume claims it — 85% chance it's real
                # (resumes sometimes exaggerate or list outdated skills)
                skill_present = 1 if random.random() < 0.85 else 0
            elif in_resume == 0 and in_github == 1:
                # Only GitHub shows it — 90% chance it's real
                # (code evidence is more reliable than claims)
                skill_present = 1 if random.random() < 0.90 else 0
            else:
                # Neither source — no evidence, skill is absent
                skill_present = 0

            data.append({
                "in_resume": in_resume,
                "in_github": in_github,
                "skill_present": skill_present,
            })

        df = pd.DataFrame(data)

        # Separate features (X) and labels (y)
        X = df[["in_resume", "in_github"]]
        y = df["skill_present"]

        print(f"   [MLModel] Generated {n_samples} synthetic training samples.")
        print(f"   [MLModel] Label distribution: {dict(y.value_counts())}")
        return X, y

    # -----------------------------------------------------------------
    #  Train Both Models
    # -----------------------------------------------------------------
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train both ML models on the provided data.

        Both models learn the same pattern: the relationship between
        skill evidence (in_resume, in_github) and true skill presence.

        Args:
            X: Feature DataFrame with columns [in_resume, in_github].
            y: Label Series with values 0 or 1.

        Returns:
            A dict with training accuracy for each model.
        """
        print("   [MLModel] Training Logistic Regression...")
        self.lr_pipeline.fit(X, y)
        self.lr_accuracy = round(self.lr_pipeline.score(X, y) * 100, 2)

        print("   [MLModel] Training Decision Tree...")
        self.dt_model.fit(X, y)
        self.dt_accuracy = round(self.dt_model.score(X, y) * 100, 2)

        self.is_trained = True

        print(f"   [MLModel] LR Accuracy: {self.lr_accuracy}%")
        print(f"   [MLModel] DT Accuracy: {self.dt_accuracy}%")

        return {
            "lr_accuracy": self.lr_accuracy,
            "dt_accuracy": self.dt_accuracy,
        }

    # -----------------------------------------------------------------
    #  Make Predictions
    # -----------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> Dict[str, list]:
        """
        Run predictions using both trained models.

        For each skill (row in X), predicts:
          - Whether the skill is "present" (binary 0/1 from both models)
          - The probability of presence (from Logistic Regression only)

        The LR probability is particularly useful — a high probability
        (e.g., 0.92) means the model is confident the skill is real,
        while a low probability (e.g., 0.35) suggests weak evidence.

        Args:
            X: Feature DataFrame with columns [in_resume, in_github].

        Returns:
            A dict with:
              - lr_predictions:   List of 0/1 from Logistic Regression
              - dt_predictions:   List of 0/1 from Decision Tree
              - lr_probabilities: List of floats (probability of class 1)
        """
        if not self.is_trained:
            print("   [MLModel] WARNING: Models not trained yet!")
            return {
                "lr_predictions": [],
                "dt_predictions": [],
                "lr_probabilities": [],
            }

        # Logistic Regression predictions + probabilities
        lr_preds = self.lr_pipeline.predict(X).tolist()
        lr_probs = self.lr_pipeline.predict_proba(X)[:, 1].tolist()  # Probability of class 1

        # Decision Tree predictions (no calibrated probabilities)
        dt_preds = self.dt_model.predict(X).tolist()

        # Round probabilities for cleaner output
        lr_probs = [round(p, 4) for p in lr_probs]

        print(f"   [MLModel] Generated predictions for {len(X)} skills.")
        return {
            "lr_predictions": lr_preds,
            "dt_predictions": dt_preds,
            "lr_probabilities": lr_probs,
        }

    # -----------------------------------------------------------------
    #  Feature Importance (Decision Tree)
    # -----------------------------------------------------------------
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Extract feature importance scores from the Decision Tree.

        Feature importance tells us which input features (in_resume, in_github)
        the tree relies on most when making decisions. A higher score means
        that feature is more important for the prediction.

        Returns:
            A dict mapping feature names to importance scores (sum to 1.0).
        """
        if not self.is_trained:
            return {}

        feature_names = ["in_resume", "in_github"]
        importances = self.dt_model.feature_importances_

        importance_dict = {
            name: round(float(imp), 4)
            for name, imp in zip(feature_names, importances)
        }

        print(f"   [MLModel] Feature importances: {importance_dict}")
        return importance_dict

    # -----------------------------------------------------------------
    #  Model Summary (Human-Readable)
    # -----------------------------------------------------------------
    def get_model_summary(self) -> Dict:
        """
        Generate a human-readable summary of both models.

        This is used in the final report to explain what the ML models
        did and how they contributed to the analysis — written in plain
        English for non-technical stakeholders.

        Returns:
            A dict with model details, accuracy, and explanations.
        """
        return {
            "models_used": ["Logistic Regression", "Decision Tree"],
            "lr_accuracy": self.lr_accuracy,
            "dt_accuracy": self.dt_accuracy,
            "feature_importance": self.get_feature_importance(),
            "lr_explanation": (
                "Logistic Regression is a linear classifier that estimates the probability "
                "of a skill being genuinely present based on whether it appears in the resume "
                "and/or on GitHub. It outputs a confidence score (0-100%) for each skill, "
                "which we use to assess how reliable the evidence is."
            ),
            "dt_explanation": (
                "Decision Tree is a rule-based classifier that creates simple if/then rules "
                "to determine skill presence. For example: 'If in_resume=1 AND in_github=1, "
                "then skill is present.' It's highly interpretable and helps validate the "
                "Logistic Regression results."
            ),
            "training_explanation": (
                "Both models were trained on 200 synthetic samples that simulate the "
                "relationship between skill evidence (resume mentions + GitHub code) "
                "and actual skill presence. The synthetic data encodes the intuition "
                "that GitHub evidence is slightly more reliable than resume claims."
            ),
        }
