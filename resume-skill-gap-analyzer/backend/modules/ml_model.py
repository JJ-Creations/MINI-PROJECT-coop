"""
=============================================================================
 ML Model — Logistic Regression + Decision Tree
=============================================================================
 MODELS: Logistic Regression and Decision Tree ONLY.

 Why these two:
   Logistic Regression -> outputs probability 0-1 (confidence score)
   Decision Tree       -> explainable IF/THEN rules + SHAP values
   Ensemble            -> average both probabilities (more robust)

 Improvements over previous version:
   - Real training data from HuggingFace (with synthetic fallback)
   - class_weight='balanced' handles imbalanced skill datasets
   - Cross-validation for honest accuracy reporting
   - GridSearchCV to find best hyperparameters automatically
   - SMOTE oversampling if class imbalance is severe
   - Proper train/test split with stratification
   - Full metrics: accuracy, precision, recall, F1, ROC-AUC
   - SHAP explainability for Decision Tree
   - Model saved to disk, loaded on next startup (faster)
=============================================================================
"""

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class SkillGapMLModel:
    """Trains and runs ML models for skill presence prediction."""

    FEATURE_NAMES = ['in_resume', 'in_github', 'both_confirmed', 'is_required']

    def __init__(self) -> None:
        """
        Initialize both models with optimized hyperparameters.
        class_weight='balanced' is critical — skill datasets are
        imbalanced (more gaps than strong matches in reality).
        """
        # Logistic Regression inside a Pipeline with StandardScaler
        self.lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                solver='lbfgs'
            ))
        ])

        # Decision Tree — no scaler needed, tree-based
        self.dt_model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            class_weight='balanced'
        )

        self.is_trained = False
        self.metrics: Dict = {}
        self.dataset_source = "not trained yet"
        self.model_save_path = Path("./models_saved")
        self.model_save_path.mkdir(exist_ok=True)

        # Backward-compat accuracy fields used by report_generator
        self.lr_accuracy: float = 0.0
        self.dt_accuracy: float = 0.0

        logger.info("[MLModel] Initialized Logistic Regression + Decision Tree models.")

    # -----------------------------------------------------------------
    #  Train Both Models
    # -----------------------------------------------------------------
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_source: str = "synthetic",
        use_cross_validation: bool = True,
        tune_hyperparameters: bool = False,
    ) -> Dict:
        """
        Full training pipeline with evaluation.

        Args:
            X: feature DataFrame with 4 columns
            y: label Series
            dataset_source: description of where data came from
            use_cross_validation: if True, runs 5-fold CV
            tune_hyperparameters: if True, runs GridSearchCV (slower)

        Returns dict with all metrics for both models.
        """
        self.dataset_source = dataset_source
        logger.info(f"Training on {len(X)} samples from {dataset_source}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        # -- Handle class imbalance with SMOTE --
        imbalance_ratio = y.value_counts().max() / max(y.value_counts().min(), 1)
        if HAS_SMOTE and imbalance_ratio > 1.5 and len(X) > 100:
            try:
                k = min(5, y.value_counts().min() - 1)
                if k >= 1:
                    smote = SMOTE(random_state=42, k_neighbors=k)
                    X_balanced, y_balanced = smote.fit_resample(X, y)
                    logger.info(f"SMOTE applied: {len(X)} -> {len(X_balanced)} samples")
                    X, y = X_balanced, pd.Series(y_balanced)
            except Exception as e:
                logger.warning(f"SMOTE failed, using original: {e}")

        # -- Train/Test Split (stratified) --
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # -- Optional: Hyperparameter Tuning --
        if tune_hyperparameters:
            logger.info("Running GridSearchCV...")
            self.lr_pipeline, self.dt_model = self._tune_hyperparameters(
                X_train, y_train
            )

        # -- Train Both Models --
        logger.info("Training Logistic Regression...")
        self.lr_pipeline.fit(X_train, y_train)

        logger.info("Training Decision Tree...")
        self.dt_model.fit(X_train, y_train)

        self.is_trained = True

        # -- Evaluate on Test Set --
        lr_metrics = self._evaluate(self.lr_pipeline, X_test, y_test, "LR")
        dt_metrics = self._evaluate(self.dt_model, X_test, y_test, "DT")

        # Backward-compat: populate lr_accuracy and dt_accuracy
        self.lr_accuracy = round(lr_metrics['accuracy'] * 100, 2)
        self.dt_accuracy = round(dt_metrics['accuracy'] * 100, 2)

        # -- Cross Validation (5-fold) --
        cv_scores = {}
        if use_cross_validation and len(X) >= 50:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            lr_cv = cross_val_score(self.lr_pipeline, X, y, cv=cv, scoring='f1')
            dt_cv = cross_val_score(self.dt_model, X, y, cv=cv, scoring='f1')
            cv_scores = {
                'lr_cv_f1_mean': round(float(lr_cv.mean()), 4),
                'lr_cv_f1_std': round(float(lr_cv.std()), 4),
                'dt_cv_f1_mean': round(float(dt_cv.mean()), 4),
                'dt_cv_f1_std': round(float(dt_cv.std()), 4),
            }
            logger.info(
                f"Cross-validation F1 | "
                f"LR: {lr_cv.mean():.3f} +/-{lr_cv.std():.3f} | "
                f"DT: {dt_cv.mean():.3f} +/-{dt_cv.std():.3f}"
            )

        # -- Store All Metrics --
        self.metrics = {
            'lr': lr_metrics,
            'dt': dt_metrics,
            'cross_validation': cv_scores,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'dataset_source': dataset_source,
            'smote_applied': imbalance_ratio > 1.5 and HAS_SMOTE,
            'features': self.FEATURE_NAMES,
        }

        self._log_training_results()
        self.save_models()

        return self.metrics

    def _evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series, label: str) -> dict:
        """Compute full evaluation metrics for one model."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'f1': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'roc_auc': round(roc_auc_score(y_test, y_proba), 4),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }
        return metrics

    def _tune_hyperparameters(self, X_train, y_train):
        """Run GridSearchCV to find best hyperparameters."""
        # LR hyperparameter grid
        lr_param_grid = {
            'classifier__C': [0.1, 0.5, 1.0, 2.0, 5.0],
            'classifier__solver': ['lbfgs', 'liblinear']
        }
        lr_grid = GridSearchCV(
            self.lr_pipeline, lr_param_grid,
            cv=3, scoring='f1', n_jobs=-1, verbose=0
        )
        lr_grid.fit(X_train, y_train)
        best_lr = lr_grid.best_estimator_
        logger.info(f"Best LR params: {lr_grid.best_params_}")

        # DT hyperparameter grid
        dt_param_grid = {
            'max_depth': [3, 4, 5, 6, 7],
            'min_samples_split': [5, 10, 15, 20],
            'min_samples_leaf': [2, 4, 6]
        }
        dt_grid = GridSearchCV(
            self.dt_model, dt_param_grid,
            cv=3, scoring='f1', n_jobs=-1, verbose=0
        )
        dt_grid.fit(X_train, y_train)
        best_dt = dt_grid.best_estimator_
        logger.info(f"Best DT params: {dt_grid.best_params_}")

        return best_lr, best_dt

    # -----------------------------------------------------------------
    #  Make Predictions
    # -----------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> Dict:
        """
        Run both models and return probabilities + predictions.
        Ensemble = weighted average of LR and DT probabilities.

        Backward compatible: still returns lr_predictions,
        dt_predictions, lr_probabilities keys.
        """
        if not self.is_trained:
            logger.warning("[MLModel] Models not trained yet!")
            return {
                "lr_predictions": [],
                "dt_predictions": [],
                "lr_probabilities": [],
                "ensemble_predictions": [],
                "ensemble_probabilities": [],
            }

        # Ensure correct feature order, handle missing columns gracefully
        for col in self.FEATURE_NAMES:
            if col not in X.columns:
                X[col] = 0
        X = X[self.FEATURE_NAMES]

        lr_proba = self.lr_pipeline.predict_proba(X)[:, 1]
        dt_proba = self.dt_model.predict_proba(X)[:, 1]

        # Ensemble: weighted average (LR slightly higher weight)
        ensemble_proba = (0.55 * lr_proba) + (0.45 * dt_proba)

        threshold = 0.5

        # Round probabilities for cleaner output
        lr_probs_rounded = [round(float(p), 4) for p in lr_proba]
        dt_probs_rounded = [round(float(p), 4) for p in dt_proba]
        ens_probs_rounded = [round(float(p), 4) for p in ensemble_proba]

        logger.debug(f"[MLModel] Generated predictions for {len(X)} skills.")

        return {
            'lr_predictions': (lr_proba >= threshold).astype(int).tolist(),
            'lr_probabilities': lr_probs_rounded,
            'dt_predictions': (dt_proba >= threshold).astype(int).tolist(),
            'dt_probabilities': dt_probs_rounded,
            'ensemble_predictions': (ensemble_proba >= threshold).astype(int).tolist(),
            'ensemble_probabilities': ens_probs_rounded,
        }

    # -----------------------------------------------------------------
    #  Feature Importance
    # -----------------------------------------------------------------
    def get_feature_importance(self) -> Dict:
        """
        Feature importance from DT and coefficient magnitude from LR.
        """
        if not self.is_trained:
            return {}

        dt_imp = dict(zip(
            self.FEATURE_NAMES,
            [round(float(v), 4) for v in self.dt_model.feature_importances_]
        ))

        lr_coefs = dict(zip(
            self.FEATURE_NAMES,
            [round(float(v), 4) for v in abs(self.lr_pipeline.named_steps['classifier'].coef_[0])]
        ))

        return {
            'dt_importance': dt_imp,
            'lr_coefficients': lr_coefs,
            'explanation': (
                "DT importance: how often each feature is used "
                "to split the tree. LR coefficients: how much "
                "each feature moves the probability."
            )
        }

    # -----------------------------------------------------------------
    #  SHAP Explainability
    # -----------------------------------------------------------------
    def get_shap_values(self, X: pd.DataFrame) -> dict:
        """
        SHAP values for Decision Tree predictions.
        Explains WHY the model made each prediction.
        """
        if not self.is_trained or not HAS_SHAP:
            return {'shap_available': False, 'reason': 'SHAP not available'}

        try:
            for col in self.FEATURE_NAMES:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.FEATURE_NAMES]

            explainer = shap.TreeExplainer(self.dt_model)
            shap_vals = explainer.shap_values(X)

            # Multi-class DT returns list — take class 1 (positive)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            per_feature_mean = {
                feat: round(float(abs(shap_vals[:, i]).mean()), 4)
                for i, feat in enumerate(self.FEATURE_NAMES)
            }

            base_val = explainer.expected_value
            if isinstance(base_val, list):
                base_val = base_val[1]

            return {
                'shap_available': True,
                'shap_values': shap_vals.tolist(),
                'base_value': float(base_val),
                'feature_names': self.FEATURE_NAMES,
                'per_feature_mean': per_feature_mean,
                'explanation': (
                    "SHAP values show each feature's contribution "
                    "to the prediction. Positive = pushes toward "
                    "'skill present', negative = pushes toward gap."
                )
            }
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return {'shap_available': False, 'reason': str(e)}

    # -----------------------------------------------------------------
    #  Model Summary (Backward Compatible)
    # -----------------------------------------------------------------
    def get_model_summary(self) -> Dict:
        """
        Generate a human-readable summary of both models.
        Maintains backward compatibility with report_generator.py
        which expects: models_used, lr_accuracy, dt_accuracy,
        feature_importance, lr_explanation, dt_explanation,
        training_explanation.
        """
        return {
            "models_used": ["Logistic Regression", "Decision Tree"],
            "is_trained": self.is_trained,
            "lr_accuracy": self.lr_accuracy,
            "dt_accuracy": self.dt_accuracy,
            "dataset_source": self.dataset_source,
            "feature_importance": self.get_feature_importance(),
            "lr_explanation": (
                "Logistic Regression calculates the probability "
                "a skill is genuinely present (0% to 100%) based "
                "on resume claims and GitHub evidence. It outputs "
                "a calibrated confidence score."
            ),
            "dt_explanation": (
                "Decision Tree builds IF/THEN rules from data. "
                "Example: IF skill in GitHub AND resume THEN "
                "confidence=95%. Fully explainable — you can "
                "trace exactly why each decision was made."
            ),
            "ensemble_explanation": (
                "Final score = 55% LR + 45% DT. Combining both "
                "models reduces individual errors and gives more "
                "reliable predictions."
            ),
            "training_explanation": (
                f"Both models were trained on data from {self.dataset_source}. "
                f"Features used: {', '.join(self.FEATURE_NAMES)}. "
                "class_weight='balanced' handles imbalanced datasets. "
                "Cross-validation ensures honest accuracy reporting."
            ),
            "metrics": self.metrics,
            "feature_names": self.FEATURE_NAMES,
        }

    # -----------------------------------------------------------------
    #  Logging
    # -----------------------------------------------------------------
    def _log_training_results(self):
        """Log a clean training summary to console."""
        lr = self.metrics.get('lr', {})
        dt = self.metrics.get('dt', {})
        cv = self.metrics.get('cross_validation', {})

        logger.info("=" * 50)
        logger.info("       ML MODEL TRAINING COMPLETE")
        logger.info("=" * 50)
        logger.info(
            f"  Logistic Regression | "
            f"Acc: {lr.get('accuracy', 0):.1%} | "
            f"F1: {lr.get('f1', 0):.3f} | "
            f"AUC: {lr.get('roc_auc', 0):.3f}"
        )
        logger.info(
            f"  Decision Tree       | "
            f"Acc: {dt.get('accuracy', 0):.1%} | "
            f"F1: {dt.get('f1', 0):.3f} | "
            f"AUC: {dt.get('roc_auc', 0):.3f}"
        )
        if cv:
            logger.info(
                f"  Cross-Val (5-fold)  | "
                f"LR F1: {cv.get('lr_cv_f1_mean', 0):.3f} "
                f"+/-{cv.get('lr_cv_f1_std', 0):.3f} | "
                f"DT F1: {cv.get('dt_cv_f1_mean', 0):.3f} "
                f"+/-{cv.get('dt_cv_f1_std', 0):.3f}"
            )
        logger.info(
            f"  Dataset: {self.dataset_source} | "
            f"Samples: {self.metrics.get('train_samples', 0)} train "
            f"/ {self.metrics.get('test_samples', 0)} test"
        )
        logger.info("=" * 50)

    # -----------------------------------------------------------------
    #  Model Persistence
    # -----------------------------------------------------------------
    def save_models(self):
        """Save both models to disk with joblib."""
        try:
            joblib.dump(self.lr_pipeline,
                        self.model_save_path / "lr_model.pkl")
            joblib.dump(self.dt_model,
                        self.model_save_path / "dt_model.pkl")
            # Save metrics as JSON
            with open(self.model_save_path / "metrics.json", "w") as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"Models saved to {self.model_save_path}")
        except Exception as e:
            logger.warning(f"Could not save models: {e}")

    def load_models(self) -> bool:
        """
        Load previously saved models from disk.
        Faster than retraining on every startup.
        Returns True if loaded successfully, False otherwise.
        """
        try:
            lr_path = self.model_save_path / "lr_model.pkl"
            dt_path = self.model_save_path / "dt_model.pkl"
            metrics_path = self.model_save_path / "metrics.json"

            if not (lr_path.exists() and dt_path.exists()):
                logger.info("No saved models found — will train fresh")
                return False

            self.lr_pipeline = joblib.load(lr_path)
            self.dt_model = joblib.load(dt_path)

            if metrics_path.exists():
                with open(metrics_path) as f:
                    self.metrics = json.load(f)
                self.dataset_source = self.metrics.get(
                    'dataset_source', 'loaded from disk'
                )

            # Restore backward-compat accuracy fields
            self.lr_accuracy = round(
                self.metrics.get('lr', {}).get('accuracy', 0) * 100, 2
            )
            self.dt_accuracy = round(
                self.metrics.get('dt', {}).get('accuracy', 0) * 100, 2
            )

            self.is_trained = True
            logger.info(
                f"Loaded saved models | "
                f"LR acc: {self.lr_accuracy}% | "
                f"DT acc: {self.dt_accuracy}%"
            )
            return True

        except Exception as e:
            logger.warning(f"Could not load saved models: {e}")
            return False
