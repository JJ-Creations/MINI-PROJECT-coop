"""
=============================================================================
 Resume Skill Gap Analyzer — Modules Package
=============================================================================
 This package contains all the core modules for the analysis pipeline:

   1. resume_parser      — Extracts text and skills from uploaded resumes
   2. github_analyzer     — Fetches and analyzes a user's GitHub profile
   3. feature_engineering — Builds feature vectors for the ML models
   4. ml_model            — Trains and runs Logistic Regression & Decision Tree
   5. skill_gap_analyzer  — Computes skill gaps against target job roles
   6. report_generator    — Compiles the final analysis report
=============================================================================
"""

from .resume_parser import ResumeParser
from .github_analyzer import GitHubAnalyzer
from .feature_engineering import FeatureEngineer
from .ml_model import SkillGapMLModel
from .skill_gap_analyzer import SkillGapAnalyzer
from .report_generator import ReportGenerator

__all__ = [
    "ResumeParser",
    "GitHubAnalyzer",
    "FeatureEngineer",
    "SkillGapMLModel",
    "SkillGapAnalyzer",
    "ReportGenerator",
]
