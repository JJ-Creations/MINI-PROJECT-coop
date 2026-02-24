"""
=============================================================================
 Report Generator Module
=============================================================================
 Role in the pipeline:
   This is the SIXTH and final stage. It takes all the analysis results
   from previous stages and compiles them into a comprehensive, structured
   report that the frontend can render.

 The report includes:
   - Executive summary with key metrics
   - Detailed skill breakdown with status badges
   - Actionable recommendations for missing skills
   - ML model insights and accuracy metrics
   - GitHub profile insights
   - A suggested learning path
=============================================================================
"""

from typing import Dict, List


class ReportGenerator:
    """Compiles analysis results into a structured, presentation-ready report."""

    def __init__(self) -> None:
        """Initialize the report generator."""
        print("   [ReportGenerator] Initialized.")

    # -----------------------------------------------------------------
    #  Score Label Helper
    # -----------------------------------------------------------------
    @staticmethod
    def _get_score_label(score: float) -> str:
        """
        Convert a numeric match score to a human-readable label.

        Thresholds:
          75-100 → Excellent (strong candidate)
          50-74  → Good (some gaps but viable)
          25-49  → Fair (significant gaps)
          0-24   → Poor (major skills missing)

        Args:
            score: The match score (0-100).

        Returns:
            A label string.
        """
        if score >= 75:
            return "Excellent"
        elif score >= 50:
            return "Good"
        elif score >= 25:
            return "Fair"
        else:
            return "Poor"

    # -----------------------------------------------------------------
    #  Confidence Rating Helper
    # -----------------------------------------------------------------
    @staticmethod
    def _get_confidence_rating(confidence: float) -> str:
        """
        Convert a confidence score to a qualitative rating.

        Args:
            confidence: The confidence percentage (0-100).

        Returns:
            A rating string.
        """
        if confidence >= 80:
            return "High — ML model is very confident in skill assessments"
        elif confidence >= 60:
            return "Medium — reasonable confidence, some skills need verification"
        elif confidence >= 40:
            return "Low — limited evidence for many skills"
        else:
            return "Very Low — insufficient data for reliable assessment"

    # -----------------------------------------------------------------
    #  Resource Hint Helper
    # -----------------------------------------------------------------
    @staticmethod
    def _get_resource_hint(skill: str) -> str:
        """
        Suggest what type of learning resource would help for a given skill.

        This is a simple rule-based mapping — in production, this could
        be connected to an actual course recommendation API.

        Args:
            skill: The skill name.

        Returns:
            A resource suggestion string.
        """
        # Map skills to resource types based on their nature
        programming_langs = {"Python", "JavaScript", "Java", "C++", "TypeScript", "Go",
                             "Rust", "R", "Kotlin", "Swift", "Scala", "C#", "PHP", "Ruby", "SQL"}
        frameworks = {"React", "Angular", "Vue", "Django", "Flask", "FastAPI", "Spring",
                      "Node.js", "Express", "TensorFlow", "PyTorch", "Keras", "Scikit-learn",
                      "Pandas", "NumPy", "Next.js"}
        tools = {"Docker", "Kubernetes", "Jenkins", "Terraform", "Ansible", "Git",
                 "Webpack", "Maven", "MLflow"}
        cloud = {"AWS", "Azure", "GCP", "Heroku", "Firebase"}
        concepts = {"Machine Learning", "Deep Learning", "NLP", "Computer Vision",
                    "Data Science", "Statistics", "Microservices", "CI/CD", "REST API",
                    "GraphQL", "DevOps", "Agile"}

        if skill in programming_langs:
            return f"Online course + practice problems (e.g., LeetCode in {skill})"
        elif skill in frameworks:
            return f"Official {skill} documentation + build a small project"
        elif skill in tools:
            return f"Hands-on tutorial + set up {skill} in a personal project"
        elif skill in cloud:
            return f"{skill} free tier + certification study path"
        elif skill in concepts:
            return f"Structured course on {skill} + implement a portfolio project"
        else:
            return f"Self-study + practice project using {skill}"

    # -----------------------------------------------------------------
    #  Generate Full Report
    # -----------------------------------------------------------------
    def generate_report(
        self,
        analysis_result: Dict,
        target_role: str,
        github_username: str,
        resume_skills: List[str],
        github_skills: List[str],
        model_summary: Dict,
        github_insights_data: Dict,
    ) -> Dict:
        """
        Compile all analysis results into a comprehensive report.

        This is the main output of the entire pipeline — everything the
        frontend needs to display a full analysis dashboard.

        Args:
            analysis_result:     Output from SkillGapAnalyzer.analyze()
            target_role:         The target job role name
            github_username:     The analyzed GitHub username
            resume_skills:       List of skills from the resume
            github_skills:       List of skills from GitHub
            model_summary:       Output from SkillGapMLModel.get_model_summary()
            github_insights_data: Raw GitHub analysis data (languages, repos, etc.)

        Returns:
            A fully structured report dict ready for JSON serialization.
        """
        print(f"\n{'='*60}")
        print(f"   [ReportGenerator] Generating report for: {target_role}")
        print(f"{'='*60}")

        match_score = analysis_result["match_score"]
        confidence = analysis_result["confidence"]

        # --- Executive Summary ---
        executive_summary = {
            "match_score": match_score,
            "match_label": self._get_score_label(match_score),
            "total_resume_skills": len(resume_skills),
            "total_github_skills": len(github_skills),
            "missing_critical_skills": len(analysis_result["missing_required"]),
            "confidence_rating": self._get_confidence_rating(confidence),
            "confidence_score": confidence,
        }

        # --- Recommendations ---
        # Generate actionable recommendations for each missing required skill
        recommendations = []
        for skill in analysis_result["missing_required"]:
            recommendations.append({
                "skill": skill,
                "priority": "Critical",
                "action": f"Learn {skill} — required for {target_role}",
                "resource_hint": self._get_resource_hint(skill),
            })

        # Also add recommendations for missing nice-to-have skills (lower priority)
        for skill in analysis_result["missing_nice_to_have"]:
            recommendations.append({
                "skill": skill,
                "priority": "Recommended",
                "action": f"Consider learning {skill} — nice to have for {target_role}",
                "resource_hint": self._get_resource_hint(skill),
            })

        # --- ML Insights ---
        ml_insights = {
            "models_used": model_summary.get("models_used", []),
            "lr_accuracy": model_summary.get("lr_accuracy", 0),
            "dt_accuracy": model_summary.get("dt_accuracy", 0),
            "feature_importance": model_summary.get("feature_importance", {}),
            "model_explanation": model_summary.get("training_explanation", ""),
            "lr_explanation": model_summary.get("lr_explanation", ""),
            "dt_explanation": model_summary.get("dt_explanation", ""),
        }

        # --- GitHub Insights ---
        # Sort languages by byte count to find the top 5
        raw_languages = github_insights_data.get("raw_languages", {})
        sorted_languages = sorted(raw_languages.items(), key=lambda x: x[1], reverse=True)
        top_languages = [{"language": lang, "bytes": bytes_count} for lang, bytes_count in sorted_languages[:5]]

        github_insights = {
            "repos_analyzed": github_insights_data.get("repos_analyzed", 0),
            "top_languages": top_languages,
            "hidden_strengths": analysis_result.get("hidden_strengths", []),
            "total_topics": len(github_insights_data.get("raw_topics", [])),
            "error": github_insights_data.get("error", ""),
        }

        # --- Compile the Full Report ---
        report = {
            "title": "Skill Gap Analysis Report",
            "target_role": target_role,
            "github_username": github_username,
            "executive_summary": executive_summary,
            "skill_breakdown": {
                "match_score": analysis_result["match_score"],
                "gap_score": analysis_result["gap_score"],
                "confidence": analysis_result["confidence"],
                "required_analysis": analysis_result["required_analysis"],
                "nice_to_have_analysis": analysis_result["nice_to_have_analysis"],
                "missing_required": analysis_result["missing_required"],
                "missing_nice_to_have": analysis_result["missing_nice_to_have"],
                "strengths": analysis_result["strengths"],
                "claims_not_proven": analysis_result["claims_not_proven"],
                "hidden_strengths": analysis_result["hidden_strengths"],
            },
            "recommendations": recommendations,
            "learning_path": self.generate_learning_path(
                analysis_result["missing_required"] + analysis_result["missing_nice_to_have"]
            ),
            "ml_insights": ml_insights,
            "github_insights": github_insights,
            "resume_skills": resume_skills,
            "github_skills": github_skills,
        }

        print(f"   [ReportGenerator] Report generated successfully.")
        print(f"   [ReportGenerator] Match: {match_score}% ({executive_summary['match_label']})")
        return report

    # -----------------------------------------------------------------
    #  Generate Learning Path
    # -----------------------------------------------------------------
    def generate_learning_path(self, missing_skills: List[str]) -> List[Dict]:
        """
        Create a prioritized learning path for the candidate's missing skills.

        Skills are ordered: critical (required) first, then recommended
        (nice-to-have). Each skill gets a suggested learning approach.

        Args:
            missing_skills: List of skill names the candidate is missing.
                            Expected to have required skills first, then nice-to-have.

        Returns:
            A list of dicts, each containing:
              - skill:          The skill name
              - priority:       "Critical" or "Recommended"
              - suggested_path: What to study / build
        """
        learning_path = []

        for i, skill in enumerate(missing_skills):
            # First half are typically required (Critical), rest are nice-to-have
            priority = "Critical" if i < len(missing_skills) // 2 + 1 else "Recommended"

            learning_path.append({
                "skill": skill,
                "priority": priority,
                "suggested_path": self._get_resource_hint(skill),
            })

        print(f"   [ReportGenerator] Learning path with {len(learning_path)} items.")
        return learning_path
