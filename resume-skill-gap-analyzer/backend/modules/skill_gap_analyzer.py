"""
=============================================================================
 Skill Gap Analyzer Module
=============================================================================
 Role in the pipeline:
   This is the FIFTH stage. It takes the parsed resume skills, GitHub
   demonstrated skills, ML predictions, and the target job role's
   requirements — then computes a comprehensive gap analysis.

 Key metrics computed:
   - match_score:   Percentage of required skills the candidate has (0-100)
   - gap_score:     Percentage of required skills missing (0-100)
   - confidence:    ML model's average confidence in the candidate's skills
   - Per-skill status: strong / claimed_only / demonstrated_only / missing
=============================================================================
"""

from typing import Dict, List

import pandas as pd


class SkillGapAnalyzer:
    """Computes skill gaps between a candidate's profile and a target job role."""

    def __init__(self) -> None:
        """Initialize the skill gap analyzer."""
        print("   [SkillGapAnalyzer] Initialized.")

    # -----------------------------------------------------------------
    #  Main Analysis Method
    # -----------------------------------------------------------------
    def analyze(
        self,
        claimed_skills: List[str],
        demonstrated_skills: List[str],
        target_role: str,
        job_roles_data: Dict,
        ml_predictions: Dict[str, list],
        lr_probabilities: List[float],
        skill_matrix: pd.DataFrame,
    ) -> Dict:
        """
        Perform a full skill gap analysis for a candidate against a target role.

        Business logic for skill status classification:
          - "strong":            Skill found in BOTH resume AND GitHub
                                 → Strongest evidence — candidate has coded it AND claims it
          - "claimed_only":      Skill found in resume but NOT on GitHub
                                 → Candidate claims it but hasn't demonstrated it publicly
          - "demonstrated_only": Skill found on GitHub but NOT in resume
                                 → Hidden strength — candidate uses it but didn't highlight it
          - "missing":           Skill found in NEITHER source
                                 → This is a gap that needs to be addressed

        Scoring:
          - match_score = (present_required / total_required) * 100
          - gap_score   = 100 - match_score
          - confidence  = average ML probability for present required skills

        Args:
            claimed_skills:      Skills extracted from the resume.
            demonstrated_skills: Skills found on GitHub.
            target_role:         The job role to compare against.
            job_roles_data:      The full job roles dictionary.
            ml_predictions:      Dict with lr_predictions, dt_predictions keys.
            lr_probabilities:    List of LR probability scores per skill.
            skill_matrix:        The skill matrix DataFrame from feature engineering.

        Returns:
            A comprehensive analysis dict (see structure below).
        """
        print(f"\n{'='*60}")
        print(f"   [SkillGapAnalyzer] Analyzing gaps for role: {target_role}")
        print(f"{'='*60}")

        # --- Step 1: Get the role's requirements ---
        role_data = job_roles_data.get(target_role, {})
        required_skills = role_data.get("required_skills", [])
        nice_to_have = role_data.get("nice_to_have", [])

        # --- Step 2: Build the combined skill set ---
        # Union of everything the candidate has shown in any source
        all_combined = set(claimed_skills) | set(demonstrated_skills)

        # --- Step 3: Classify each required skill ---
        required_analysis = []
        missing_required = []
        strengths = []
        claims_not_proven = []
        hidden_strengths = []

        for i, skill in enumerate(required_skills):
            in_resume = skill in claimed_skills
            in_github = skill in demonstrated_skills

            # Determine skill status based on evidence from both sources
            if in_resume and in_github:
                status = "strong"
                strengths.append(skill)
            elif in_resume and not in_github:
                status = "claimed_only"
                claims_not_proven.append(skill)
            elif not in_resume and in_github:
                status = "demonstrated_only"
                hidden_strengths.append(skill)
            else:
                status = "missing"
                missing_required.append(skill)

            # Get ML prediction and probability for this skill (if available)
            # The skill_matrix rows align with required + nice_to_have skills
            ml_pred = ml_predictions.get("lr_predictions", [0] * len(required_skills))
            ml_prob = lr_probabilities

            skill_analysis = {
                "skill": skill,
                "status": status,
                "in_resume": in_resume,
                "in_github": in_github,
                "ml_prediction": ml_pred[i] if i < len(ml_pred) else 0,
                "probability": round(ml_prob[i], 4) if i < len(ml_prob) else 0.0,
            }
            required_analysis.append(skill_analysis)

        # --- Step 4: Classify each nice-to-have skill ---
        nice_to_have_analysis = []
        missing_nice_to_have = []

        for skill in nice_to_have:
            in_resume = skill in claimed_skills
            in_github = skill in demonstrated_skills

            if in_resume and in_github:
                status = "strong"
            elif in_resume:
                status = "claimed_only"
            elif in_github:
                status = "demonstrated_only"
            else:
                status = "missing"
                missing_nice_to_have.append(skill)

            nice_to_have_analysis.append({
                "skill": skill,
                "status": status,
                "in_resume": in_resume,
                "in_github": in_github,
            })

        # --- Step 5: Calculate overall scores ---

        # Match score: what percentage of REQUIRED skills does the candidate have?
        total_required = len(required_skills)
        present_required = total_required - len(missing_required)

        if total_required > 0:
            match_score = round((present_required / total_required) * 100, 1)
        else:
            match_score = 0.0

        # Gap score: inverse of match score
        gap_score = round(100 - match_score, 1)

        # Confidence score: average ML probability for skills that ARE present
        # This tells us how confident the ML model is in the candidate's skills
        present_probs = [
            lr_probabilities[i]
            for i in range(min(len(required_skills), len(lr_probabilities)))
            if required_skills[i] in all_combined
        ]

        if present_probs:
            confidence = round(sum(present_probs) / len(present_probs) * 100, 1)
        else:
            confidence = 0.0

        print(f"   [SkillGapAnalyzer] Match Score: {match_score}%")
        print(f"   [SkillGapAnalyzer] Gap Score:   {gap_score}%")
        print(f"   [SkillGapAnalyzer] Confidence:  {confidence}%")
        print(f"   [SkillGapAnalyzer] Missing required: {missing_required}")

        # --- Step 6: Compile and return the full analysis ---
        return {
            "match_score": match_score,
            "gap_score": gap_score,
            "confidence": confidence,
            "required_analysis": required_analysis,
            "nice_to_have_analysis": nice_to_have_analysis,
            "missing_required": missing_required,
            "missing_nice_to_have": missing_nice_to_have,
            "strengths": strengths,
            "claims_not_proven": claims_not_proven,
            "hidden_strengths": hidden_strengths,
        }
