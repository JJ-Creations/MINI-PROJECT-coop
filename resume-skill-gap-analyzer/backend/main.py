"""
=============================================================================
 Resume Skill Gap Analyzer — FastAPI Backend
=============================================================================
 This is the main entry point for the application. It:
   1. Loads configuration and data files on startup
   2. Initializes all pipeline modules
   3. Trains the ML models with synthetic data
   4. Exposes REST API endpoints for the frontend

 Pipeline flow per request:
   Upload Resume + GitHub Username + Target Role
       → Resume Parser (extract skills from text)
       → GitHub Analyzer (fetch demonstrated skills)
       → Feature Engineering (build skill matrix)
       → ML Model (predict skill presence + confidence)
       → Skill Gap Analyzer (compute gaps vs. role requirements)
       → Report Generator (compile final report)
       → JSON Response to Frontend

 Run with: uvicorn main:app --reload --port 8000
=============================================================================
"""

import json
import os
from contextlib import asynccontextmanager
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import all pipeline modules
from modules.resume_parser import ResumeParser
from modules.github_analyzer import GitHubAnalyzer
from modules.feature_engineering import FeatureEngineer
from modules.ml_model import SkillGapMLModel
from modules.skill_gap_analyzer import SkillGapAnalyzer
from modules.report_generator import ReportGenerator

# ---------------------------------------------------------------------------
#  Load Environment Variables
# ---------------------------------------------------------------------------
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# ---------------------------------------------------------------------------
#  Global State (loaded on startup)
# ---------------------------------------------------------------------------
# These are populated in the lifespan event and used by all endpoints
job_roles_data: Dict = {}
skills_master: Dict[str, List[str]] = {}
resume_parser: ResumeParser = None      # type: ignore
github_analyzer: GitHubAnalyzer = None  # type: ignore
feature_engineer: FeatureEngineer = None  # type: ignore
ml_model: SkillGapMLModel = None        # type: ignore
skill_gap_analyzer: SkillGapAnalyzer = None  # type: ignore
report_generator: ReportGenerator = None    # type: ignore


# ---------------------------------------------------------------------------
#  Application Lifespan — Startup & Shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler — runs on startup and shutdown.

    On startup:
      1. Load data files (job_roles.json, skills_master.json)
      2. Initialize all pipeline module classes
      3. Train ML models with synthetic data
      4. Print readiness status
    """
    global job_roles_data, skills_master
    global resume_parser, github_analyzer, feature_engineer
    global ml_model, skill_gap_analyzer, report_generator

    print("\n" + "=" * 60)
    print("  RESUME SKILL GAP ANALYZER — Starting Up")
    print("=" * 60)

    # --- Load data files ---
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    with open(os.path.join(data_dir, "job_roles.json"), "r") as f:
        job_roles_data = json.load(f)
    print(f"\n  Loaded {len(job_roles_data)} job roles.")

    with open(os.path.join(data_dir, "skills_master.json"), "r") as f:
        skills_master = json.load(f)
    total_skills = sum(len(v) for v in skills_master.values())
    print(f"  Loaded {total_skills} skills across {len(skills_master)} categories.")

    # --- Initialize pipeline modules ---
    print("\n  Initializing pipeline modules...")
    resume_parser = ResumeParser()
    github_analyzer = GitHubAnalyzer(github_token=GITHUB_TOKEN if GITHUB_TOKEN else None)
    feature_engineer = FeatureEngineer()
    ml_model = SkillGapMLModel()
    skill_gap_analyzer = SkillGapAnalyzer()
    report_generator = ReportGenerator()

    # --- Train ML models with synthetic data ---
    print("\n  Training ML models with synthetic data...")
    X_train, y_train = ml_model.generate_training_data(skills_master)
    accuracies = ml_model.train(X_train, y_train)

    # --- Print readiness status ---
    print("\n" + "=" * 60)
    print("  Resume Skill Gap Analyzer API is running")
    print(f"  Loaded {len(job_roles_data)} job roles, {total_skills} skills in master list")
    print(f"  ML Models trained — LR Accuracy: {accuracies['lr_accuracy']}%, DT Accuracy: {accuracies['dt_accuracy']}%")
    print("=" * 60 + "\n")

    yield  # Application runs here

    # Shutdown
    print("\n  Shutting down Resume Skill Gap Analyzer API.")


# ---------------------------------------------------------------------------
#  Create FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Resume Skill Gap Analyzer",
    description="Analyzes resumes and GitHub profiles to identify skill gaps for target job roles.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS Middleware ---
# Allow all origins so the frontend (opened as a local file) can make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
#  Pydantic Models for Request/Response Validation
# ---------------------------------------------------------------------------
class TextAnalyzeRequest(BaseModel):
    """Request body for the /analyze-text endpoint."""
    resume_text: str
    github_username: str
    target_role: str


# ---------------------------------------------------------------------------
#  ENDPOINT: Health Check
# ---------------------------------------------------------------------------
@app.get("/")
async def health_check():
    """
    Health check endpoint — confirms the API is running.
    Returns the API status and a welcome message.
    """
    return {
        "status": "running",
        "message": "Resume Skill Gap Analyzer API",
        "available_roles": list(job_roles_data.keys()),
    }


# ---------------------------------------------------------------------------
#  ENDPOINT: Get Available Job Roles
# ---------------------------------------------------------------------------
@app.get("/job-roles")
async def get_job_roles():
    """
    Return the list of available job roles and their requirements.
    The frontend uses this to populate the role selection dropdown.
    """
    roles = []
    for role_name, role_data in job_roles_data.items():
        roles.append({
            "name": role_name,
            "required_skills": role_data["required_skills"],
            "nice_to_have": role_data.get("nice_to_have", []),
        })
    return {"job_roles": roles}


# ---------------------------------------------------------------------------
#  ENDPOINT: Get Skills Master List
# ---------------------------------------------------------------------------
@app.get("/skills-master")
async def get_skills_master():
    """
    Return the full master skills list organized by category.
    Useful for frontend autocomplete or skill browsing.
    """
    return {"skills_master": skills_master}


# ---------------------------------------------------------------------------
#  ENDPOINT: Analyze (File Upload)
# ---------------------------------------------------------------------------
@app.post("/analyze")
async def analyze(
    resume_file: UploadFile = File(...),
    github_username: str = Form(...),
    target_role: str = Form(...),
):
    """
    Main analysis endpoint — accepts a resume file upload.

    Pipeline:
      1. Validate inputs (file type, role existence)
      2. Parse resume → extract skills
      3. Analyze GitHub profile → get demonstrated skills
      4. Build feature matrix for ML
      5. Run ML predictions
      6. Compute skill gap analysis
      7. Generate final report

    Args:
        resume_file:     Uploaded resume file (.pdf or .txt)
        github_username: GitHub username to analyze
        target_role:     Target job role to compare against

    Returns:
        Complete analysis report as JSON
    """
    print(f"\n{'#'*60}")
    print(f"  NEW ANALYSIS REQUEST")
    print(f"  File: {resume_file.filename}")
    print(f"  GitHub: {github_username}")
    print(f"  Role: {target_role}")
    print(f"{'#'*60}")

    # --- Validation ---
    # Check file extension
    filename = resume_file.filename or ""
    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a .pdf or .txt file.",
        )

    # Check that the target role exists
    if target_role not in job_roles_data:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown role: '{target_role}'. Available roles: {list(job_roles_data.keys())}",
        )

    try:
        # --- Step 1: Parse Resume ---
        print("\n  Step 1/6: Parsing resume...")
        file_bytes = await resume_file.read()
        resume_result = resume_parser.parse(file_bytes, filename, skills_master)
        claimed_skills = resume_result["extracted_skills"]

        # --- Step 2: Analyze GitHub ---
        print("\n  Step 2/6: Analyzing GitHub profile...")
        github_result = github_analyzer.analyze_github_profile(github_username, skills_master)
        demonstrated_skills = github_result["demonstrated_skills"]

        # --- Step 3: Feature Engineering ---
        print("\n  Step 3/6: Building feature matrix...")
        role_data = job_roles_data[target_role]
        skill_matrix = feature_engineer.create_skill_matrix(
            claimed_skills,
            demonstrated_skills,
            role_data["required_skills"],
            role_data.get("nice_to_have", []),
        )
        X, y = feature_engineer.encode_for_model(skill_matrix)

        # --- Step 4: ML Predictions ---
        print("\n  Step 4/6: Running ML predictions...")
        predictions = ml_model.predict(X)
        lr_probabilities = predictions["lr_probabilities"]

        # --- Step 5: Skill Gap Analysis ---
        print("\n  Step 5/6: Computing skill gaps...")
        analysis = skill_gap_analyzer.analyze(
            claimed_skills=claimed_skills,
            demonstrated_skills=demonstrated_skills,
            target_role=target_role,
            job_roles_data=job_roles_data,
            ml_predictions=predictions,
            lr_probabilities=lr_probabilities,
            skill_matrix=skill_matrix,
        )

        # --- Step 6: Generate Report ---
        print("\n  Step 6/6: Generating report...")
        report = report_generator.generate_report(
            analysis_result=analysis,
            target_role=target_role,
            github_username=github_username,
            resume_skills=claimed_skills,
            github_skills=demonstrated_skills,
            model_summary=ml_model.get_model_summary(),
            github_insights_data=github_result,
        )

        print(f"\n  Analysis complete! Match Score: {analysis['match_score']}%")
        return report

    except Exception as e:
        print(f"\n  ERROR during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ---------------------------------------------------------------------------
#  ENDPOINT: Analyze Text (No File Upload)
# ---------------------------------------------------------------------------
@app.post("/analyze-text")
async def analyze_text(request: TextAnalyzeRequest):
    """
    Alternative analysis endpoint — accepts raw resume text instead of a file.
    Useful for testing and for frontends that handle file reading client-side.

    Same pipeline as /analyze but skips file parsing — goes straight to
    skill extraction on the provided text.
    """
    print(f"\n{'#'*60}")
    print(f"  NEW TEXT ANALYSIS REQUEST")
    print(f"  GitHub: {request.github_username}")
    print(f"  Role: {request.target_role}")
    print(f"{'#'*60}")

    # --- Validation ---
    if request.target_role not in job_roles_data:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown role: '{request.target_role}'. Available: {list(job_roles_data.keys())}",
        )

    if not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="Resume text cannot be empty.")

    try:
        # --- Step 1: Extract skills from raw text ---
        print("\n  Step 1/6: Extracting skills from text...")
        claimed_skills = resume_parser.extract_skills(request.resume_text, skills_master)

        # --- Step 2: Analyze GitHub ---
        print("\n  Step 2/6: Analyzing GitHub profile...")
        github_result = github_analyzer.analyze_github_profile(
            request.github_username, skills_master
        )
        demonstrated_skills = github_result["demonstrated_skills"]

        # --- Step 3: Feature Engineering ---
        print("\n  Step 3/6: Building feature matrix...")
        role_data = job_roles_data[request.target_role]
        skill_matrix = feature_engineer.create_skill_matrix(
            claimed_skills,
            demonstrated_skills,
            role_data["required_skills"],
            role_data.get("nice_to_have", []),
        )
        X, y = feature_engineer.encode_for_model(skill_matrix)

        # --- Step 4: ML Predictions ---
        print("\n  Step 4/6: Running ML predictions...")
        predictions = ml_model.predict(X)
        lr_probabilities = predictions["lr_probabilities"]

        # --- Step 5: Skill Gap Analysis ---
        print("\n  Step 5/6: Computing skill gaps...")
        analysis = skill_gap_analyzer.analyze(
            claimed_skills=claimed_skills,
            demonstrated_skills=demonstrated_skills,
            target_role=request.target_role,
            job_roles_data=job_roles_data,
            ml_predictions=predictions,
            lr_probabilities=lr_probabilities,
            skill_matrix=skill_matrix,
        )

        # --- Step 6: Generate Report ---
        print("\n  Step 6/6: Generating report...")
        report = report_generator.generate_report(
            analysis_result=analysis,
            target_role=request.target_role,
            github_username=request.github_username,
            resume_skills=claimed_skills,
            github_skills=demonstrated_skills,
            model_summary=ml_model.get_model_summary(),
            github_insights_data=github_result,
        )

        print(f"\n  Analysis complete! Match Score: {analysis['match_score']}%")
        return report

    except Exception as e:
        print(f"\n  ERROR during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ---------------------------------------------------------------------------
#  Run with Uvicorn (if executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
