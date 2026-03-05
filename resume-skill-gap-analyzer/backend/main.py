"""
=============================================================================
 Resume Skill Gap Analyzer — FastAPI Backend
=============================================================================
 This is the main entry point for the application. It:
   1. Loads configuration and data files on startup
   2. Initializes all pipeline modules
   3. Trains the ML models (HuggingFace data or synthetic fallback)
   4. Exposes REST API endpoints for the frontend

 Pipeline flow per request:
   Upload Resume + GitHub Username + Target Role
       -> Resume Parser (extract skills from text)
       -> GitHub Analyzer (fetch demonstrated skills)
       -> Feature Engineering (build skill matrix)
       -> ML Model (predict skill presence + confidence)
       -> Skill Gap Analyzer (compute gaps vs. role requirements)
       -> Report Generator (compile final report)
       -> JSON Response to Frontend

 Run with: uvicorn main:app --reload --port 8000
=============================================================================
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

# Import all pipeline modules
from modules.resume_parser import ResumeParser
from modules.github_analyzer import GitHubAnalyzer
from modules.feature_engineering import FeatureEngineer
from modules.ml_model import SkillGapMLModel
from modules.skill_gap_analyzer import SkillGapAnalyzer
from modules.report_generator import ReportGenerator
from data.dataset_loader import DatasetLoader

# ---------------------------------------------------------------------------
#  Load Environment Variables
# ---------------------------------------------------------------------------
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# ---------------------------------------------------------------------------
#  Global State (loaded on startup)
# ---------------------------------------------------------------------------
job_roles_data: Dict = {}
skills_master: Dict[str, List[str]] = {}
resume_parser: ResumeParser = None      # type: ignore
github_analyzer: GitHubAnalyzer = None  # type: ignore
feature_engineer: FeatureEngineer = None  # type: ignore
ml_model: SkillGapMLModel = None        # type: ignore
skill_gap_analyzer: SkillGapAnalyzer = None  # type: ignore
report_generator: ReportGenerator = None    # type: ignore
dataset_loader: DatasetLoader = None    # type: ignore

# Track last retrain time for rate limiting
_last_retrain_time: float = 0.0
# Store last extractor validation results
_extractor_validation: dict = {}


# ---------------------------------------------------------------------------
#  Background Task: Validate Skill Extractor
# ---------------------------------------------------------------------------
async def _validate_extractor_background():
    """Runs after startup, doesn't block the server."""
    global _extractor_validation
    try:
        results = dataset_loader.validate_skill_extractor(
            parser=resume_parser, sample_size=50
        )
        _extractor_validation = results
        logger.info(
            f"Skill extractor validation: "
            f"avg {results.get('avg_skills_per_resume', 0)} skills/resume"
        )
    except Exception as e:
        logger.warning(f"Extractor validation failed: {e}")
        _extractor_validation = {"validated": False, "reason": str(e)}


# ---------------------------------------------------------------------------
#  Startup Banner
# ---------------------------------------------------------------------------
def _print_startup_banner():
    """Print startup banner with real metrics."""
    lr_metrics = ml_model.metrics.get('lr', {})
    dt_metrics = ml_model.metrics.get('dt', {})
    total_skills = sum(len(v) for v in skills_master.values())

    lr_acc = f"{ml_model.lr_accuracy}%"
    lr_f1 = f"{lr_metrics.get('f1', 0):.3f}"
    lr_auc = f"{lr_metrics.get('roc_auc', 0):.3f}"
    dt_acc = f"{ml_model.dt_accuracy}%"
    dt_f1 = f"{dt_metrics.get('f1', 0):.3f}"
    dt_auc = f"{dt_metrics.get('roc_auc', 0):.3f}"
    source = ml_model.dataset_source

    logger.info("")
    logger.info("+" + "=" * 52 + "+")
    logger.info("|       Resume Skill Gap Analyzer  v1.0.0           |")
    logger.info("+" + "=" * 52 + "+")
    logger.info(f"|  Status   : Running                               |")
    logger.info(f"|  Dataset  : {source:<40s}|")
    logger.info(f"|  LR Model : Acc {lr_acc:<6s} F1 {lr_f1:<6s} AUC {lr_auc:<6s}    |")
    logger.info(f"|  DT Model : Acc {dt_acc:<6s} F1 {dt_f1:<6s} AUC {dt_auc:<6s}    |")
    logger.info(f"|  Skills   : {total_skills} skills loaded{' ' * (30 - len(str(total_skills)))}|")
    logger.info(f"|  Roles    : {len(job_roles_data)} roles loaded{' ' * (31 - len(str(len(job_roles_data))))}|")
    logger.info(f"|  Docs     : http://localhost:8000/docs             |")
    logger.info("+" + "=" * 52 + "+")
    logger.info("")


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
      3. Try loading saved models (fast path) or train from scratch
      4. Start background skill extractor validation
      5. Print startup banner
    """
    global job_roles_data, skills_master
    global resume_parser, github_analyzer, feature_engineer
    global ml_model, skill_gap_analyzer, report_generator
    global dataset_loader

    logger.info("RESUME SKILL GAP ANALYZER — Starting Up")

    # --- Load data files ---
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    with open(os.path.join(data_dir, "job_roles.json"), "r") as f:
        job_roles_data = json.load(f)
    logger.info(f"Loaded {len(job_roles_data)} job roles.")

    with open(os.path.join(data_dir, "skills_master.json"), "r") as f:
        skills_master = json.load(f)
    total_skills = sum(len(v) for v in skills_master.values())
    logger.info(f"Loaded {total_skills} skills across {len(skills_master)} categories.")

    # --- Initialize pipeline modules ---
    logger.info("Initializing pipeline modules...")
    resume_parser = ResumeParser()
    github_analyzer = GitHubAnalyzer(github_token=GITHUB_TOKEN if GITHUB_TOKEN else None)
    feature_engineer = FeatureEngineer()
    ml_model = SkillGapMLModel()
    skill_gap_analyzer = SkillGapAnalyzer()
    report_generator = ReportGenerator()
    dataset_loader = DatasetLoader()

    # --- Step 3: Try loading saved models first (faster startup) ---
    models_loaded = ml_model.load_models()

    # --- Step 4: Only retrain if no saved models found ---
    if not models_loaded:
        logger.info("Training ML models from scratch...")

        # Load training data (HuggingFace or synthetic)
        X, y, source = dataset_loader.load_training_data()

        # Train with cross-validation
        ml_model.train(
            X, y,
            dataset_source=source,
            use_cross_validation=True,
            tune_hyperparameters=False
        )
    else:
        logger.info("Using cached models — skipping retraining")

    # --- Step 5: Validate skill extractor in background ---
    asyncio.create_task(_validate_extractor_background())

    # --- Step 6: Print startup banner ---
    _print_startup_banner()

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down Resume Skill Gap Analyzer API.")


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
      2. Parse resume -> extract skills
      3. Analyze GitHub profile -> get demonstrated skills
      4. Build feature matrix for ML
      5. Run ML predictions
      6. Compute skill gap analysis
      7. Generate final report
    """
    logger.info(f"NEW ANALYSIS REQUEST | File: {resume_file.filename} | "
                f"GitHub: {github_username} | Role: {target_role}")

    # --- Validation ---
    filename = resume_file.filename or ""
    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a .pdf or .txt file.",
        )

    if target_role not in job_roles_data:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown role: '{target_role}'. Available roles: {list(job_roles_data.keys())}",
        )

    # --- Step 1: Parse Resume ---
    logger.info("Step 1/6: Parsing resume...")
    try:
        file_bytes = await resume_file.read()
        resume_result = resume_parser.parse(file_bytes, filename, skills_master)
        claimed_skills = resume_result["extracted_skills"]
    except Exception as e:
        raise HTTPException(
            422,
            f"Resume parsing failed: {str(e)}. "
            f"Make sure file is valid PDF or TXT."
        )

    # --- Step 2: Analyze GitHub ---
    logger.info("Step 2/6: Analyzing GitHub profile...")
    try:
        github_result = github_analyzer.analyze_github_profile(github_username, skills_master)
        demonstrated_skills = github_result["demonstrated_skills"]
    except Exception as e:
        raise HTTPException(
            502,
            f"GitHub analysis failed: {str(e)}. "
            f"Check username exists and token is valid."
        )

    # --- Step 3: Feature Engineering ---
    logger.info("Step 3/6: Building feature matrix...")
    try:
        role_data = job_roles_data[target_role]
        skill_matrix = feature_engineer.create_skill_matrix(
            claimed_skills,
            demonstrated_skills,
            role_data["required_skills"],
            role_data.get("nice_to_have", []),
        )
        X, y = feature_engineer.encode_for_model(skill_matrix)
    except Exception as e:
        raise HTTPException(500, f"Feature engineering failed: {str(e)}")

    # --- Step 4: ML Predictions ---
    logger.info("Step 4/6: Running ML predictions...")
    try:
        predictions = ml_model.predict(X)
        lr_probabilities = predictions["lr_probabilities"]
    except Exception as e:
        raise HTTPException(500, f"ML prediction failed: {str(e)}")

    # --- Step 5: Skill Gap Analysis ---
    logger.info("Step 5/6: Computing skill gaps...")
    try:
        analysis = skill_gap_analyzer.analyze(
            claimed_skills=claimed_skills,
            demonstrated_skills=demonstrated_skills,
            target_role=target_role,
            job_roles_data=job_roles_data,
            ml_predictions=predictions,
            lr_probabilities=lr_probabilities,
            skill_matrix=skill_matrix,
        )
    except Exception as e:
        raise HTTPException(500, f"Skill gap analysis failed: {str(e)}")

    # --- Step 6: Generate Report ---
    logger.info("Step 6/6: Generating report...")
    try:
        report = report_generator.generate_report(
            analysis_result=analysis,
            target_role=target_role,
            github_username=github_username,
            resume_skills=claimed_skills,
            github_skills=demonstrated_skills,
            model_summary=ml_model.get_model_summary(),
            github_insights_data=github_result,
        )
    except Exception as e:
        raise HTTPException(500, f"Report generation failed: {str(e)}")

    logger.info(f"Analysis complete! Match Score: {analysis['match_score']}%")
    return report


# ---------------------------------------------------------------------------
#  ENDPOINT: Analyze Text (No File Upload)
# ---------------------------------------------------------------------------
@app.post("/analyze-text")
async def analyze_text(request: TextAnalyzeRequest):
    """
    Alternative analysis endpoint — accepts raw resume text instead of a file.
    Same pipeline as /analyze but skips file parsing.
    """
    logger.info(f"NEW TEXT ANALYSIS REQUEST | "
                f"GitHub: {request.github_username} | Role: {request.target_role}")

    # --- Validation ---
    if request.target_role not in job_roles_data:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown role: '{request.target_role}'. Available: {list(job_roles_data.keys())}",
        )

    if not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="Resume text cannot be empty.")

    # --- Step 1: Extract skills from raw text ---
    logger.info("Step 1/6: Extracting skills from text...")
    try:
        claimed_skills = resume_parser.extract_skills(request.resume_text, skills_master)
    except Exception as e:
        raise HTTPException(422, f"Skill extraction failed: {str(e)}")

    # --- Step 2: Analyze GitHub ---
    logger.info("Step 2/6: Analyzing GitHub profile...")
    try:
        github_result = github_analyzer.analyze_github_profile(
            request.github_username, skills_master
        )
        demonstrated_skills = github_result["demonstrated_skills"]
    except Exception as e:
        raise HTTPException(
            502,
            f"GitHub analysis failed: {str(e)}. "
            f"Check username exists and token is valid."
        )

    # --- Step 3: Feature Engineering ---
    logger.info("Step 3/6: Building feature matrix...")
    try:
        role_data = job_roles_data[request.target_role]
        skill_matrix = feature_engineer.create_skill_matrix(
            claimed_skills,
            demonstrated_skills,
            role_data["required_skills"],
            role_data.get("nice_to_have", []),
        )
        X, y = feature_engineer.encode_for_model(skill_matrix)
    except Exception as e:
        raise HTTPException(500, f"Feature engineering failed: {str(e)}")

    # --- Step 4: ML Predictions ---
    logger.info("Step 4/6: Running ML predictions...")
    try:
        predictions = ml_model.predict(X)
        lr_probabilities = predictions["lr_probabilities"]
    except Exception as e:
        raise HTTPException(500, f"ML prediction failed: {str(e)}")

    # --- Step 5: Skill Gap Analysis ---
    logger.info("Step 5/6: Computing skill gaps...")
    try:
        analysis = skill_gap_analyzer.analyze(
            claimed_skills=claimed_skills,
            demonstrated_skills=demonstrated_skills,
            target_role=request.target_role,
            job_roles_data=job_roles_data,
            ml_predictions=predictions,
            lr_probabilities=lr_probabilities,
            skill_matrix=skill_matrix,
        )
    except Exception as e:
        raise HTTPException(500, f"Skill gap analysis failed: {str(e)}")

    # --- Step 6: Generate Report ---
    logger.info("Step 6/6: Generating report...")
    try:
        report = report_generator.generate_report(
            analysis_result=analysis,
            target_role=request.target_role,
            github_username=request.github_username,
            resume_skills=claimed_skills,
            github_skills=demonstrated_skills,
            model_summary=ml_model.get_model_summary(),
            github_insights_data=github_result,
        )
    except Exception as e:
        raise HTTPException(500, f"Report generation failed: {str(e)}")

    logger.info(f"Analysis complete! Match Score: {analysis['match_score']}%")
    return report


# ---------------------------------------------------------------------------
#  ENDPOINT: Model Metrics
# ---------------------------------------------------------------------------
@app.get("/model-metrics")
async def get_model_metrics():
    """Returns detailed ML model training metrics."""
    summary = ml_model.get_model_summary()
    summary["feature_importance"] = ml_model.get_feature_importance()
    return summary


# ---------------------------------------------------------------------------
#  ENDPOINT: Model Retrain
# ---------------------------------------------------------------------------
@app.get("/model-retrain")
async def retrain_model(tune: bool = False):
    """
    Forces retraining of ML models with fresh data.
    Rate limited to once per hour.
    Pass ?tune=true to enable GridSearchCV (slower but better).
    """
    global _last_retrain_time

    # Rate limit: 1 per hour
    now = time.time()
    if now - _last_retrain_time < 3600:
        remaining = int(3600 - (now - _last_retrain_time))
        raise HTTPException(
            429,
            f"Retrain rate limited. Try again in {remaining} seconds."
        )

    logger.info("Retraining ML models with fresh data...")
    _last_retrain_time = now

    # Load fresh training data
    X, y, source = dataset_loader.load_training_data()

    # Retrain with optional hyperparameter tuning
    metrics = ml_model.train(
        X, y,
        dataset_source=source,
        use_cross_validation=True,
        tune_hyperparameters=tune,
    )

    return {
        "status": "retrained",
        "dataset_source": source,
        "tune_hyperparameters": tune,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
#  ENDPOINT: Dataset Status
# ---------------------------------------------------------------------------
@app.get("/dataset-status")
async def get_dataset_status():
    """Shows HuggingFace dataset availability and cache status."""
    cache_dir = dataset_loader.cache_dir

    # Check which datasets are cached
    cached_datasets = []
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            if os.path.isdir(os.path.join(cache_dir, item)):
                cached_datasets.append(item)

    return {
        "huggingface_available": len(cached_datasets) > 0,
        "cached_datasets": cached_datasets,
        "training_data_source": ml_model.dataset_source,
        "cache_dir": cache_dir,
        "extractor_validation": _extractor_validation,
        "model_trained": ml_model.is_trained,
    }


# ---------------------------------------------------------------------------
#  Run with Uvicorn (if executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
