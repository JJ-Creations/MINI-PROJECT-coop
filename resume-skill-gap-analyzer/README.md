# Resume Skill Gap Analyzer

A single-agent ML-powered application that analyzes resumes and GitHub profiles to identify skill gaps for target job roles.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (HTML/CSS/JS)                       │
│   Upload Resume  +  GitHub Username  +  Target Role  →  [Analyze]  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  POST /analyze
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND (main.py)                      │
│                                                                     │
│  ┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐   │
│  │ Resume Parser │   │ GitHub Analyzer  │   │  Skills Master    │   │
│  │ (spaCy+Regex)│   │ (REST API v3)    │   │  + Job Roles DB   │   │
│  └──────┬───────┘   └────────┬─────────┘   └─────────┬─────────┘   │
│         │  claimed_skills    │  demonstrated_skills   │             │
│         ▼                    ▼                        ▼             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Feature Engineering (Pandas)                    │   │
│  │         Build skill matrix: in_resume × in_github           │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              ML Models (Scikit-learn)                        │   │
│  │    Logistic Regression  +  Decision Tree Classifier         │   │
│  │    → Predictions + Probabilities per skill                  │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Skill Gap Analyzer                              │   │
│  │    Compare candidate skills vs. role requirements            │   │
│  │    → match_score, gap_score, per-skill status               │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Report Generator                                │   │
│  │    Compile: summary + breakdown + recommendations + ML info  │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                              │                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               │  JSON Report
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FRONTEND RESULTS DASHBOARD                      │
│   Score Card  │  Skill Table  │  Recommendations  │  ML Insights   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repo-url>
cd resume-skill-gap-analyzer
```

### 2. Install Python Dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 4. Configure GitHub Token (Optional)

```bash
cp .env.example backend/.env
# Edit backend/.env and add your GitHub personal access token
```

Without a token, the GitHub API is limited to 60 requests/hour. With a token, you get 5,000 requests/hour.

### 5. Start the Backend Server

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 6. Open the Frontend

Open `frontend/index.html` in your browser. It connects to the API at `http://localhost:8000`.

---

## API Endpoints

| Method | Path             | Description                                      |
|--------|------------------|--------------------------------------------------|
| GET    | `/`              | Health check — confirms API is running            |
| GET    | `/job-roles`     | Returns all available job roles with requirements |
| GET    | `/skills-master` | Returns the full master skills list by category   |
| POST   | `/analyze`       | Main analysis — accepts resume file upload        |
| POST   | `/analyze-text`  | Alternative — accepts raw resume text (JSON body) |

---

## How the ML Models Work

### Logistic Regression
A linear classifier that estimates the probability of a skill being genuinely present based on two features: whether it appears in the resume and whether it appears on GitHub. It outputs a calibrated confidence score (0-100%) for each skill, which is used to assess evidence reliability. Wrapped in a pipeline with StandardScaler for feature normalization.

### Decision Tree
A rule-based classifier (max depth 4) that creates interpretable if/then decision rules. For example: "If the skill is in the resume AND on GitHub, predict present." It validates the Logistic Regression results and provides feature importance scores showing which evidence source matters more.

### Training Data
Both models are trained on 200 synthetic samples encoding the intuition that: (a) skills confirmed by both resume and GitHub are always real, (b) GitHub-only evidence is 90% reliable, (c) resume-only claims are 85% reliable, and (d) no evidence means the skill is absent.

---

## How Skill Gap Scoring Works

1. **Match Score**: `(required_skills_present / total_required_skills) * 100`
2. **Gap Score**: `100 - match_score`
3. **Per-Skill Status**:
   - **Strong**: Found in both resume AND GitHub (strongest evidence)
   - **Claimed Only**: In resume but not GitHub (unverified claim)
   - **Demonstrated Only**: On GitHub but not in resume (hidden strength)
   - **Missing**: In neither source (skill gap)
4. **Confidence**: Average ML probability for present required skills

---

## Sample API Response

```json
{
  "title": "Skill Gap Analysis Report",
  "target_role": "Data Scientist",
  "github_username": "alexjohnson",
  "executive_summary": {
    "match_score": 85.7,
    "match_label": "Excellent",
    "total_resume_skills": 18,
    "total_github_skills": 5,
    "missing_critical_skills": 1,
    "confidence_rating": "High",
    "confidence_score": 82.3
  },
  "skill_breakdown": {
    "required_analysis": [
      {
        "skill": "Python",
        "status": "strong",
        "in_resume": true,
        "in_github": true,
        "ml_prediction": 1,
        "probability": 0.95
      }
    ],
    "missing_required": ["Statistics"],
    "strengths": ["Python", "Pandas", "NumPy"],
    "claims_not_proven": ["Scikit-learn"],
    "hidden_strengths": []
  },
  "recommendations": [
    {
      "skill": "Statistics",
      "priority": "Critical",
      "action": "Learn Statistics — required for Data Scientist",
      "resource_hint": "Structured course on Statistics + portfolio project"
    }
  ]
}
```

---

## Project Structure

```
resume-skill-gap-analyzer/
├── backend/
│   ├── main.py                    # FastAPI application entry point
│   ├── requirements.txt           # Python dependencies
│   ├── modules/
│   │   ├── __init__.py            # Package initialization
│   │   ├── resume_parser.py       # PDF/TXT parsing + skill extraction
│   │   ├── github_analyzer.py     # GitHub API profile analysis
│   │   ├── feature_engineering.py # Build ML feature vectors
│   │   ├── ml_model.py            # Logistic Regression + Decision Tree
│   │   ├── skill_gap_analyzer.py  # Compute skill gaps vs. role
│   │   └── report_generator.py    # Compile final analysis report
│   ├── data/
│   │   ├── job_roles.json         # Job role requirements database
│   │   └── skills_master.json     # Master list of 80+ tech skills
│   └── models/                    # Directory for saved ML models
├── frontend/
│   ├── index.html                 # Single-page frontend
│   ├── style.css                  # Pure CSS styling
│   └── app.js                     # Vanilla JavaScript logic
├── sample_resumes/
│   └── sample_resume.txt          # Sample resume for testing
├── .env.example                   # Environment variable template
└── README.md                      # This file
```

---

## Limitations and Future Improvements

### Current Limitations
- ML models use synthetic training data (200 samples) — accuracy reflects simulated patterns, not real hiring data
- GitHub analysis only covers public repositories — private repos and organizational work are not captured
- Skill matching uses exact keyword matching — synonyms and variations may be missed (e.g., "JS" vs "JavaScript")
- Without a GitHub token, API rate limiting restricts analysis to 60 requests/hour
- PDF parsing depends on text-based PDFs — scanned/image PDFs are not supported

### Future Improvements
- Integrate real labeled training data from hiring pipelines for more accurate ML predictions
- Add support for LinkedIn profile analysis as a third evidence source
- Implement semantic similarity matching (using embeddings) instead of exact keyword matching
- Add support for image-based PDF parsing using OCR (Tesseract)
- Create user accounts to track skill progress over time
- Add job posting URL parser to extract requirements from actual job listings
- Implement model persistence (save/load trained models) for faster startup
- Add more job roles and expand the skills master list
