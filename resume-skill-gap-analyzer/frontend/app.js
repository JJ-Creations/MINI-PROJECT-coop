/**
 * =============================================================================
 *  Resume Skill Gap Analyzer — Frontend Application
 * =============================================================================
 *  This JavaScript file handles all frontend logic:
 *    1. Fetching available job roles from the API on page load
 *    2. Handling form submission (file upload + analysis request)
 *    3. Rendering the analysis results dynamically
 *    4. Managing loading states and error display
 *
 *  No frameworks — pure vanilla JavaScript for maximum clarity.
 * =============================================================================
 */

// --- Configuration ---
const API_BASE_URL = "http://localhost:8000";

// --- DOM Element References ---
const analyzeForm = document.getElementById("analyze-form");
const resumeFileInput = document.getElementById("resume-file");
const githubUsernameInput = document.getElementById("github-username");
const targetRoleSelect = document.getElementById("target-role");
const analyzeBtn = document.getElementById("analyze-btn");
const fileNameDisplay = document.getElementById("file-name-display");
const errorDisplay = document.getElementById("error-display");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingStep = document.getElementById("loading-step");
const resultsSection = document.getElementById("results-section");
const dropZone = document.getElementById("drop-zone");

// =====================================================================
//  INITIALIZATION — Runs on Page Load
// =====================================================================

/**
 * Fetch available job roles from the API and populate the dropdown.
 * This runs automatically when the page loads.
 */
async function fetchJobRoles() {
    try {
        const response = await fetch(`${API_BASE_URL}/job-roles`);

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();

        // Clear the loading placeholder and populate options
        targetRoleSelect.innerHTML = '<option value="">Select a target role...</option>';

        data.job_roles.forEach(function (role) {
            const option = document.createElement("option");
            option.value = role.name;
            option.textContent = role.name;
            targetRoleSelect.appendChild(option);
        });

    } catch (error) {
        console.error("Failed to fetch job roles:", error);
        targetRoleSelect.innerHTML = '<option value="">Failed to load roles</option>';
        showError("Cannot connect to the API. Make sure the backend is running on " + API_BASE_URL);
    }
}

// Fetch roles when the page loads
document.addEventListener("DOMContentLoaded", fetchJobRoles);

// =====================================================================
//  FILE UPLOAD — Display selected file name + drag-and-drop
// =====================================================================

/**
 * Show the selected file name when a user picks a file.
 */
resumeFileInput.addEventListener("change", function () {
    if (this.files.length > 0) {
        fileNameDisplay.textContent = "Selected: " + this.files[0].name;
    } else {
        fileNameDisplay.textContent = "";
    }
});

/**
 * Add drag-over visual feedback to the upload area.
 */
dropZone.addEventListener("dragover", function (e) {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", function () {
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", function (e) {
    e.preventDefault();
    dropZone.classList.remove("drag-over");

    // Transfer the dropped files to the file input
    if (e.dataTransfer.files.length > 0) {
        resumeFileInput.files = e.dataTransfer.files;
        fileNameDisplay.textContent = "Selected: " + e.dataTransfer.files[0].name;
    }
});

// =====================================================================
//  FORM SUBMISSION — Send analysis request to the API
// =====================================================================

/**
 * Handle form submission: validate inputs, send to API, render results.
 */
analyzeForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    hideError();

    // --- Validate inputs ---
    const file = resumeFileInput.files[0];
    const githubUsername = githubUsernameInput.value.trim();
    const targetRole = targetRoleSelect.value;

    if (!file) {
        showError("Please upload a resume file (.pdf or .txt).");
        return;
    }

    if (!githubUsername) {
        showError("Please enter a GitHub username.");
        return;
    }

    if (!targetRole) {
        showError("Please select a target job role.");
        return;
    }

    // --- Build FormData for multipart upload ---
    const formData = new FormData();
    formData.append("resume_file", file);
    formData.append("github_username", githubUsername);
    formData.append("target_role", targetRole);

    // --- Show loading state ---
    showLoading();
    disableForm(true);

    try {
        // Simulate step-by-step loading messages
        updateLoadingStep("Parsing resume...");

        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        const report = await response.json();

        // --- Hide loading and render results ---
        hideLoading();
        renderResults(report);

    } catch (error) {
        hideLoading();
        showError("Analysis failed: " + error.message);
        console.error("Analysis error:", error);
    } finally {
        disableForm(false);
    }
});

// =====================================================================
//  RENDER RESULTS — Build all result sections from API response
// =====================================================================

/**
 * Render the complete analysis report in the results section.
 * Builds all UI components dynamically from the report data.
 *
 * @param {Object} report - The full report object from the API
 */
function renderResults(report) {
    // Show the results section
    resultsSection.style.display = "block";

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

    // --- Render Score Card ---
    renderScoreCard(report);

    // --- Render Executive Summary ---
    renderExecutiveSummary(report.executive_summary);

    // --- Render Required Skills Table ---
    renderSkillTable(
        "required-skills-body",
        report.skill_breakdown.required_analysis,
        true  // show ML confidence column
    );

    // --- Render Nice-to-Have Skills Table ---
    renderSkillTable(
        "nice-skills-body",
        report.skill_breakdown.nice_to_have_analysis,
        false  // no ML confidence column
    );

    // --- Render Recommendations ---
    renderRecommendations(report.recommendations);

    // --- Render ML Insights ---
    renderMLInsights(report.ml_insights);

    // --- Render GitHub Insights ---
    renderGitHubInsights(report.github_insights);
}

// =====================================================================
//  RENDER: Score Card
// =====================================================================

/**
 * Render the large circular score display with color coding.
 * Green: 75+, Yellow: 50-74, Orange: 25-49, Red: 0-24
 */
function renderScoreCard(report) {
    const score = report.skill_breakdown.match_score;
    const label = report.executive_summary.match_label;
    const scoreValue = document.getElementById("score-value");
    const scoreLabel = document.getElementById("score-label");
    const scoreCircle = document.getElementById("score-circle");
    const resultRole = document.getElementById("result-target-role");

    // Set the score text
    scoreValue.textContent = Math.round(score);
    scoreLabel.textContent = label;
    resultRole.textContent = report.target_role;

    // Color-code the score circle based on thresholds
    let color;
    if (score >= 75) {
        color = "var(--color-score-excellent)";
    } else if (score >= 50) {
        color = "var(--color-score-good)";
    } else if (score >= 25) {
        color = "var(--color-score-fair)";
    } else {
        color = "var(--color-score-poor)";
    }

    scoreCircle.style.borderColor = color;
    scoreValue.style.color = color;
    scoreLabel.style.color = color;
}

// =====================================================================
//  RENDER: Executive Summary Cards
// =====================================================================

/**
 * Render the 4-card summary grid with key metrics.
 */
function renderExecutiveSummary(summary) {
    const grid = document.getElementById("summary-grid");

    grid.innerHTML = `
        <div class="summary-card">
            <div class="stat-value">${summary.total_resume_skills}</div>
            <div class="stat-label">Resume Skills</div>
        </div>
        <div class="summary-card">
            <div class="stat-value">${summary.total_github_skills}</div>
            <div class="stat-label">GitHub Skills</div>
        </div>
        <div class="summary-card">
            <div class="stat-value">${summary.missing_critical_skills}</div>
            <div class="stat-label">Missing Critical</div>
        </div>
        <div class="summary-card">
            <div class="stat-value">${summary.confidence_score.toFixed(1)}%</div>
            <div class="stat-label">ML Confidence</div>
        </div>
    `;
}

// =====================================================================
//  RENDER: Skill Breakdown Tables
// =====================================================================

/**
 * Render a skills table (required or nice-to-have).
 * Each row fades in sequentially for a polished animation effect.
 *
 * @param {string}  tbodyId       - ID of the table body element
 * @param {Array}   skills        - Array of skill analysis objects
 * @param {boolean} showConfidence - Whether to show the ML confidence column
 */
function renderSkillTable(tbodyId, skills, showConfidence) {
    const tbody = document.getElementById(tbodyId);
    tbody.innerHTML = "";

    skills.forEach(function (skill, index) {
        const row = document.createElement("tr");

        // Stagger the fade-in animation for each row
        row.style.animationDelay = (index * 0.08) + "s";

        // Format the status as a colored badge
        const statusBadge = `<span class="badge badge-${skill.status}">${formatStatus(skill.status)}</span>`;

        // Check / X marks for resume and GitHub columns
        const resumeMark = skill.in_resume
            ? '<span class="check-mark">&#10003;</span>'
            : '<span class="x-mark">&#10007;</span>';
        const githubMark = skill.in_github
            ? '<span class="check-mark">&#10003;</span>'
            : '<span class="x-mark">&#10007;</span>';

        let html = `
            <td><strong>${skill.skill}</strong></td>
            <td>${statusBadge}</td>
            <td>${resumeMark}</td>
            <td>${githubMark}</td>
        `;

        // Add ML confidence bar if applicable
        if (showConfidence && skill.probability !== undefined) {
            const prob = Math.round(skill.probability * 100);
            const barColor = prob >= 70 ? "var(--color-strong)" : prob >= 40 ? "var(--color-claimed)" : "var(--color-missing)";

            html += `
                <td>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar">
                            <div class="confidence-bar-fill" style="width: ${prob}%; background: ${barColor};"></div>
                        </div>
                        <span class="confidence-text">${prob}%</span>
                    </div>
                </td>
            `;
        }

        row.innerHTML = html;
        tbody.appendChild(row);
    });
}

/**
 * Convert a status key to a human-readable label.
 */
function formatStatus(status) {
    const labels = {
        "strong": "Strong",
        "claimed_only": "Claimed Only",
        "demonstrated_only": "GitHub Only",
        "missing": "Missing",
    };
    return labels[status] || status;
}

// =====================================================================
//  RENDER: Recommendations
// =====================================================================

/**
 * Render the list of actionable recommendations for missing skills.
 */
function renderRecommendations(recommendations) {
    const container = document.getElementById("recommendations-container");

    if (recommendations.length === 0) {
        container.innerHTML = '<p style="color: var(--color-strong); font-weight: 600;">No gaps found — you match all requirements!</p>';
        return;
    }

    let html = "";
    recommendations.forEach(function (rec) {
        const priorityClass = rec.priority === "Critical" ? "priority-critical" : "priority-recommended";

        html += `
            <div class="recommendation-item">
                <span class="priority-badge ${priorityClass}">${rec.priority}</span>
                <div class="recommendation-content">
                    <div class="recommendation-action">${rec.action}</div>
                    <div class="recommendation-hint">${rec.resource_hint}</div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

// =====================================================================
//  RENDER: ML Insights
// =====================================================================

/**
 * Render the ML model insights panel showing accuracy and explanations.
 */
function renderMLInsights(insights) {
    const container = document.getElementById("ml-insights-container");

    container.innerHTML = `
        <div class="ml-grid">
            <div class="ml-metric">
                <div class="metric-value">${insights.lr_accuracy}%</div>
                <div class="metric-label">Logistic Regression Accuracy</div>
            </div>
            <div class="ml-metric">
                <div class="metric-value">${insights.dt_accuracy}%</div>
                <div class="metric-label">Decision Tree Accuracy</div>
            </div>
        </div>
        <div class="ml-explanation">
            <strong>How it works:</strong> ${insights.model_explanation}
        </div>
        <div class="ml-explanation" style="margin-top: 8px;">
            <strong>Logistic Regression:</strong> ${insights.lr_explanation}
        </div>
        <div class="ml-explanation" style="margin-top: 8px;">
            <strong>Decision Tree:</strong> ${insights.dt_explanation}
        </div>
    `;
}

// =====================================================================
//  RENDER: GitHub Insights
// =====================================================================

/**
 * Render GitHub profile insights — repo count, top languages, etc.
 */
function renderGitHubInsights(insights) {
    const container = document.getElementById("github-insights-container");

    // Check for GitHub errors
    if (insights.error) {
        container.innerHTML = `
            <div class="error-message" style="display: block;">
                GitHub: ${insights.error}
            </div>
        `;
        return;
    }

    // Repos analyzed stat
    let html = `
        <div class="github-stat">
            <span class="github-stat-label">Repositories Analyzed</span>
            <span class="github-stat-value">${insights.repos_analyzed}</span>
        </div>
    `;

    // Top languages with visual bars
    if (insights.top_languages && insights.top_languages.length > 0) {
        html += '<h3 style="margin: 16px 0 8px; font-size: 0.95rem;">Top Languages</h3>';

        // Find max bytes for relative bar sizing
        const maxBytes = insights.top_languages[0].bytes || 1;

        insights.top_languages.forEach(function (lang) {
            const percentage = Math.round((lang.bytes / maxBytes) * 100);
            const bytesDisplay = formatBytes(lang.bytes);

            html += `
                <div class="language-bar">
                    <span class="language-bar-name">${lang.language}</span>
                    <div class="language-bar-track">
                        <div class="language-bar-fill" style="width: ${percentage}%;"></div>
                    </div>
                    <span class="language-bar-bytes">${bytesDisplay}</span>
                </div>
            `;
        });
    }

    // Hidden strengths
    if (insights.hidden_strengths && insights.hidden_strengths.length > 0) {
        html += `
            <div class="github-stat" style="margin-top: 12px;">
                <span class="github-stat-label">Hidden Strengths (on GitHub, not in resume)</span>
                <span class="github-stat-value">${insights.hidden_strengths.join(", ")}</span>
            </div>
        `;
    }

    container.innerHTML = html;
}

/**
 * Format byte counts into human-readable sizes (KB, MB).
 */
function formatBytes(bytes) {
    if (bytes >= 1048576) {
        return (bytes / 1048576).toFixed(1) + " MB";
    } else if (bytes >= 1024) {
        return (bytes / 1024).toFixed(1) + " KB";
    }
    return bytes + " B";
}

// =====================================================================
//  UTILITY FUNCTIONS — Loading, Error, Form State
// =====================================================================

/**
 * Show the loading overlay with a step message.
 */
function showLoading() {
    loadingOverlay.style.display = "flex";
}

/**
 * Hide the loading overlay.
 */
function hideLoading() {
    loadingOverlay.style.display = "none";
}

/**
 * Update the loading step text.
 */
function updateLoadingStep(step) {
    loadingStep.textContent = step;
}

/**
 * Show an error message to the user.
 */
function showError(message) {
    errorDisplay.textContent = message;
    errorDisplay.style.display = "block";
}

/**
 * Hide the error message.
 */
function hideError() {
    errorDisplay.style.display = "none";
}

/**
 * Enable or disable the form during submission.
 */
function disableForm(disabled) {
    analyzeBtn.disabled = disabled;
    analyzeBtn.textContent = disabled ? "Analyzing..." : "Analyze Skill Gaps";
    resumeFileInput.disabled = disabled;
    githubUsernameInput.disabled = disabled;
    targetRoleSelect.disabled = disabled;
}
