import "./cssFile/InputSection.css";
import ResumeUpload from "./ResumeUpload";
import GithubUpload from "./GithubUpload";
import Role from "./Role";
import { useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
function InputSection({onError,onAnalyze,obtainedReport}) {
    const [resumeFile, setResumeFile] = useState(null);
    const [gitHubUserName, setGitHubUserName] = useState("");
    const [targetRole, setTargetRole] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");
        if (!resumeFile) {
            setError("Please upload a resume file (.pdf or .txt).");
            onError(error);
            return;
        }
        if (!gitHubUserName) {
            setError("Please enter a GitHub Username.");
            onError(error);
            return;
        }
        if (!targetRole) {
            setError("Please select a target role.");
            onError(error);
            return;
        }
        const formData = new FormData();
        formData.append("resume_file", resumeFile);
        formData.append("github_username", gitHubUserName);
        formData.append("target_role", targetRole);
        setLoading(true);
        onAnalyze(true);
        try {
            const response = await fetch(`${API_BASE_URL}/analyze`, {
                method: "POST",
                body: formData,
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Server Error");
            }
            const report = await response.json();
            obtainedReport(report);
            console.log(report);
        } catch (err) {
            setError("Analysis Failed: " + err.message);
            console.error(err);
        } finally {
            setLoading(false);
            onAnalyze(false);
        }
    };
    return (
        <form onSubmit={handleSubmit}>
            <h2 className="section-title">Analyze Your Profile</h2>
            <div className="upload_div">
            <ResumeUpload onFileSelect={setResumeFile}/>
          </div>

            <div className="github_div">
            <GithubUpload onUserNameEnter={setGitHubUserName}/>
          </div>

            <div className="role_div">
            <Role onTargetSet={setTargetRole}/>
          </div>

            <button type="submit" className="submit-btn">
                {loading ? "Analyzing..." : "Analyze Skill Gaps"}
            </button>
        </form>
    );
}

export default InputSection;