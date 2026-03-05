import "./cssFile/App.css";
import InputSection from "./InputSection";
import { useState } from "react";
import Results from "./Results";

function App() {
  const [error,setError]=useState("");
  const [loading,setLoading]=useState(false);
  const [report,setReport]=useState(null);
  return (
    <>
      <header className="app-header">
        <div className="container">
          <h1 className="app-title">Resume Skill Gap Analyzer</h1>
          <p className="app-subtitle">Upload your resume, connect your GitHub, and discover your skill gaps for any target role — powered by Machine Learning.</p>
        </div>
      </header>
      <main className="container">
        <section className="input-section">
          <InputSection err={setError} onAnalyze={setLoading} obtainedReport={setReport}/>
        </section>
        {error &&(
          <div className="error-message">{error}</div>
        )}
        {loading &&(
          <div className="loading-overlay" id="loading-overlay">
            <div className="spinner-container">
                <div className="spinner"></div>
                <p className="loading-text">Analyzing your profile...</p>
                <p className="loading-subtext" id="loading-step">Parsing resume...</p>
            </div>
        </div>
        )}
        {report && <Results report={report}/>}
      </main>
      <footer className="app-footer">
        <div className="container">
          <p>Resume Skill Gap Analyzer &mdash; Powered by FastAPI, Scikit-learn & spaCy</p>
        </div>
      </footer>
    </>
  );
}

export default App;