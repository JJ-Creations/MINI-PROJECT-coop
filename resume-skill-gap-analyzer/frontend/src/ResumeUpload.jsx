import { useState, useRef } from "react";
import "./cssFile/ResumeUpload.css";

function ResumeUpload({onFileSelect}) {
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const validTypes = ["application/pdf", "text/plain"];

  const handleFile = (file) => {
    if (!file) return;

    if (validTypes.includes(file.type)) {
      setFileName(file.name);
      setError("");
      onFileSelect(file);
    } else {
      setFileName("");
      setError("Only PDF or TXT files are allowed.");
      onFileSelect(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    handleFile(file);
    e.target.value = "";
  };

  return (
    <div className="resume-upload-container">
      <label htmlFor="resume-upload" className="upload-label">
        Upload Resume
      </label>

      <div
        className={`upload-box ${isDragging ? "dragging" : ""}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleBrowseClick}
        role="button"
        tabIndex={0}
      >
        <span className="upload-icon">📄</span>
        <p className="upload-title">
          <strong>Drag & drop your resume here</strong>
        </p>
        <p className="upload-subtext">or click to browse (.pdf, .txt)</p>

        {fileName && (
          <p className="selected-file">
            Selected: {fileName}
          </p>
        )}

        {error && (
          <p className="error-text">
            {error}
          </p>
        )}
      </div>

      <input
        id="resume-upload"
        type="file"
        accept=".pdf,.txt"
        ref={fileInputRef}
        onChange={handleFileChange}
        hidden
      />
    </div>
  );
}

export default ResumeUpload;