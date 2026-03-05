import { useState } from "react";
import "./cssFile/GithubUpload.css";

const GithubUpload = ({onUserNameEnter}) => {
  const [username, setUsername] = useState("");

  const handleChange=(e)=>{
    const value=e.target.value;
    setUsername(value);
    onUserNameEnter(value);
  };

  return (
    <div className="form-group">
      <label htmlFor="github-username" className="form-label">
        GitHub Username
      </label>

      <div className="input-with-prefix">
        <span className="input-prefix">@</span>
        <input
          type="text"
          id="github-username"
          name="github-username"
          placeholder="your-github-username"
          className="text-input"
          value={username}
          onChange={handleChange}
          autoComplete="off"
          required
        />
      </div>
    </div>
  );
};

export default GithubUpload;