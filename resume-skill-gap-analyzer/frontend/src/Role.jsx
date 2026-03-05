import { useState, useEffect } from "react";
import "./cssFile/Role.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
function Role({onTargetSet}) {
    const [jobRoles, setJobRoles] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");

    useEffect(() => {
        const fetchJobRoles = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/job-roles`);
                if (!response.ok) {
                    throw new Error(`Server error: ${response.status}`);
                }
                const data = await response.json();
                setJobRoles(data.job_roles);
            } catch (err) {
                console.error("Failed to fetch job roles:", err);
                setError("Cannot connect to the API. Make sure the backend is running.");
            } finally {
                setLoading(false);
            }
        };
        fetchJobRoles();
    }, []);
    return (
        <>
            <div className="form-group">
                <label htmlFor="target-role" className="form-label">
                    Target Job Role
                </label>

                <select id="target-role" className="select-input" onChange={(e)=>onTargetSet(e.target.value)} required>
                    {loading && <option>Loading roles...</option>}

                    {error && <option>{error}</option>}

                    {!loading && !error && (
                        <>
                            <option value="">Select a target role...</option>
                            {jobRoles.map((role, index) => (
                                <option key={index} value={role.name}>
                                    {role.name}
                                </option>
                            ))}
                        </>
                    )}
                </select>
            </div>
        </>
    );
}
export default Role;