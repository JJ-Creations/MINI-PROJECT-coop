import "./cssFile/SkillTable.css";
import ConfidenceBar from "./ConfidenceBar"
function SkillTable({ title,analysis }) {
    if (!analysis) return null;
    const showConfidence=analysis.some(item=>item.probability!==undefined);

    return (
        <div className="card">
            <h3>{title}</h3>
            <div className="table-container">
                <table className="skill-table">
                    <thead>
                        <tr>
                            <th>Skill</th>
                            <th>Status</th>
                            <th>In Resume</th>
                            <th>On GitHub</th>
                            {showConfidence &&<th>ML Confidence</th>}
                        </tr>
                    </thead>
                    <tbody>
                        {analysis.map((item, index) => (
                            <tr key={index}>
                                <td>{item.skill}</td>
                                <td>
                                    <span className={`badge badge-${item.status}`}>
                                        {formatStatus(item.status)}
                                    </span>
                                </td>
                                <td>
                                    {item.in_resume ? (
                                        <span className="check-mark">✔</span>
                                    ) : (
                                        <span className="x-mark">✘</span>
                                    )}
                                </td>
                                <td>
                                    {item.in_github? (
                                        <span className="check-mark">✔</span>
                                    ) : (
                                        <span className="x-mark">✘</span>
                                    )}
                                </td>
                                {showConfidence &&(<td>
                                    {item.probability != null ? (
                                        <ConfidenceBar probability={item.probability} />) : ("-")}
                                </td>
                                )}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

function formatStatus(status) {
    switch (status) {
        case "strong":
            return "Strong";
        case "claimed_only":
            return "Claimed Only";
        case "missing":
            return "Missing";
        case "demonstrated_only":
            return "Demonstrated Only";
        default:
            return status;
    }
}
export default SkillTable;