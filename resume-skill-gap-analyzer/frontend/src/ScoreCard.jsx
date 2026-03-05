import "./cssFile/ScoreCard.css";
function ScoreCard({ report }) {
    const score = Math.round(report.skill_breakdown.match_score);
    const label = report.executive_summary.match_label;
    const targetRole = report.target_role;
    let colour;
    if (score >= 75) {
        colour = "var(--color-excellent)";
    } else if (score >= 50) {
        colour = "var(--color-good)";
    } else if (score >= 25) {
        colour = "var(--color-fair)";
    } else {
        colour = "var(--color-poor)";
    }
    return (
        <div className="score-card">
            <h2 className="section-title">Match Score</h2>
            <div className="score-display">
                <div className="score-circle" style={{ borderColor: colour }}>
                    <span className="score-value" style={{ color:colour }}>
                        {score}
                    </span>
                    <span className="score-percent">%</span>
                </div>
                <p className="score-label" style={{ color:colour }}>
                    {label}
                </p>
            </div>
            <p className="target-display">Target: <strong>{targetRole}</strong></p>
        </div>
    )
}
export default ScoreCard;