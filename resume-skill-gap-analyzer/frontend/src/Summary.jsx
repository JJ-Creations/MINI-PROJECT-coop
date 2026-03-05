import "./cssFile/Summary.css";
function Summary({summary}){
    const cards=[
        {value:summary.total_resume_skills,label:"Resume Skills"},
        {value:summary.total_github_skills,label:"GitHub Skills"},
        {value:summary.missing_critical_skills,label:"Missing Critical Skills"},
        {value:Number(summary.confidence_score || 0).toFixed(1) + "%",label:"ML Confidence"},
    ]
    return(
        <div className="summary-grid">
            {cards.map((card,index)=>(
                <div className="summary-card" key={index}>
                    <div className="stat-value">{card.value}</div>
                    <div className="stat-label">{card.label}</div>
                </div>
            ))}
        </div>
    );
}
export default Summary;