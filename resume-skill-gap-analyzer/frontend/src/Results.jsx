import ScoreCard from "./ScoreCard";
import Summary from "./Summary";
import SkillTable from "./SkillTable";
import "./cssFile/Results.css";

function Results({ report }) {
    const ml_insights = report.ml_insights;
    const git_insights = report.github_insights;
    const maxLang=git_insights.top_languages[0].bytes
    return (
        <section className="results-section">
            <ScoreCard report={report} />
            <Summary summary={report.executive_summary} />
            <div className="skill-breakdown">
                <h2>Skill Breakdown</h2>
                <SkillTable title="Required Skills" analysis={report.skill_breakdown.required_analysis} />
                <SkillTable title="Nice to have Skills" analysis={report.skill_breakdown.nice_to_have_analysis} />
            </div>
            <div className="recommendations">
                <h3>Recommendations</h3>
                {report.recommendations.map((item, index) => (
                    <div className="recommendation-item" key={index}>
                        <span className={`badge badge-${item.priority}`}>{item.priority}</span>
                        <div className="recommendation-content">
                            <div className="recommended-action">{item.action}</div>
                            <div className="recommended-hints">{item.resource_hint}</div>
                        </div>
                    </div>
                ))}
            </div>
            <div className="ml-insights">
                <h3>ML Model Insights</h3>
                <div className="ml-insights-container">
                    <div className="ml-grid">
                        <div className="ml-metric">
                            <div className="metric-value">{ml_insights.lr_accuracy}%</div>
                            <div className="metric-label">Logistic Regression Accuracy</div>
                        </div>
                        <div className="ml-metric">
                            <div className="metric-value">{ml_insights.dt_accuracy}%</div>
                            <div className="metric-label">Decision Tree Accuracy</div>
                        </div>
                    </div>
                    <div className="ml-explanation">
                        <strong>How it works: </strong>
                        {ml_insights.model_explanation}
                    </div>
                    <div className="ml-explanation">
                        <strong>Logistic Regression: </strong>
                        {ml_insights.lr_explanation}
                    </div>
                    <div className="ml-explanation">
                        <strong>Decision Tree: </strong>
                        {ml_insights.dt_explanation}
                    </div>
                </div>
            </div>
            <div className="github-insights">
                <h3>GitHub Insights</h3>
                <div className="github-insights-container">
                    <div className="github-stat">
                        <span className="github-stat-label">Repositories Analyzed</span>
                        <span className="github-stat-value">{git_insights.repos_analyzed}</span>
                    </div>
                    <h4>Top Languages</h4>
                    {git_insights.top_languages.map((item, index) => (
                        <div className="language-bar" key={index}>
                            <span className="language-bar-name">{item.language}</span>
                            <div className="language-bar-track">
                                <div className="language-bar-fill" style={{ width: `${(item.bytes / maxLang) * 100}%` }}></div>
                            </div>
                            <span className="language-bar-bytes">{(item.bytes/1000).toFixed(1)} KB</span>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}
export default Results;