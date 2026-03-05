import "./cssFile/ConfidenceBar.css";
function ConfidenceBar({ probability }) {
    const prob = Math.round(probability * 100);

    let barColor;
    if (prob >= 70) {
        barColor = "var(--color-strong)";
    } else if (prob >= 40) {
        barColor = "var(--color-claimed)";
    } else {
        barColor = "var(--color-missing)";
    }

    return (
        <div className="confidence-bar-container">
            <div className="confidence-bar">
                <div
                    className="confidence-bar-fill"
                    style={{ width: `${prob}%`,backgroundColor:barColor}}
                />
            </div>
            <span className="confidence-text">{prob}%</span>
        </div>
    );
}
export default ConfidenceBar;