import { formatPercent } from '../utils/score.js';

export default function ScoreBar({ label, value }) {
  const pct = formatPercent(value);
  return (
    <div className="score-bar">
      <span className="score-bar__label">{label}</span>
      <div className="score-bar__track" role="progressbar" aria-label={label}
        aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100}>
        <div className="score-bar__fill" style={{ width: pct }} />
      </div>
      <span className="score-bar__value">{pct}</span>
    </div>
  );
}
