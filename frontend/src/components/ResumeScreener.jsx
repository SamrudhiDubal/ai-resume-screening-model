import { useState } from 'react';
import { screenResume, matchResume } from '../api.js';
import { formatPercent, scoreLabel } from '../utils/score.js';
import ScoreBar from './ScoreBar.jsx';
import SkillList from './SkillList.jsx';

export default function ResumeScreener({ api = { screenResume, matchResume } }) {
  const [resume, setResume] = useState('');
  const [job, setJob] = useState('');
  const [result, setResult] = useState(null);
  const [match, setMatch] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  async function onSubmit(e) {
    e.preventDefault();
    if (!resume.trim()) {
      setError('Paste a resume first.');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const [screen, m] = await Promise.all([
        api.screenResume(resume),
        job.trim() ? api.matchResume(resume, job) : Promise.resolve(null),
      ]);
      setResult(screen);
      setMatch(m);
    } catch (err) {
      setError(err.message);
      setResult(null);
      setMatch(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="card">
      <h2>Screen a resume</h2>
      <form onSubmit={onSubmit}>
        <label htmlFor="resume">Resume text</label>
        <textarea id="resume" rows={8} value={resume} onChange={(e) => setResume(e.target.value)}
          placeholder="Paste the candidate's resume" />
        <label htmlFor="job">Job description (optional, for a match score)</label>
        <textarea id="job" rows={5} value={job} onChange={(e) => setJob(e.target.value)}
          placeholder="Paste the job description" />
        <button type="submit" disabled={loading}>{loading ? 'Analysing…' : 'Analyse resume'}</button>
      </form>

      {error && <p role="alert" className="error">{error}</p>}

      {result && (
        <div className="result" data-testid="screen-result">
          <h3>Predicted category: <strong>{result.category}</strong></h3>
          <p>Confidence {formatPercent(result.confidence)}</p>
          {result.topCategories.map((c) => (
            <ScoreBar key={c.category} label={c.category} value={c.probability} />
          ))}
          <SkillList title="Detected skills" skills={result.skills} />
        </div>
      )}

      {match && (
        <div className="result" data-testid="match-result">
          <h3>Job match: {formatPercent(match.score)} <span className="badge">{scoreLabel(match.score)}</span></h3>
          <ScoreBar label="Text similarity" value={match.cosine} />
          <ScoreBar label="Required skills covered" value={match.skillMatch} />
          <SkillList title="Matched skills" skills={match.matchedSkills} variant="good" />
          <SkillList title="Missing skills" skills={match.missingSkills} variant="bad" />
        </div>
      )}
    </section>
  );
}
