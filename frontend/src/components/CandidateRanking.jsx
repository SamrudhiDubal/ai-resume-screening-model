import { useState } from 'react';
import { rankCandidates } from '../api.js';
import { formatPercent, parseCandidates, scoreLabel } from '../utils/score.js';

export default function CandidateRanking({ api = { rankCandidates } }) {
  const [job, setJob] = useState('');
  const [resumes, setResumes] = useState('');
  const [rows, setRows] = useState([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  async function onSubmit(e) {
    e.preventDefault();
    const candidates = parseCandidates(resumes);
    if (!job.trim() || candidates.length === 0) {
      setError('Add a job description and at least one resume.');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const data = await api.rankCandidates(job, candidates);
      setRows(data.results);
    } catch (err) {
      setError(err.message);
      setRows([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="card">
      <h2>Rank candidates for a job</h2>
      <form onSubmit={onSubmit}>
        <label htmlFor="rank-job">Job description</label>
        <textarea id="rank-job" rows={4} value={job} onChange={(e) => setJob(e.target.value)} />
        <label htmlFor="rank-resumes">Resumes (separate with a line containing ---, optional first line "Name: ...")</label>
        <textarea id="rank-resumes" rows={10} value={resumes} onChange={(e) => setResumes(e.target.value)} />
        <button type="submit" disabled={loading}>{loading ? 'Ranking…' : 'Rank candidates'}</button>
      </form>

      {error && <p role="alert" className="error">{error}</p>}

      {rows.length > 0 && (
        <table>
          <thead>
            <tr><th>Rank</th><th>Candidate</th><th>Category</th><th>Score</th><th>Fit</th><th>Matched skills</th></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td>{r.rank}</td>
                <td>{r.id}</td>
                <td>{r.category}</td>
                <td>{formatPercent(r.score)}</td>
                <td>{scoreLabel(r.score)}</td>
                <td>{r.matchedSkills.join(', ') || '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
