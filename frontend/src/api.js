const BASE_URL = (import.meta.env?.VITE_API_URL || 'http://localhost:5000').replace(/\/$/, '');

async function post(path, body, fetchImpl = fetch) {
  let res;
  try {
    res = await fetchImpl(`${BASE_URL}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error('Cannot reach the screening API. Is the backend running?');
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

export const screenResume = (resumeText, fetchImpl) => post('/api/screen', { resumeText }, fetchImpl);
export const matchResume = (resumeText, jobDescription, fetchImpl) =>
  post('/api/match', { resumeText, jobDescription }, fetchImpl);
export const rankCandidates = (jobDescription, candidates, topK, fetchImpl) =>
  post('/api/rank', { jobDescription, candidates, ...(topK ? { topK } : {}) }, fetchImpl);
export { BASE_URL };
