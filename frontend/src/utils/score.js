/** 0.734 -> "73%" */
export function formatPercent(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '–';
  return `${Math.round(Math.min(Math.max(value, 0), 1) * 100)}%`;
}

/** Turns a 0..1 match score into a label recruiters can read at a glance. */
export function scoreLabel(score) {
  if (score >= 0.5) return 'Strong match';
  if (score >= 0.3) return 'Good match';
  if (score >= 0.15) return 'Partial match';
  return 'Weak match';
}

/**
 * Splits pasted text into candidates. Resumes are separated by a line containing only "---".
 * An optional first line "Name: ..." becomes the candidate id.
 */
export function parseCandidates(text) {
  return String(text || '')
    .split(/^\s*---\s*$/m)
    .map((block) => block.trim())
    .filter(Boolean)
    .map((block, i) => {
      const m = block.match(/^name:\s*(.+)$/im);
      const resumeText = m ? block.replace(m[0], '').trim() : block;
      return { id: m ? m[1].trim() : `Candidate ${i + 1}`, resumeText };
    })
    .filter((c) => c.resumeText);
}
