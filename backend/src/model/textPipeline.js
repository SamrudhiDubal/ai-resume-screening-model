'use strict';

/**
 * JavaScript port of the preprocessing and TF-IDF transform from the Python notebook.
 * It must stay identical to `clean_text` and scikit-learn's TfidfVectorizer settings there,
 * which the parity tests in tests/predictor.test.js check.
 */

function cleanText(text, { stopWords, replacements }) {
  let t = String(text ?? '').toLowerCase();
  t = t.replace(/https?:\/\/\S+|www\.\S+/g, ' ');
  t = t.replace(/\S+@\S+/g, ' ');
  for (const [from, to] of replacements) t = t.split(from).join(to);
  t = t.replace(/[^a-z0-9\s]/g, ' ');
  t = t.replace(/\b\d+\b/g, ' ');
  return t
    .split(/\s+/)
    .filter((tok) => tok.length > 1 && !stopWords.has(tok))
    .join(' ');
}

function ngrams(tokens, [minN, maxN]) {
  const out = [];
  for (let n = minN; n <= maxN; n += 1) {
    for (let i = 0; i + n <= tokens.length; i += 1) out.push(tokens.slice(i, i + n).join(' '));
  }
  return out;
}

/** Returns a sparse, L2-normalised TF-IDF vector as a Map<featureIndex, weight>. */
function tfidfVector(cleaned, { vocabulary, idf, ngramRange, sublinearTf }) {
  const tokens = cleaned ? cleaned.split(' ').filter((tok) => tok.length >= 2) : [];
  const counts = new Map();
  for (const term of ngrams(tokens, ngramRange)) {
    const idx = vocabulary[term];
    if (idx !== undefined) counts.set(idx, (counts.get(idx) || 0) + 1);
  }
  const vec = new Map();
  let norm = 0;
  for (const [idx, tf] of counts) {
    const w = (sublinearTf ? 1 + Math.log(tf) : tf) * idf[idx];
    vec.set(idx, w);
    norm += w * w;
  }
  norm = Math.sqrt(norm);
  if (norm > 0) for (const [idx, w] of vec) vec.set(idx, w / norm);
  return vec;
}

function cosine(a, b) {
  // Both vectors are already L2-normalised, so the dot product is the cosine.
  const [small, large] = a.size <= b.size ? [a, b] : [b, a];
  let dot = 0;
  for (const [idx, w] of small) {
    const v = large.get(idx);
    if (v !== undefined) dot += w * v;
  }
  return dot;
}

function extractSkills(cleaned, skills) {
  const padded = ` ${cleaned} `;
  return skills.filter((s) => padded.includes(` ${s} `)).sort();
}

module.exports = { cleanText, ngrams, tfidfVector, cosine, extractSkills };
