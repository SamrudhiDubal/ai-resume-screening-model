'use strict';

const MAX_TEXT_LENGTH = 50000;
const MAX_CANDIDATES = 200;

class ValidationError extends Error {
  constructor(message) {
    super(message);
    this.status = 400;
  }
}

function requireText(body, field) {
  const value = body?.[field];
  if (typeof value !== 'string' || value.trim() === '') {
    throw new ValidationError(`"${field}" is required and must be a non-empty string`);
  }
  if (value.length > MAX_TEXT_LENGTH) {
    throw new ValidationError(`"${field}" must be at most ${MAX_TEXT_LENGTH} characters`);
  }
  return value;
}

function optionalTopK(body, fallback) {
  if (body?.topK === undefined) return fallback;
  const k = body.topK;
  if (!Number.isInteger(k) || k < 1) throw new ValidationError('"topK" must be a positive integer');
  return k;
}

function requireCandidates(body) {
  const list = body?.candidates;
  if (!Array.isArray(list) || list.length === 0) {
    throw new ValidationError('"candidates" must be a non-empty array');
  }
  if (list.length > MAX_CANDIDATES) {
    throw new ValidationError(`At most ${MAX_CANDIDATES} candidates per request`);
  }
  return list.map((c, i) => ({
    id: c?.id ?? i,
    resumeText: requireText(c, 'resumeText'),
  }));
}

module.exports = { ValidationError, requireText, optionalTopK, requireCandidates, MAX_TEXT_LENGTH, MAX_CANDIDATES };
