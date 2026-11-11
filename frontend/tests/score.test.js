import { describe, it, expect } from 'vitest';
import { formatPercent, scoreLabel, parseCandidates } from '../src/utils/score.js';

describe('formatPercent', () => {
  it('formats and clamps fractions', () => {
    expect(formatPercent(0.734)).toBe('73%');
    expect(formatPercent(1.5)).toBe('100%');
    expect(formatPercent(-1)).toBe('0%');
  });
  it('shows a dash for missing values', () => {
    expect(formatPercent(undefined)).toBe('–');
    expect(formatPercent(NaN)).toBe('–');
  });
});

describe('scoreLabel', () => {
  it.each([[0.8, 'Strong match'], [0.5, 'Strong match'], [0.35, 'Good match'], [0.2, 'Partial match'], [0.05, 'Weak match']])(
    '%f -> %s', (score, label) => expect(scoreLabel(score)).toBe(label));
});

describe('parseCandidates', () => {
  it('splits on --- lines and reads optional names', () => {
    const text = 'Name: Asha\nPython developer\n---\nJava developer\n  ---  \n\n';
    expect(parseCandidates(text)).toEqual([
      { id: 'Asha', resumeText: 'Python developer' },
      { id: 'Candidate 2', resumeText: 'Java developer' },
    ]);
  });
  it('drops empty blocks and handles empty input', () => {
    expect(parseCandidates('---\n---')).toEqual([]);
    expect(parseCandidates('')).toEqual([]);
    expect(parseCandidates('Name: Only a name')).toEqual([]);
  });
});
