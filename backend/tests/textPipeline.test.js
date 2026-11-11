const { cleanText, ngrams, tfidfVector, cosine, extractSkills } = require('../src/model/textPipeline');

const prep = { stopWords: new Set(['the', 'and', 'with', 'years', 'experience']), replacements: [['c++', ' cplusplus '], ['node.js', ' node ']] };

describe('cleanText', () => {
  test('lowercases and strips punctuation', () => {
    expect(cleanText('Python, SQL & Pandas!', prep)).toBe('python sql pandas');
  });

  test('removes URLs, emails, numbers and stop words', () => {
    const raw = 'Visit https://me.dev or www.me.dev, mail me@x.com. 5 years experience with the cloud';
    expect(cleanText(raw, prep)).toBe('visit or mail cloud');
  });

  test('applies replacements before stripping symbols', () => {
    expect(cleanText('C++ and Node.js', prep)).toBe('cplusplus node');
  });

  test('drops single-character tokens and handles empty input', () => {
    expect(cleanText('a b cd', prep)).toBe('cd');
    expect(cleanText('', prep)).toBe('');
    expect(cleanText(null, prep)).toBe('');
  });
});

describe('ngrams', () => {
  test('builds unigrams and bigrams', () => {
    expect(ngrams(['a', 'b', 'c'], [1, 2])).toEqual(['a', 'b', 'c', 'a b', 'b c']);
  });
});

describe('tfidfVector and cosine', () => {
  const tfidf = { vocabulary: { python: 0, sql: 1, java: 2 }, idf: [1, 2, 1], ngramRange: [1, 1], sublinearTf: true };

  test('produces a unit-length vector', () => {
    const v = tfidfVector('python sql sql', tfidf);
    const norm = Math.sqrt([...v.values()].reduce((s, w) => s + w * w, 0));
    expect(norm).toBeCloseTo(1, 10);
  });

  test('applies sublinear tf: 1 + ln(count)', () => {
    const v = tfidfVector('sql sql', tfidf);
    expect(v.get(1)).toBeCloseTo(1, 10); // a single feature normalises to 1
    const w = tfidfVector('python sql sql', tfidf);
    expect(w.get(1) / w.get(0)).toBeCloseTo((1 + Math.log(2)) * 2, 10);
  });

  test('ignores out-of-vocabulary words and returns an empty vector for empty text', () => {
    expect(tfidfVector('golang rust', tfidf).size).toBe(0);
    expect(tfidfVector('', tfidf).size).toBe(0);
  });

  test('cosine is 1 for identical text, 0 for disjoint text', () => {
    expect(cosine(tfidfVector('python sql', tfidf), tfidfVector('python sql', tfidf))).toBeCloseTo(1, 10);
    expect(cosine(tfidfVector('python', tfidf), tfidfVector('java', tfidf))).toBe(0);
  });
});

describe('extractSkills', () => {
  test('matches whole multi-word skills only', () => {
    const skills = ['machine learning', 'java', 'javascript', 'sql'];
    expect(extractSkills('python machine learning javascript', skills)).toEqual(['javascript', 'machine learning']);
  });
});
