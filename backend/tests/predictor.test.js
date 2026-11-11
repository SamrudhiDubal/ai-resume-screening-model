const path = require('path');
const { ResumeModel, softmax } = require('../src/model/predictor');
const parity = require('../model/parity_samples.json');

const model = ResumeModel.load(path.join(__dirname, '..', 'model', 'model.json'));

describe('softmax', () => {
  test('sums to 1 and is stable for large scores', () => {
    const p = softmax([1000, 1001, 1002]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    expect(p[2]).toBeGreaterThan(p[1]);
  });
});

describe('ResumeModel', () => {
  test('rejects an invalid model file', () => {
    expect(() => new ResumeModel({})).toThrow('Invalid model file');
  });

  test('exposes model info', () => {
    const info = model.info();
    expect(info.classes.length).toBeGreaterThan(1);
    expect(info.features).toBeGreaterThan(0);
    expect(info.metrics).toHaveProperty('accuracy');
  });

  test('probabilities cover every class and sum to 1', () => {
    const p = model.predictProba('python machine learning pandas');
    expect(p).toHaveLength(model.classes.length);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
  });

  test('screen returns a sorted top-k list and extracted skills', () => {
    const r = model.screen('Data scientist. Skills: python, machine learning, pandas, tensorflow, statistics.', 3);
    expect(r.category).toBe('Data Science');
    expect(r.topCategories).toHaveLength(3);
    expect(r.topCategories[0].probability).toBeGreaterThanOrEqual(r.topCategories[1].probability);
    expect(r.skills).toEqual(expect.arrayContaining(['python', 'machine learning', 'pandas']));
  });

  test('match scores a relevant resume above an irrelevant one', () => {
    const job = 'Hiring a DevOps engineer. Required skills: docker, kubernetes, aws, terraform, jenkins.';
    const good = model.match('DevOps engineer: docker, kubernetes, aws, terraform, jenkins, linux', job);
    const bad = model.match('HR executive: recruitment, onboarding, payroll', job);
    expect(good.score).toBeGreaterThan(bad.score);
    expect(good.skillMatch).toBe(1);
    expect(good.missingSkills).toEqual([]);
    expect(bad.matchedSkills).toEqual([]);
    for (const r of [good, bad]) {
      expect(r.score).toBeGreaterThanOrEqual(0);
      expect(r.score).toBeLessThanOrEqual(1);
    }
  });

  test('rank orders candidates by score and respects topK', () => {
    const job = 'Java developer with spring boot, hibernate, microservices and kafka';
    const ranked = model.rank(job, [
      { id: 'hr', resumeText: 'Recruiter skilled in onboarding and payroll' },
      { id: 'java', resumeText: 'Java developer: spring boot, hibernate, microservices, kafka, junit' },
      { id: 'web', resumeText: 'React developer: javascript, css, html' },
    ], 2);
    expect(ranked).toHaveLength(2);
    expect(ranked[0]).toMatchObject({ rank: 1, id: 'java', category: 'Java Developer' });
    expect(ranked[0].score).toBeGreaterThanOrEqual(ranked[1].score);
  });
});

// The JavaScript port must give the same answers as scikit-learn in the notebook.
describe('parity with the Python model', () => {
  test.each(parity.map((p, i) => [i, p]))('sample %i', (_i, sample) => {
    expect(model.clean(sample.text)).toBe(sample.clean);
    const r = model.screen(sample.text);
    expect(r.category).toBe(sample.category);
    expect(r.confidence).toBeCloseTo(sample.confidence, 3);
    const m = model.match(sample.text, sample.match_job);
    expect(m.score).toBeCloseTo(sample.match.score, 3);
    expect(m.cosine).toBeCloseTo(sample.match.cosine, 3);
    expect(m.matchedSkills).toEqual(sample.match.matched_skills);
  });
});
