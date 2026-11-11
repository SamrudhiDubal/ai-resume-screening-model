const request = require('supertest');
const { createApp } = require('../src/app');
const { ResumeModel } = require('../src/model/predictor');
const { MAX_TEXT_LENGTH, MAX_CANDIDATES } = require('../src/validation');

const app = createApp({ model: ResumeModel.load() });

const DS_RESUME = 'Data scientist with 4 years of experience. Skills: python, machine learning, pandas, numpy, scikit-learn, sql.';
const WEB_RESUME = 'Frontend developer. Skills: javascript, react, redux, html, css, typescript, jest.';
const DS_JOB = 'We are hiring a Data Scientist. Required skills: python, machine learning, pandas, sql, statistics.';

describe('GET /api/health', () => {
  test('returns ok with model info', async () => {
    const res = await request(app).get('/api/health');
    expect(res.status).toBe(200);
    expect(res.body.status).toBe('ok');
    expect(res.body.model.classes).toContain('Data Science');
  });
});

describe('POST /api/screen', () => {
  test('classifies a resume', async () => {
    const res = await request(app).post('/api/screen').send({ resumeText: DS_RESUME });
    expect(res.status).toBe(200);
    expect(res.headers['content-type']).toMatch(/json/);
    expect(res.body).toMatchObject({ category: 'Data Science' });
    expect(res.body.confidence).toBeGreaterThan(0);
    expect(res.body.topCategories).toHaveLength(3);
    expect(res.body.skills).toContain('python');
  });

  test('honours topK', async () => {
    const res = await request(app).post('/api/screen').send({ resumeText: WEB_RESUME, topK: 5 });
    expect(res.status).toBe(200);
    expect(res.body.category).toBe('Web Development');
    expect(res.body.topCategories).toHaveLength(5);
  });

  test.each([
    [{}, /resumeText/],
    [{ resumeText: '' }, /resumeText/],
    [{ resumeText: '   ' }, /resumeText/],
    [{ resumeText: 42 }, /resumeText/],
    [{ resumeText: 'x'.repeat(MAX_TEXT_LENGTH + 1) }, /at most/],
    [{ resumeText: DS_RESUME, topK: 0 }, /topK/],
    [{ resumeText: DS_RESUME, topK: 'three' }, /topK/],
  ])('rejects invalid body %#', async (body, msg) => {
    const res = await request(app).post('/api/screen').send(body);
    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(msg);
  });

  test('rejects malformed JSON', async () => {
    const res = await request(app).post('/api/screen').set('Content-Type', 'application/json').send('{"resumeText": ');
    expect(res.status).toBe(400);
    expect(res.body.error).toBe('Malformed JSON body');
  });
});

describe('POST /api/match', () => {
  test('scores a matching resume higher than a non-matching one', async () => {
    const good = await request(app).post('/api/match').send({ resumeText: DS_RESUME, jobDescription: DS_JOB });
    const bad = await request(app).post('/api/match').send({ resumeText: WEB_RESUME, jobDescription: DS_JOB });
    expect(good.status).toBe(200);
    expect(good.body.score).toBeGreaterThan(bad.body.score);
    expect(good.body.matchedSkills).toEqual(expect.arrayContaining(['python', 'pandas', 'sql']));
    expect(good.body.missingSkills).toContain('statistics');
  });

  test('requires a job description', async () => {
    const res = await request(app).post('/api/match').send({ resumeText: DS_RESUME });
    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/jobDescription/);
  });
});

describe('POST /api/rank', () => {
  const candidates = [
    { id: 'web-1', resumeText: WEB_RESUME },
    { id: 'ds-1', resumeText: DS_RESUME },
  ];

  test('ranks candidates best first', async () => {
    const res = await request(app).post('/api/rank').send({ jobDescription: DS_JOB, candidates });
    expect(res.status).toBe(200);
    expect(res.body.total).toBe(2);
    expect(res.body.results.map((r) => r.id)).toEqual(['ds-1', 'web-1']);
    expect(res.body.results[0]).toMatchObject({ rank: 1, category: 'Data Science' });
  });

  test('limits results with topK and defaults ids to the index', async () => {
    const res = await request(app).post('/api/rank').send({
      jobDescription: DS_JOB, topK: 1, candidates: [{ resumeText: WEB_RESUME }, { resumeText: DS_RESUME }],
    });
    expect(res.body.results).toHaveLength(1);
    expect(res.body.results[0].id).toBe(1);
  });

  test.each([
    [{ jobDescription: DS_JOB }, /candidates/],
    [{ jobDescription: DS_JOB, candidates: [] }, /candidates/],
    [{ jobDescription: DS_JOB, candidates: [{ id: 1 }] }, /resumeText/],
    [{ jobDescription: DS_JOB, candidates: Array(MAX_CANDIDATES + 1).fill({ resumeText: 'x' }) }, /At most/],
    [{ candidates }, /jobDescription/],
  ])('rejects invalid body %#', async (body, msg) => {
    const res = await request(app).post('/api/rank').send(body);
    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(msg);
  });
});

describe('errors', () => {
  test('unknown routes return 404 JSON', async () => {
    const res = await request(app).get('/api/nope');
    expect(res.status).toBe(404);
    expect(res.body.error).toMatch(/not found/);
  });

  test('unexpected model failures return 500 without leaking details', async () => {
    const broken = { info: () => ({}), screen: () => { throw new Error('secret stack'); } };
    const spy = jest.spyOn(console, 'error').mockImplementation(() => {});
    const res = await request(createApp({ model: broken })).post('/api/screen').send({ resumeText: 'hi' });
    expect(res.status).toBe(500);
    expect(res.body.error).toBe('Internal server error');
    spy.mockRestore();
  });

  test('sends CORS headers for the frontend', async () => {
    const res = await request(app).get('/api/health').set('Origin', 'http://localhost:5173');
    expect(res.headers['access-control-allow-origin']).toBe('*');
  });
});
