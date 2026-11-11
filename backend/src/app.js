'use strict';

const express = require('express');
const cors = require('cors');
const { ResumeModel } = require('./model/predictor');
const { requireText, optionalTopK, requireCandidates } = require('./validation');

/**
 * Builds the Express app. Pass `model` to inject a model (tests do this);
 * otherwise model/model.json is loaded.
 */
function createApp({ model = ResumeModel.load() } = {}) {
  const app = express();
  app.use(cors());
  app.use(express.json({ limit: '1mb' }));

  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', model: model.info() });
  });

  // Classify one resume into a job category.
  app.post('/api/screen', (req, res) => {
    const resumeText = requireText(req.body, 'resumeText');
    res.json(model.screen(resumeText, optionalTopK(req.body, 3)));
  });

  // Score one resume against one job description.
  app.post('/api/match', (req, res) => {
    const resumeText = requireText(req.body, 'resumeText');
    const jobDescription = requireText(req.body, 'jobDescription');
    res.json(model.match(resumeText, jobDescription));
  });

  // Rank many candidates for one job.
  app.post('/api/rank', (req, res) => {
    const jobDescription = requireText(req.body, 'jobDescription');
    const candidates = requireCandidates(req.body);
    const topK = optionalTopK(req.body, candidates.length);
    res.json({ jobDescription, total: candidates.length, results: model.rank(jobDescription, candidates, topK) });
  });

  app.use((req, res) => res.status(404).json({ error: `Route ${req.method} ${req.path} not found` }));

  // eslint-disable-next-line no-unused-vars
  app.use((err, req, res, next) => {
    if (err.type === 'entity.parse.failed') return res.status(400).json({ error: 'Malformed JSON body' });
    if (err.type === 'entity.too.large') return res.status(413).json({ error: 'Request body too large' });
    const status = err.status || 500;
    if (status >= 500) console.error(err);
    return res.status(status).json({ error: status >= 500 ? 'Internal server error' : err.message });
  });

  return app;
}

module.exports = { createApp };
