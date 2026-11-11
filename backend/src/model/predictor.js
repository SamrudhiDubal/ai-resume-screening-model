'use strict';

const fs = require('fs');
const path = require('path');
const { cleanText, tfidfVector, cosine, extractSkills } = require('./textPipeline');

const DEFAULT_MODEL_PATH = path.join(__dirname, '..', '..', 'model', 'model.json');

const round = (x, d = 4) => Math.round(x * 10 ** d) / 10 ** d;

function softmax(scores) {
  const max = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/** Wraps the exported model.json and exposes screening and matching. */
class ResumeModel {
  constructor(raw) {
    if (!raw || !raw.tfidf || !raw.classifier) throw new Error('Invalid model file');
    this.raw = raw;
    this.prep = {
      stopWords: new Set(raw.preprocessing.stop_words),
      replacements: raw.preprocessing.replacements,
    };
    this.tfidf = {
      vocabulary: raw.tfidf.vocabulary,
      idf: raw.tfidf.idf,
      ngramRange: raw.tfidf.ngram_range,
      sublinearTf: raw.tfidf.sublinear_tf,
    };
    this.classes = raw.classifier.classes;
    this.coef = raw.classifier.coef;
    this.intercept = raw.classifier.intercept;
    this.skills = raw.skills;
    this.weights = raw.matching;
  }

  static load(filePath = process.env.MODEL_PATH || DEFAULT_MODEL_PATH) {
    return new ResumeModel(JSON.parse(fs.readFileSync(filePath, 'utf8')));
  }

  info() {
    return {
      version: this.raw.version,
      createdAt: this.raw.created_at,
      classes: this.classes,
      features: this.tfidf.idf.length,
      metrics: this.raw.metrics,
    };
  }

  clean(text) {
    return cleanText(text, this.prep);
  }

  vectorize(text) {
    return tfidfVector(this.clean(text), this.tfidf);
  }

  predictProba(text) {
    const x = this.vectorize(text);
    const scores = this.coef.map((row, k) => {
      let s = this.intercept[k];
      for (const [idx, w] of x) s += row[idx] * w;
      return s;
    });
    return softmax(scores);
  }

  /** Classify a resume into a job category. */
  screen(resumeText, topK = 3) {
    const probs = this.predictProba(resumeText);
    const ranked = this.classes
      .map((category, i) => ({ category, probability: round(probs[i]) }))
      .sort((a, b) => b.probability - a.probability);
    return {
      category: ranked[0].category,
      confidence: ranked[0].probability,
      topCategories: ranked.slice(0, topK),
      skills: extractSkills(this.clean(resumeText), this.skills),
    };
  }

  /** Score how well a resume fits a job description (0..1). */
  match(resumeText, jobDescription) {
    const rClean = this.clean(resumeText);
    const jClean = this.clean(jobDescription);
    const cos = cosine(tfidfVector(rClean, this.tfidf), tfidfVector(jClean, this.tfidf));
    const rSkills = new Set(extractSkills(rClean, this.skills));
    const jSkills = extractSkills(jClean, this.skills);
    const matched = jSkills.filter((s) => rSkills.has(s));
    const skillMatch = jSkills.length ? matched.length / jSkills.length : 0;
    return {
      score: round(this.weights.cosine_weight * cos + this.weights.skill_weight * skillMatch),
      cosine: round(cos),
      skillMatch: round(skillMatch),
      matchedSkills: matched,
      missingSkills: jSkills.filter((s) => !rSkills.has(s)),
    };
  }

  /** Rank candidates for a job, best first. */
  rank(jobDescription, candidates, topK = candidates.length) {
    return candidates
      .map((c) => ({
        id: c.id,
        ...this.match(c.resumeText, jobDescription),
        category: this.screen(c.resumeText, 1).category,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map((c, i) => ({ rank: i + 1, ...c }));
  }
}

module.exports = { ResumeModel, softmax, DEFAULT_MODEL_PATH };
