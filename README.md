# AI Resume Screening & Candidate Matching

The ML model for the job portal, plus an Express API that serves it and a React UI, each with tests.

```
ml/          Google Colab notebook (pandas, NumPy, Matplotlib, Seaborn, scikit-learn)
backend/     Node/Express API serving the model            -> Jest + Supertest
frontend/    React (Vite) screening and ranking UI         -> Vitest + Testing Library
```

## What the model does

1. **Resume classification**: TF-IDF features + Logistic Regression predict a resume's category
   (10 categories in the sample data: Data Science, Web Development, Java Developer, DevOps, HR, Testing,
   Business Analyst, Mechanical Engineer, Network Security, Sales), with a confidence per category.
   The notebook compares Logistic Regression, Linear SVM, Naive Bayes, Random Forest and KNN with
   5-fold cross validation, then tunes the chosen model with GridSearchCV.
2. **Candidate matching**: `score = 0.6 x cosine(TF-IDF resume, TF-IDF job) + 0.4 x share of the job's
   required skills found in the resume`. It returns matched and missing skills, and ranks candidates for a job.

On the bundled sample data: test accuracy and macro F1 are about 0.99, and ranking precision@5 is about 0.97.
The sample is synthetic, so expect lower numbers on real resumes (the notebook works unchanged with the
Kaggle *UpdatedResumeDataSet.csv*, which has the same `Category`, `Resume` columns).

## 1. Train the model in Google Colab

1. Open https://colab.research.google.com, then File > Upload notebook > `ml/resume_screening.ipynb`.
2. Runtime > Run all.
3. The upload cell asks for an optional CSV with columns `Category`, `Resume`. Press **Cancel** to use the
   built-in sample dataset (generated in the notebook, so nothing needs downloading).
4. The last cell downloads `model.json`, `resume_model.joblib` and `parity_samples.json`.
   Copy `model.json` and `parity_samples.json` into `backend/model/` (already done for the sample data).

Plots produced: category distribution, resume length, top words, top skills, model comparison, confusion matrix,
per-class metrics, learning curve, top terms per category, and a resume-vs-job match heatmap.

To run locally instead: `cd ml && pip install -r requirements.txt && python resume_screening.py`.
`resume_screening.py` and the notebook are both generated from `build_notebook.py`, so edit that file and run
`python build_notebook.py` to keep them in sync.

## 2. Run the API

```bash
cd backend
npm install
npm start          # http://localhost:5000
npm test           # Jest + Supertest
```

The API runs the model in plain JavaScript from `model.json` (no Python at runtime). The parity tests check that
it gives the same categories, confidences and match scores as scikit-learn.

| Method | Route | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | | model version, classes, metrics |
| POST | `/api/screen` | `{ resumeText, topK? }` | `category`, `confidence`, `topCategories`, `skills` |
| POST | `/api/match` | `{ resumeText, jobDescription }` | `score`, `cosine`, `skillMatch`, `matchedSkills`, `missingSkills` |
| POST | `/api/rank` | `{ jobDescription, candidates: [{ id, resumeText }], topK? }` | ranked `results` |

Invalid input returns 400 with an `error` message. `MODEL_PATH` overrides the model file location.

## 3. Run the frontend

```bash
cd frontend
npm install
cp .env.example .env   # VITE_API_URL=http://localhost:5000
npm run dev            # http://localhost:5173
npm test               # Vitest
```

## Using it in the job portal

Call `/api/screen` when a candidate uploads a resume (store the category and skills on their profile),
and `/api/rank` from the recruiter's job page to list the best applicants. To read PDF or DOCX resumes,
extract the text first (for example with `pdf-parse` or `mammoth` in Node) and send the text.

## Retraining

When you change the cleaning rules in the notebook (`clean_text`, `STOP_WORDS`, `REPLACEMENTS`), update
`backend/src/model/textPipeline.js` to match; the parity tests will fail if the two drift apart.
