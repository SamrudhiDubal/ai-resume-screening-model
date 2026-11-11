"""Builds resume_screening.ipynb and resume_screening.py from one list of cells.

Run:  python build_notebook.py
Edit the cells here (not the generated files) so the notebook and the script stay identical.
"""
import json
from pathlib import Path

CELLS = []


def md(src):
    CELLS.append(("markdown", src.strip("\n")))


def code(src):
    CELLS.append(("code", src.strip("\n")))


# ---------------------------------------------------------------------------
md("""
# AI-Based Resume Screening & Candidate Matching

This notebook trains the ML model behind the job portal. It does two jobs:

1. **Resume classification**: predict a resume's job category (Data Science, Web Development, HR, ...)
   with TF-IDF features and a supervised classifier.
2. **Candidate matching**: score how well a resume fits a job description using
   TF-IDF cosine similarity plus skill overlap, then rank candidates for each job.

Stack: pandas, NumPy, Matplotlib, Seaborn, scikit-learn. Runs as-is in Google Colab.

**Data.** By default the notebook generates a labelled sample dataset so it runs with no downloads.
To use a real dataset, upload a CSV with columns `Category` and `Resume`
(for example the public Kaggle *UpdatedResumeDataSet.csv*) when the upload cell asks, or put it next to the notebook as `resumes.csv`.

The last section exports `model.json`, which the Node/Express API loads to serve predictions.
""")

md("## 1. Setup")
code("""
# Colab already has these; the line is here for local runs.
# !pip install -q pandas numpy matplotlib seaborn scikit-learn joblib

import json
import os
import random
import re
import time
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_score,
                                     learning_curve, train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 100
os.makedirs("outputs", exist_ok=True)
print("Libraries loaded.")
""")

md("""
## 2. Load data

Priority: an uploaded CSV in Colab, then `resumes.csv` / `data/resumes.csv` on disk, then a generated sample.
The generator builds realistic resumes from per-category skill and title pools and mixes in
skills from other categories, so the task is not trivially separable.
""")
code("""
CATEGORY_PROFILES = {
    "Data Science": {
        "titles": ["Data Scientist", "Machine Learning Engineer", "Data Analyst", "AI Engineer"],
        "skills": ["python", "machine learning", "deep learning", "pandas", "numpy", "scikit-learn",
                   "tensorflow", "pytorch", "sql", "statistics", "nlp", "data visualization",
                   "matplotlib", "seaborn", "tableau", "feature engineering", "regression", "jupyter"],
        "tasks": ["built predictive models to forecast customer churn",
                  "trained classification models with cross validation and hyperparameter tuning",
                  "cleaned and analysed large datasets to find business insights",
                  "deployed machine learning models as REST APIs",
                  "designed A/B tests and reported statistical significance"],
    },
    "Web Development": {
        "titles": ["Frontend Developer", "Full Stack Developer", "Web Developer", "React Developer"],
        "skills": ["javascript", "react", "node", "express", "html", "css", "typescript", "mongodb",
                   "rest api", "redux", "tailwind", "git", "next", "vite", "jest", "responsive design"],
        "tasks": ["developed responsive single page applications in react",
                  "built REST APIs with node and express backed by mongodb",
                  "improved page load time through code splitting and caching",
                  "wrote unit and integration tests for frontend components",
                  "collaborated with designers to implement pixel perfect user interfaces"],
    },
    "Java Developer": {
        "titles": ["Java Developer", "Backend Engineer", "Software Engineer", "J2EE Developer"],
        "skills": ["java", "spring boot", "hibernate", "microservices", "maven", "jpa", "mysql",
                   "junit", "rest api", "kafka", "oop", "design patterns", "jdbc", "tomcat"],
        "tasks": ["designed microservices using spring boot and kafka",
                  "optimised hibernate queries to reduce database latency",
                  "wrote junit tests to keep code coverage above eighty percent",
                  "maintained enterprise J2EE applications deployed on tomcat",
                  "applied object oriented design patterns to refactor legacy modules"],
    },
    "DevOps Engineer": {
        "titles": ["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer", "Platform Engineer"],
        "skills": ["docker", "kubernetes", "aws", "terraform", "jenkins", "ci cd", "linux", "ansible",
                   "bash", "prometheus", "grafana", "azure", "helm", "git", "monitoring"],
        "tasks": ["automated infrastructure provisioning with terraform on aws",
                  "built ci cd pipelines in jenkins for dozens of services",
                  "migrated workloads to kubernetes clusters with helm charts",
                  "set up monitoring and alerting with prometheus and grafana",
                  "reduced cloud costs by right sizing instances"],
    },
    "HR": {
        "titles": ["HR Executive", "Talent Acquisition Specialist", "HR Manager", "Recruiter"],
        "skills": ["recruitment", "onboarding", "employee relations", "payroll", "hrms",
                   "performance management", "talent acquisition", "labour law", "interviewing",
                   "communication", "training and development", "compliance"],
        "tasks": ["managed end to end recruitment for technical and non technical roles",
                  "designed onboarding programs for new employees",
                  "handled payroll processing and statutory compliance",
                  "resolved employee relations issues and grievances",
                  "ran performance appraisal cycles across departments"],
    },
    "Testing": {
        "titles": ["QA Engineer", "Test Automation Engineer", "Software Tester", "SDET"],
        "skills": ["selenium", "manual testing", "automation testing", "test cases", "jira",
                   "regression testing", "api testing", "postman", "cucumber", "testng",
                   "bug tracking", "agile", "performance testing", "jmeter"],
        "tasks": ["wrote and executed test cases for web and mobile applications",
                  "built selenium automation frameworks with testng",
                  "performed api testing with postman and rest assured",
                  "logged and tracked defects in jira through the release cycle",
                  "ran load tests with jmeter before major releases"],
    },
    "Business Analyst": {
        "titles": ["Business Analyst", "Product Analyst", "Functional Consultant", "BA Lead"],
        "skills": ["requirements gathering", "stakeholder management", "sql", "excel", "power bi",
                   "user stories", "process mapping", "agile", "jira", "uml", "gap analysis",
                   "documentation", "data analysis"],
        "tasks": ["gathered business requirements from stakeholders and wrote user stories",
                  "created process maps and gap analysis for system upgrades",
                  "built power bi dashboards for leadership reporting",
                  "acted as liaison between business teams and developers",
                  "prioritised backlog items with product owners in agile sprints"],
    },
    "Mechanical Engineer": {
        "titles": ["Mechanical Engineer", "Design Engineer", "Production Engineer", "CAD Engineer"],
        "skills": ["autocad", "solidworks", "catia", "ansys", "manufacturing", "gd&t", "thermodynamics",
                   "quality control", "lean manufacturing", "six sigma", "cnc", "product design"],
        "tasks": ["designed mechanical components in solidworks and catia",
                  "ran finite element analysis in ansys to validate designs",
                  "improved production line efficiency with lean manufacturing",
                  "prepared engineering drawings with gd&t annotations",
                  "led root cause analysis for quality control issues"],
    },
    "Network Security": {
        "titles": ["Security Analyst", "Network Engineer", "Cyber Security Engineer", "SOC Analyst"],
        "skills": ["firewall", "siem", "penetration testing", "vulnerability assessment", "networking",
                   "tcp ip", "cisco", "ids ips", "incident response", "linux", "wireshark",
                   "cryptography", "iso 27001"],
        "tasks": ["monitored security events in a siem and triaged alerts",
                  "performed vulnerability assessment and penetration testing",
                  "configured cisco firewalls routers and switches",
                  "led incident response for phishing and malware attacks",
                  "prepared documentation for iso 27001 audits"],
    },
    "Sales": {
        "titles": ["Sales Executive", "Business Development Manager", "Account Manager", "Sales Lead"],
        "skills": ["lead generation", "negotiation", "crm", "salesforce", "b2b sales", "client relationship",
                   "cold calling", "market research", "communication", "target achievement",
                   "presentation", "account management"],
        "tasks": ["exceeded quarterly sales targets through b2b lead generation",
                  "managed key client accounts and renewals in salesforce",
                  "negotiated contracts with enterprise customers",
                  "ran cold calling campaigns to open new territories",
                  "presented product demos to prospective clients"],
    },
}

DEGREES = ["B.Tech in Computer Science", "B.E. in Information Technology", "MBA", "B.Sc",
           "M.Tech", "B.E. in Mechanical Engineering", "BBA", "M.Sc in Statistics", "B.Com"]
SOFT = ["team player", "problem solving", "leadership", "time management", "attention to detail"]


RELATED = {
    "Data Science": ["Business Analyst", "Web Development"],
    "Web Development": ["Java Developer", "Testing"],
    "Java Developer": ["Web Development", "Testing"],
    "DevOps Engineer": ["Network Security", "Java Developer"],
    "HR": ["Sales", "Business Analyst"],
    "Testing": ["Web Development", "Java Developer"],
    "Business Analyst": ["Data Science", "Sales"],
    "Mechanical Engineer": ["Business Analyst", "Sales"],
    "Network Security": ["DevOps Engineer", "Testing"],
    "Sales": ["HR", "Business Analyst"],
}


def make_resume(category, rng):
    p = CATEGORY_PROFILES[category]
    others = [c for c in CATEGORY_PROFILES if c != category]
    skills = rng.sample(p["skills"], k=rng.randint(3, 7))
    # Noise: skills from related (career-adjacent) and random categories, so classes overlap
    # the way real resumes do (a QA engineer who knows react, a BA who writes SQL).
    for oc in RELATED[category] + rng.sample(others, k=rng.randint(0, 2)):
        if rng.random() < 0.7:
            skills += rng.sample(CATEGORY_PROFILES[oc]["skills"], k=rng.randint(1, 4))
    skills = list(dict.fromkeys(skills))
    rng.shuffle(skills)
    tasks = rng.sample(p["tasks"], k=rng.randint(1, 3))
    if rng.random() < 0.5:  # a past role in an adjacent field
        tasks.append(rng.choice(CATEGORY_PROFILES[rng.choice(RELATED[category])]["tasks"]))
    rng.shuffle(tasks)
    title_cat = category if rng.random() > 0.2 else rng.choice(RELATED[category])
    years = rng.randint(0, 12)
    return (f"{rng.choice(CATEGORY_PROFILES[title_cat]['titles'])} with {years} years of experience. "
            f"Skills: {', '.join(skills)}. "
            f"Experience: {'. '.join(t.capitalize() for t in tasks)}. "
            f"Education: {rng.choice(DEGREES)}. "
            f"Strengths: {', '.join(rng.sample(SOFT, 2))}. Contact: candidate{rng.randint(100, 999)}@mail.com")


def make_job(category, rng, i):
    p = CATEGORY_PROFILES[category]
    title = rng.choice(p["titles"])
    must = rng.sample(p["skills"], k=6)
    return {
        "job_id": f"J{i:03d}",
        "title": title,
        "category": category,
        "description": (f"We are hiring a {title}. Required skills: {', '.join(must)}. "
                        f"Responsibilities: {'. '.join(rng.sample(p['tasks'], 2))}. "
                        f"{rng.randint(1, 6)}+ years of experience preferred."),
    }


def generate_sample_data(n_per_class=60, jobs_per_class=3, seed=RANDOM_STATE):
    rng = random.Random(seed)
    rows = [{"Category": c, "Resume": make_resume(c, rng)}
            for c in CATEGORY_PROFILES for _ in range(n_per_class)]
    # Uneven class sizes, like real data.
    rows = [r for r in rows if rng.random() > 0.15]
    jobs, i = [], 1
    for c in CATEGORY_PROFILES:
        for _ in range(jobs_per_class):
            jobs.append(make_job(c, rng, i))
            i += 1
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True), pd.DataFrame(jobs)
""")
code("""
def load_resumes():
    # 1) Colab upload (skip by pressing Cancel; the sample data is used instead)
    try:
        from google.colab import files  # noqa: F401
        print("Optional: upload a CSV with columns Category, Resume. Cancel to use the sample data.")
        try:
            uploaded = files.upload()
        except Exception:
            uploaded = {}
        for name in uploaded:
            if name.endswith(".csv"):
                print(f"Using uploaded file {name}")
                return pd.read_csv(name), "uploaded"
    except ImportError:
        pass
    # 2) Local file
    for path in ["resumes.csv", "data/resumes.csv", "UpdatedResumeDataSet.csv"]:
        if os.path.exists(path):
            print(f"Using {path}")
            return pd.read_csv(path), path
    # 3) Generated sample
    print("No CSV found: generating the sample dataset.")
    return None, "generated"


df, source = load_resumes()
sample_df, jobs_df = generate_sample_data()
if df is None:
    df = sample_df
if os.path.exists("data/jobs.csv"):
    jobs_df = pd.read_csv("data/jobs.csv")

df = df[["Category", "Resume"]].dropna().drop_duplicates().reset_index(drop=True)
print(f"Source: {source} | resumes: {len(df)} | categories: {df['Category'].nunique()} | jobs: {len(jobs_df)}")
df.head()
""")

md("## 3. Exploratory data analysis")
code("""
df.info()
print()
print(df["Category"].value_counts())
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
order = df["Category"].value_counts().index
sns.countplot(data=df, y="Category", order=order, ax=axes[0])
axes[0].set_title("Resumes per category")
axes[0].set_xlabel("Count")

counts = df["Category"].value_counts()
axes[1].pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
            colors=sns.color_palette("deep", len(counts)))
axes[1].set_title("Category share")
plt.tight_layout()
plt.savefig("outputs/category_distribution.png", bbox_inches="tight")
plt.show()
""")
code("""
df["char_len"] = df["Resume"].str.len()
df["word_count"] = df["Resume"].str.split().str.len()
print(df[["char_len", "word_count"]].describe().round(1))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(df["word_count"], bins=30, kde=True, ax=axes[0])
axes[0].set_title("Resume length (words)")
sns.boxplot(data=df, x="word_count", y="Category", order=order, ax=axes[1])
axes[1].set_title("Resume length by category")
plt.tight_layout()
plt.savefig("outputs/resume_length.png", bbox_inches="tight")
plt.show()
""")

md("""
## 4. Text preprocessing

The cleaning rules are deliberately simple and deterministic, because the Node API re-implements them
(`backend/src/model/textPipeline.js`). If you change them here, change them there too.
""")
code("""
STOP_WORDS = set('''
a about above after again against all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not
now of off on once only or other our ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to too under until up very was we
were what when where which while who whom why will with would you your yours yourself yourselves
also etc using used use within per via years year experience
'''.split())

REPLACEMENTS = [("c++", " cplusplus "), ("c#", " csharp "), (".net", " dotnet "),
                ("node.js", " node "), ("nodejs", " node "), ("react.js", " react "),
                ("next.js", " next "), ("gd&t", " gdt "), ("ci/cd", " ci cd ")]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\\S+|www\\.\\S+", " ", text)   # URLs
    text = re.sub(r"\\S+@\\S+", " ", text)                    # emails
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)                # punctuation and symbols
    text = re.sub(r"\\b\\d+\\b", " ", text)                    # standalone numbers
    tokens = [t for t in text.split() if len(t) > 1 and t not in STOP_WORDS]
    return " ".join(tokens)


df["clean"] = df["Resume"].apply(clean_text)
jobs_df["clean"] = jobs_df["description"].apply(clean_text)
print("RAW:  ", df.loc[0, "Resume"][:200])
print("CLEAN:", df.loc[0, "clean"][:200])
""")
code("""
word_freq = Counter(" ".join(df["clean"]).split())
top = pd.DataFrame(word_freq.most_common(25), columns=["word", "count"])
plt.figure(figsize=(12, 6))
sns.barplot(data=top, x="count", y="word")
plt.title("Top 25 words after cleaning")
plt.tight_layout()
plt.savefig("outputs/top_words.png", bbox_inches="tight")
plt.show()
""")

md("""
## 5. Skill extraction

A curated skill vocabulary (in cleaned form) is used to extract skills from resumes and jobs.
Skill overlap is part of the match score and makes the results explainable to recruiters.
""")
code("""
SKILLS = sorted({clean_text(s) for p in CATEGORY_PROFILES.values() for s in p["skills"]} - {""})


def extract_skills(clean, skills=SKILLS):
    padded = f" {clean} "
    return sorted(s for s in skills if f" {s} " in padded)


df["skills"] = df["clean"].apply(extract_skills)
jobs_df["skills"] = jobs_df["clean"].apply(extract_skills)
df["n_skills"] = df["skills"].str.len()
print(f"{len(SKILLS)} skills in vocabulary; mean skills per resume: {df['n_skills'].mean():.1f}")

skill_counts = Counter(s for lst in df["skills"] for s in lst).most_common(20)
plt.figure(figsize=(12, 6))
sns.barplot(x=[c for _, c in skill_counts], y=[s for s, _ in skill_counts])
plt.title("20 most common skills")
plt.xlabel("Resumes mentioning skill")
plt.tight_layout()
plt.savefig("outputs/top_skills.png", bbox_inches="tight")
plt.show()
""")

md("## 6. Train / test split and TF-IDF features")
code("""
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["Category"], test_size=0.2, stratify=df["Category"], random_state=RANDOM_STATE)

TFIDF_PARAMS = dict(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.9,
                    max_features=5000, lowercase=False, token_pattern=r"(?u)\\b\\w\\w+\\b")
print(f"Train: {len(X_train)}  Test: {len(X_test)}")
""")

md("## 7. Compare classifiers (5-fold cross validation)")
code("""
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=10),
    "Linear SVM": LinearSVC(C=1.0),
    "Multinomial NB": MultinomialNB(alpha=0.1),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "KNN": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = []
for name, clf in models.items():
    pipe = Pipeline([("tfidf", TfidfVectorizer(**TFIDF_PARAMS)), ("clf", clf)])
    t0 = time.time()
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    results.append({"model": name, "cv_f1_mean": scores.mean(), "cv_f1_std": scores.std(),
                    "seconds": time.time() - t0})
results_df = pd.DataFrame(results).sort_values("cv_f1_mean", ascending=False)
print(results_df.round(4).to_string(index=False))

plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="cv_f1_mean", y="model")
plt.errorbar(results_df["cv_f1_mean"], range(len(results_df)), xerr=results_df["cv_f1_std"],
             fmt="none", c="black", capsize=4)
plt.xlim(0, 1.05)
plt.title("Cross-validated macro F1 by model")
plt.xlabel("Macro F1")
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", bbox_inches="tight")
plt.show()
""")

md("""
## 8. Tune the production model

Logistic Regression is used in production: it is close to the best score, gives calibrated
probabilities (a confidence for each category), and its weights export cleanly to JSON for the Node API.
""")
code("""
pipe = Pipeline([("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
                 ("clf", LogisticRegression(max_iter=3000))])
grid = GridSearchCV(pipe, {"clf__C": [1, 5, 10, 30], "tfidf__ngram_range": [(1, 1), (1, 2)]},
                    cv=cv, scoring="f1_macro", n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print(f"Best CV macro F1: {grid.best_score_:.4f}")
best_model = grid.best_estimator_
""")

md("## 9. Evaluate on the held-out test set")
code("""
y_pred = best_model.predict(X_test)
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, y_pred, average="macro"),
}
for k, v in metrics.items():
    print(f"{k:>16}: {v:.4f}")
print()
print(classification_report(y_test, y_pred, zero_division=0))
""")
code("""
labels = sorted(df["Category"].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion matrix (test set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")
plt.show()
""")
code("""
report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).T
report = report.loc[labels, ["precision", "recall", "f1-score"]]
report.plot(kind="barh", figsize=(11, 7), xlim=(0, 1.05))
plt.title("Per-category precision, recall and F1")
plt.tight_layout()
plt.savefig("outputs/per_class_metrics.png", bbox_inches="tight")
plt.show()
""")
code("""
sizes, train_scores, val_scores = learning_curve(
    best_model, df["clean"], df["Category"], cv=cv, scoring="f1_macro",
    train_sizes=np.linspace(0.2, 1.0, 6), n_jobs=-1)
plt.figure(figsize=(9, 5))
plt.plot(sizes, train_scores.mean(1), "o-", label="Train")
plt.plot(sizes, val_scores.mean(1), "o-", label="Validation")
plt.fill_between(sizes, val_scores.mean(1) - val_scores.std(1), val_scores.mean(1) + val_scores.std(1), alpha=0.2)
plt.title("Learning curve (Logistic Regression)")
plt.xlabel("Training resumes")
plt.ylabel("Macro F1")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/learning_curve.png", bbox_inches="tight")
plt.show()
""")
code("""
# Most influential terms per category: what the model actually learned.
vec = best_model.named_steps["tfidf"]
clf = best_model.named_steps["clf"]
terms = np.array(vec.get_feature_names_out())
n = len(clf.classes_)
fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(14, 3 * ((n + 1) // 2)))
for ax, idx in zip(axes.ravel(), range(n)):
    top_idx = np.argsort(clf.coef_[idx])[-8:]
    ax.barh(terms[top_idx], clf.coef_[idx][top_idx])
    ax.set_title(clf.classes_[idx])
for ax in axes.ravel()[n:]:
    ax.axis("off")
plt.suptitle("Top weighted terms per category", y=1.0)
plt.tight_layout()
plt.savefig("outputs/top_terms_per_class.png", bbox_inches="tight")
plt.show()
""")

md("""
## 10. Resume-to-job matching

`match_score = 0.6 * cosine(TF-IDF(resume), TF-IDF(job)) + 0.4 * (job skills found in resume / job skills)`

Cosine similarity captures overall wording, the skill term rewards the exact requirements. Both are in [0, 1].
""")
code("""
COSINE_WEIGHT, SKILL_WEIGHT = 0.6, 0.4


def match_resume_to_job(resume_text, job_text, vectorizer=vec):
    r_clean, j_clean = clean_text(resume_text), clean_text(job_text)
    vecs = vectorizer.transform([r_clean, j_clean])
    cos = float(cosine_similarity(vecs[0], vecs[1])[0, 0])
    r_sk, j_sk = set(extract_skills(r_clean)), set(extract_skills(j_clean))
    skill = len(r_sk & j_sk) / len(j_sk) if j_sk else 0.0
    return {"score": round(COSINE_WEIGHT * cos + SKILL_WEIGHT * skill, 4),
            "cosine": round(cos, 4), "skill_match": round(skill, 4),
            "matched_skills": sorted(r_sk & j_sk), "missing_skills": sorted(j_sk - r_sk)}


def rank_candidates(job_text, resumes, top_k=5):
    rows = [{"candidate": i, **match_resume_to_job(r, job_text)} for i, r in enumerate(resumes)]
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k)


job = jobs_df.iloc[0]
print(f"Job {job.job_id}: {job.title} ({job.category})\\n{job.description}\\n")
test_resumes = df.loc[X_test.index].reset_index(drop=True)
ranked = rank_candidates(job.description, test_resumes["Resume"].tolist())
ranked["true_category"] = test_resumes.loc[ranked["candidate"], "Category"].values
ranked[["candidate", "true_category", "score", "cosine", "skill_match", "matched_skills"]]
""")
code("""
# Evaluate ranking: a candidate is "relevant" to a job when the categories agree.
def precision_at_k(job_row, k):
    ranked = rank_candidates(job_row.description, test_resumes["Resume"].tolist(), top_k=k)
    return (test_resumes.loc[ranked["candidate"], "Category"].values == job_row.category).mean()


for k in (1, 3, 5):
    p = np.mean([precision_at_k(j, k) for j in jobs_df.itertuples()])
    print(f"Mean precision@{k} over {len(jobs_df)} jobs: {p:.3f}")
""")
code("""
# Mean match score between every resume category and every job category: a strong diagonal is good.
score_rows = []
for j in jobs_df.itertuples():
    for r in test_resumes.itertuples():
        score_rows.append({"job_category": j.category, "resume_category": r.Category,
                           "score": match_resume_to_job(r.Resume, j.description)["score"]})
heat = pd.DataFrame(score_rows).pivot_table(index="resume_category", columns="job_category", values="score")
plt.figure(figsize=(11, 9))
sns.heatmap(heat, annot=True, fmt=".2f", cmap="viridis")
plt.title("Mean match score: resume category vs job category")
plt.tight_layout()
plt.savefig("outputs/match_heatmap.png", bbox_inches="tight")
plt.show()
""")

md("## 11. Try it on your own resume")
code("""
my_resume = '''Full stack developer with 3 years of experience building React and Node.js apps.
Skills: JavaScript, TypeScript, React, Redux, Node.js, Express, MongoDB, REST API, Jest, Git.
Built a job portal with resume upload and recruiter dashboards.'''

probs = best_model.predict_proba([clean_text(my_resume)])[0]
top3 = sorted(zip(best_model.classes_, probs), key=lambda x: -x[1])[:3]
print("Predicted category:", top3[0][0])
for c, p in top3:
    print(f"  {c:<20} {p:.3f}")
print()
print(match_resume_to_job(my_resume, jobs_df[jobs_df.category == "Web Development"].iloc[0].description))
""")

md("""
## 12. Export the model

- `outputs/resume_model.joblib`: the full scikit-learn pipeline, for Python services.
- `outputs/model.json`: vocabulary, IDF weights, classifier weights, skills and matching weights,
  for the Node/Express API (`backend/model/model.json`). It contains everything needed to reproduce
  predictions without Python.
- `outputs/parity_samples.json`: a few resumes with their Python predictions, used by the backend tests to
  check that the JavaScript implementation gives the same answers.
""")
code("""
joblib.dump(best_model, "outputs/resume_model.joblib")

vocab = vec.vocabulary_
export = {
    "version": 1,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "preprocessing": {"stop_words": sorted(STOP_WORDS), "replacements": REPLACEMENTS},
    "tfidf": {
        "ngram_range": list(vec.ngram_range),
        "sublinear_tf": vec.sublinear_tf,
        "vocabulary": {t: int(i) for t, i in vocab.items()},
        "idf": [round(float(x), 6) for x in vec.idf_],
    },
    "classifier": {
        "type": "logistic_regression",
        "classes": clf.classes_.tolist(),
        "coef": [[round(float(w), 6) for w in row] for row in clf.coef_],
        "intercept": [round(float(b), 6) for b in clf.intercept_],
    },
    "skills": SKILLS,
    "matching": {"cosine_weight": COSINE_WEIGHT, "skill_weight": SKILL_WEIGHT},
    "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
}
with open("outputs/model.json", "w") as f:
    json.dump(export, f)

parity = []
for text in df["Resume"].sample(15, random_state=1).tolist() + [my_resume]:
    p = best_model.predict_proba([clean_text(text)])[0]
    parity.append({"text": text, "clean": clean_text(text), "category": best_model.classes_[p.argmax()],
                   "confidence": round(float(p.max()), 4)})
parity_job = jobs_df.iloc[0].description
for item in parity:
    item["match_job"] = parity_job
    item["match"] = match_resume_to_job(item["text"], parity_job)
with open("outputs/parity_samples.json", "w") as f:
    json.dump(parity, f, indent=1)

df[["Category", "Resume"]].to_csv("outputs/resumes.csv", index=False)
jobs_df[["job_id", "title", "category", "description"]].to_csv("outputs/jobs.csv", index=False)
print("Saved:", sorted(os.listdir("outputs")))
print(f"model.json size: {os.path.getsize('outputs/model.json') / 1024:.0f} KB")
""")
code("""
# In Colab: download the exported files, then copy model.json into backend/model/.
try:
    from google.colab import files
    for name in ["model.json", "resume_model.joblib", "parity_samples.json"]:
        files.download(f"outputs/{name}")
except ImportError:
    print("Not in Colab: files are in ./outputs")
""")


# ---------------------------------------------------------------------------
def build():
    here = Path(__file__).parent
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if t == "code" else {})}
            for t, s in CELLS
        ],
        "metadata": {
            "colab": {"provenance": [], "name": "resume_screening.ipynb"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    (here / "resume_screening.ipynb").write_text(json.dumps(nb, indent=1))
    script = ['"""AI resume screening and candidate matching. Generated from build_notebook.py."""']
    for t, s in CELLS:
        if t == "markdown":
            script.append("\n".join("# " + line if line else "#" for line in s.splitlines()))
        else:
            # Drop notebook-only trailing expressions that just display a DataFrame.
            lines = s.splitlines()
            if lines and re.fullmatch(r"[\w\[\]\"', .()]+", lines[-1]) and not lines[-1].startswith(("print", "plt")):
                lines[-1] = f"print({lines[-1]})"
            script.append("\n".join(lines))
    (here / "resume_screening.py").write_text("\n\n\n".join(script) + "\n")
    print("Wrote resume_screening.ipynb and resume_screening.py")


import re  # noqa: E402

if __name__ == "__main__":
    build()
