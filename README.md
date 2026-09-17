# Machine Learning: Zero to Expert

An interactive, visual HTML textbook that takes a working professional from beginner to expert in applied machine learning. 24 chapters across four levels, with hand-drawn SVG figures, interactive widgets, runnable Python, self-grading quizzes, and exercises.

## Read it

Open `index.html` in any browser. No build step, no server, no internet required. Progress and theme are stored in your browser.

## Structure

| Part | Level | Chapters |
|---|---|---|
| I · Foundations | Beginner | 01 What ML is · 02 Data · 03 Math · 04 Python toolkit · 05 Linear regression · 06 Classification & metrics · 07 Overfitting & validation |
| II · The Core Toolkit | Intermediate | 08 Feature engineering · 09 Trees · 10 Forests & boosting · 11 kNN/SVM/NB · 12 Clustering & PCA · 13 Tuning, imbalance, calibration · 14 Time series |
| III · Deep Learning | Advanced | 15 Neural nets from scratch · 16 Training with PyTorch · 17 CNNs · 18 Attention & Transformers · 19 NLP & LLMs |
| IV · Expert & Production | Expert | 20 Interpretability & SHAP · 21 MLOps · 22 Recsys, anomalies, causality, RL · 23 End-to-end playbook · 24 Cheat sheets, glossary, interviews |

## Run the code

```bash
pip install numpy pandas scikit-learn matplotlib xgboost lightgbm shap statsmodels torch
python3 tools/check_code.py chapters/05-linear-regression.html   # executes every Python block in a chapter
```

## Layout

```
index.html            landing page + curriculum map + study plans
chapters/NN-*.html    one self-contained file per chapter
assets/style.css      design system (light + dark)
assets/app.js         sidebar, pager, progress, quizzes, copy buttons
assets/chapters.js    the chapter manifest (single source of truth)
tools/check_code.py   runs every code block in a chapter
tools/shot.js         Playwright screenshot + JS-error check
AUTHORING.md          the contract each chapter follows (add new chapters with it)
```
