# Ensemble Techniques Playground

An interactive **Streamlit** application that visually demonstrates the most popular ensemble learning techniques in machine learning — **Bagging, Random Forest, AdaBoost, Gradient Boosting, Voting, and Stacking** — on multiple datasets, with live decision-boundary plots, learning curves, feature importance, confusion matrices and side-by-side comparisons.

> Deploy-ready for **Render** via **GitHub** in under 5 minutes.

---

## Table of contents

- [Live demo](#live-demo)
- [Features](#features)
- [Ensemble techniques covered](#ensemble-techniques-covered)
- [Datasets](#datasets)
- [Project structure](#project-structure)
- [Quick start (local)](#quick-start-local)
- [Deploy to Render via GitHub](#deploy-to-render-via-github)
  - [Option A — render.yaml (recommended)](#option-a--renderyaml-recommended)
  - [Option B — Manual setup in the Render dashboard](#option-b--manual-setup-in-the-render-dashboard)
  - [Troubleshooting](#troubleshooting)
- [How the app works](#how-the-app-works)
- [Tech stack](#tech-stack)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Live demo

Once deployed, your app will be available at a URL like:

```
https://ensemble-techniques-playground.onrender.com
```

Replace this with your real Render URL after deployment.

---

## Features

- **6 datasets** — 3 synthetic 2D problems (Moons, Circles, Blobs) for visualisation, plus the classic Iris, Wine and Breast Cancer benchmarks.
- **8 techniques** — Single Decision Tree (baseline), Bagging, Random Forest, AdaBoost, Gradient Boosting, **XGBoost**, Voting (hard/soft), and Stacking.
- **Built-in "Techniques Overview" tab** — at-a-glance comparison table, **decision guide** ("which one should I pick?"), bias/variance map, and a card per technique with **✅ Use when**, **❌ Avoid when**, and a real-world example.
- **"Why pick this technique?" callout** on the header — every selected technique gets a tinted box explaining when to use it and a real-world use case, plus family / speed / reduces-bias-or-variance badges.
- **Live decision-boundary plots** that update as you tune hyper-parameters.
- **4 visualisation modes** in the boundary tab — toggle between **Hard boundary**, **Probability heatmap** (`predict_proba` surface), **Uncertainty (entropy)** and **Per-classifier votes** to see the same model from completely different angles.
- **Ensemble evolution scrubber** — drag a slider to watch the boundary form as the ensemble grows from 1 to *N* learners (uses `staged_predict` for AdaBoost / Gradient Boosting, `iteration_range` for XGBoost, and the first-*k* base estimators for Bagging / Random Forest).
- **Compare A vs B** — toggle a second technique in the sidebar to get side-by-side metrics, decision boundaries and confusion matrices, plus an *agreement* score between the two models.
- **"Weak learners → Ensemble"** view — see five individual bootstrap-sampled trees side-by-side with their aggregated ensemble to truly understand *why* ensembling works.
- **Learning-curve sweep** — accuracy vs `n_estimators` for bagging/boosting methods (including XGBoost).
- **Feature importance** charts (where supported).
- **Confusion matrix** + full classification report.
- **Full comparison mode** — train every technique on the chosen dataset and rank them by accuracy / F1 / training time in a heat-mapped table and bar chart.
- **Plot themes** — instantly restyle every chart between **Light**, **Dark** and **Vibrant**.
- **Mesh resolution** toggle — *Fast* (0.05 step) for snappy interaction, *Detailed* (0.02 step) for crisp boundaries.
- **Plain-English algorithm walkthroughs**, mathematical formulas, pros/cons.
- **Modern UI** with custom theme, gradient title and metric cards.

---

## Ensemble techniques covered

| Technique | Family | Key idea | Reduces |
|---|---|---|---|
| **Single Decision Tree** | Baseline | Recursive binary splits | — |
| **Bagging** | Parallel | Bootstrap + aggregate | Variance |
| **Random Forest** | Parallel | Bagging + random feature subsets | Variance |
| **AdaBoost** | Sequential | Re-weight misclassified samples | Bias |
| **Gradient Boosting** | Sequential | Fit residuals of a differentiable loss | Bias |
| **XGBoost** | Sequential | Newton-step boosting + L1/L2 reg + sub-sampling | Bias + Variance |
| **Voting** | Heterogeneous | Combine different model types | Both |
| **Stacking** | Heterogeneous | Meta-learner on base models' outputs | Both |

For each technique the app shows:

1. **Intuition** in plain English.
2. **Mathematical formulation** rendered with LaTeX.
3. **Pros & Cons**.
4. A **step-by-step algorithm walkthrough**.
5. Live decision boundary, metrics, and (where applicable) feature importance and learning curves.

---

## Datasets

| Dataset | Type | Features | Why it's included |
|---|---|---|---|
| Moons (2D) | Synthetic | 2 | Non-linear, perfect for boundary plots |
| Circles (2D) | Synthetic | 2 | Concentric — linear models fail |
| Classification (2D blobs) | Synthetic | 2 | Adjustable noise / overlap |
| Iris | Real | 4 | Classic 3-class problem |
| Wine | Real | 13 | Multi-class, high-dim |
| Breast Cancer | Real | 30 | Binary, real-world high-dim |

---

## Project structure

```
Ensemble_techniques/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Pinned Python dependencies
├── runtime.txt             # Python version for Render
├── render.yaml             # Render Blueprint (one-click deploy config)
├── Procfile                # Alternative start command for Render
├── .streamlit/
│   └── config.toml         # Streamlit theme & server config
├── .gitignore
└── README.md
```

---

## Quick start (local)

### Prerequisites

- Python **3.10+** (3.11 recommended)
- `pip` and (optionally) `venv`

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/ensemble-techniques-playground.git
cd ensemble-techniques-playground
```

### 2. Create a virtual environment

**macOS / Linux**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

---

## Deploy to Render via GitHub

[Render](https://render.com) offers a generous free tier that runs Streamlit apps perfectly. There are two ways to deploy — pick whichever you prefer.

### Push your code to GitHub first

```bash
git init
git add .
git commit -m "Initial commit: Ensemble Techniques Playground"
git branch -M main
git remote add origin https://github.com/<your-username>/ensemble-techniques-playground.git
git push -u origin main
```

### Option A — `render.yaml` (recommended)

This repo already contains a `render.yaml` Blueprint, so deployment is a single click.

1. Sign in to [Render](https://dashboard.render.com).
2. Click **New +** → **Blueprint**.
3. Connect your GitHub account and select the repository.
4. Render reads `render.yaml`, shows the planned service, and asks you to **Apply**.
5. Click **Apply** — Render builds the image and starts the service.
6. After ~3–5 minutes the URL becomes live (e.g. `https://ensemble-techniques-playground.onrender.com`).

That's it. Every `git push` to `main` will trigger an auto-deploy.

### Option B — Manual setup in the Render dashboard

If you prefer no Blueprint:

1. Sign in to [Render](https://dashboard.render.com).
2. Click **New +** → **Web Service** → connect GitHub → pick the repo.
3. Fill in the form:
   - **Name**: `ensemble-techniques-playground`
   - **Region**: closest to your users
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false
     ```
   - **Instance type**: `Free` (or higher if you want)
4. Add an environment variable:
   - `PYTHON_VERSION` = `3.11.9`
5. Click **Create Web Service**.

### Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails on `numpy` / `scikit-learn` | Make sure `PYTHON_VERSION=3.11.9` is set; older Pythons may lack wheels. |
| App boots but the URL shows "Bad Gateway" | Confirm the start command uses `--server.port=$PORT` and `--server.address=0.0.0.0`. |
| First load is slow / app sleeps | Render's free tier sleeps after ~15 min of inactivity. First request wakes it (~30 s). Upgrade to a paid plan to keep it warm. |
| `ModuleNotFoundError` | Re-check `requirements.txt` is at the repo root and matches `import` names in `app.py`. |
| White screen on Streamlit | Ensure `--server.headless=true` is set in the start command. |
| `_posixsubprocess` / multiprocessing error locally on Windows | Caused by Microsoft Store Python's sandbox. The app already defaults to `N_JOBS=1`. To go faster, install the regular [python.org](https://www.python.org/downloads/) build and set `N_JOBS=-1`. |

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | (set by Render) | HTTP port Streamlit binds to |
| `PYTHON_VERSION` | `3.11.9` | Python version on Render |
| `N_JOBS` | `1` locally / `-1` on Render | Parallel workers for Bagging / Random Forest / Voting / Stacking |

---

## How the app works

```
┌────────────────┐      ┌────────────────────┐      ┌────────────────┐
│   Sidebar      │ ───▶ │  load_dataset()     │ ───▶ │ train/test     │
│  (controls)    │      │  StandardScaler     │      │ split          │
└────────────────┘      └────────────────────┘      └───────┬────────┘
                                                            │
                                                            ▼
                                              ┌──────────────────────────┐
                                              │  build_model(technique)  │
                                              │  → fit() → predict()     │
                                              └─────────────┬────────────┘
                                                            │
        ┌──────────────┬──────────────┬──────────────┬──────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼      ▼              ▼              ▼
  Decision         Weak         Confusion      Feature   Learning     Compare       Full
  boundary tab     learners →   matrix &       importance curve over   A vs B        comparison
  (4 view-modes    Ensemble     classification bar chart  n_estimators (side-by-side  across all 8
   + scrubber)     visual                                              technique B)   techniques
```

Each tab in the **Decision boundary** tab can be toggled between four completely different visualisations:

| View mode | What you see | Powered by |
|---|---|---|
| Hard boundary | Coloured argmax regions + scrubber to see the boundary at step *k* | `predict` / `staged_predict` / `iteration_range` |
| Probability heatmap | Smooth `P(class=1)` surface (or max-prob for multi-class) | `predict_proba` |
| Uncertainty (entropy) | Bright = high predictive entropy ⇒ model is unsure | `predict_proba` + `−Σ p log p` |
| Per-classifier votes | Heat = fraction of base learners voting class 1 | iterating over `model.estimators_` / `staged_predict` |

Key implementation choices:

- **`@st.cache_data`** memoises dataset loading so changing only the technique is instant.
- **2D mesh prediction** for decision boundaries (`np.meshgrid` + `model.predict`).
- The **"Weak learners → Ensemble"** view trains 5 trees on independent bootstrap samples and compares their boundaries to the bagged aggregate — a visual proof of variance reduction.
- The **full comparison** trains every technique with sensible default hyper-parameters and ranks them by accuracy / F1 / training time.

---

## Tech stack

- [Streamlit](https://streamlit.io) — UI & server
- [scikit-learn](https://scikit-learn.org) — Bagging, Random Forest, AdaBoost, Gradient Boosting, Voting, Stacking
- [XGBoost](https://xgboost.readthedocs.io) — eXtreme Gradient Boosting
- [NumPy](https://numpy.org) / [Pandas](https://pandas.pydata.org) — data wrangling
- [Matplotlib](https://matplotlib.org) / [Seaborn](https://seaborn.pydata.org) — plotting
- [Render](https://render.com) — cloud hosting

---

## Roadmap

- [x] Add **XGBoost** as an optional technique (auto-detected, gracefully hidden if not installed).
- [ ] Add **LightGBM** and **CatBoost**.
- [ ] **Regression mode** with the same UX (use `make_regression`, California housing).
- [ ] **Custom CSV upload** so users can try their own dataset.
- [ ] Save / share configurations via URL query params.
- [ ] Dark theme toggle.
- [ ] Unit tests + GitHub Actions CI.

Pull requests are very welcome.

---

## Contributing

1. Fork the repo and create your branch: `git checkout -b feature/amazing-thing`.
2. Commit your changes: `git commit -m 'feat: add amazing thing'`.
3. Push the branch: `git push origin feature/amazing-thing`.
4. Open a Pull Request.

Please run `streamlit run app.py` locally before opening the PR to make sure nothing is broken.

---

## License

Released under the [MIT License](https://opensource.org/licenses/MIT). Use it, fork it, ship it.

---

**Built with curiosity and a lot of bootstrap samples.** If this project helped you understand ensembles better, consider giving the repo a ⭐ on GitHub.
