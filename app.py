"""
Ensemble Techniques Playground
A Streamlit app to interactively demonstrate Bagging, Boosting, Random Forest,
Gradient Boosting, Voting and Stacking on classification datasets.
"""

from __future__ import annotations

import io
import os
import sys
import time
import warnings
from dataclasses import dataclass

# Defensive: matplotlib's Path.__deepcopy__ recurses through nested Paths when
# copying tick properties. Default Python recursion (1000) is enough for
# normal plots but can blow up under some Python/matplotlib version pairs
# (e.g. Python 3.14 default Render runtime). Bumping it is cheap and safe.
sys.setrecursionlimit(10_000)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import ListedColormap
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    make_circles,
    make_classification,
    make_moons,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# Some Python distributions (e.g. Microsoft Store Python on Windows) fail to spawn
# the multiprocessing resource-tracker. Default to single-job; Render / multi-core
# servers can opt into parallelism with the N_JOBS env var.
N_JOBS = int(os.environ.get("N_JOBS", "1"))

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ensemble Techniques Playground",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f46e5, #06b6d4, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .technique-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #4338ca;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.4rem;
    }
    .badge-fam-baseline       { background:#f3f4f6; color:#374151; }
    .badge-fam-parallel       { background:#dcfce7; color:#166534; }
    .badge-fam-sequential     { background:#fee2e2; color:#991b1b; }
    .badge-fam-heterogeneous  { background:#e0e7ff; color:#3730a3; }
    .badge-speed-fast   { background:#dcfce7; color:#166534; }
    .badge-speed-medium { background:#fef9c3; color:#854d0e; }
    .badge-speed-slow   { background:#fee2e2; color:#991b1b; }
    .technique-card {
        background:#ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        height: 100%;
    }
    .technique-card h4 {
        margin: 0 0 0.4rem 0;
        font-size: 1.05rem;
        color: #111827;
    }
    .technique-card .one-liner {
        color: #4b5563;
        font-size: 0.92rem;
        margin: 0.6rem 0 0.7rem 0;
    }
    .technique-card .section-label {
        font-weight: 700;
        font-size: 0.82rem;
        color: #374151;
        margin-top: 0.6rem;
        margin-bottom: 0.2rem;
        display: block;
    }
    .technique-card ul {
        margin: 0;
        padding-left: 1.1rem;
        color: #1f2937;
        font-size: 0.88rem;
    }
    .technique-card li { margin: 0.1rem 0; }
    .why-callout {
        background: linear-gradient(135deg, #ecfeff 0%, #eef2ff 100%);
        border-left: 4px solid #4f46e5;
        padding: 0.85rem 1rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        color: #1e293b;
    }
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 2.5rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

sns.set_style("whitegrid")
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------
@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    name: str
    description: str


@st.cache_data(show_spinner=False)
def load_dataset(name: str, n_samples: int = 500, noise: float = 0.25, seed: int = 42) -> Dataset:
    """Build or fetch a dataset based on the user selection."""
    if name == "Moons (2D, non-linear)":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        return Dataset(
            X, y,
            ["feature_1", "feature_2"],
            ["class_0", "class_1"],
            name,
            "Two interleaving half-circles. Highlights how ensembles capture non-linear boundaries.",
        )
    if name == "Circles (2D, concentric)":
        X, y = make_circles(n_samples=n_samples, noise=noise * 0.4, factor=0.5, random_state=seed)
        return Dataset(
            X, y,
            ["feature_1", "feature_2"],
            ["outer", "inner"],
            name,
            "Concentric circles. Linear models fail; ensembles of trees succeed.",
        )
    if name == "Classification (2D blobs)":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=1.2,
            flip_y=noise * 0.2,
            random_state=seed,
        )
        return Dataset(
            X, y,
            ["feature_1", "feature_2"],
            ["class_0", "class_1"],
            name,
            "Synthetic 2D classification problem with adjustable label noise.",
        )
    if name == "Iris (classic)":
        ds = load_iris()
        return Dataset(
            ds.data, ds.target,
            list(ds.feature_names), list(ds.target_names),
            name,
            "Classic flower dataset, 4 features, 3 species.",
        )
    if name == "Wine":
        ds = load_wine()
        return Dataset(
            ds.data, ds.target,
            list(ds.feature_names), list(ds.target_names),
            name,
            "Chemical analysis of wines from three different cultivars.",
        )
    if name == "Breast Cancer":
        ds = load_breast_cancer()
        return Dataset(
            ds.data, ds.target,
            list(ds.feature_names), list(ds.target_names),
            name,
            "Diagnostic dataset for benign vs malignant tumours (30 features).",
        )
    raise ValueError(f"Unknown dataset: {name}")


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def build_model(technique: str, params: dict) -> object:
    """Instantiate the selected ensemble technique with supplied hyper-parameters."""
    seed = params.get("random_state", 42)

    if technique == "Single Decision Tree (baseline)":
        return DecisionTreeClassifier(max_depth=params["max_depth"], random_state=seed)

    if technique == "Bagging":
        base = DecisionTreeClassifier(max_depth=params["max_depth"], random_state=seed)
        return BaggingClassifier(
            estimator=base,
            n_estimators=params["n_estimators"],
            max_samples=params["max_samples"],
            bootstrap=True,
            random_state=seed,
            n_jobs=N_JOBS,
        )

    if technique == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            max_features="sqrt",
            random_state=seed,
            n_jobs=N_JOBS,
        )

    if technique == "AdaBoost":
        base = DecisionTreeClassifier(max_depth=1, random_state=seed)
        return AdaBoostClassifier(
            estimator=base,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=seed,
        )

    if technique == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=seed,
        )

    if technique == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError(
                "xgboost is not installed. Add `xgboost` to requirements.txt and reinstall."
            )
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=seed,
            n_jobs=N_JOBS,
            eval_metric="logloss",
            tree_method="hist",
            verbosity=0,
        )

    if technique == "Voting (hard / soft)":
        estimators = [
            ("logreg", LogisticRegression(max_iter=1000, random_state=seed)),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
            ("tree", DecisionTreeClassifier(max_depth=params["max_depth"], random_state=seed)),
            ("nb", GaussianNB()),
        ]
        return VotingClassifier(estimators=estimators, voting=params["voting"], n_jobs=N_JOBS)

    if technique == "Stacking":
        estimators = [
            ("tree", DecisionTreeClassifier(max_depth=params["max_depth"], random_state=seed)),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
            ("svc", SVC(probability=True, random_state=seed)),
        ]
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
            cv=5,
            n_jobs=N_JOBS,
        )

    raise ValueError(f"Unknown technique: {technique}")


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
CMAP_LIGHT = ListedColormap(["#FFD8E1", "#D8E5FF", "#D7F7DC"])
CMAP_BOLD = ListedColormap(["#E11D48", "#2563EB", "#16A34A"])


# ----------------------------------------------------------------------------
# Theme & visualisation helpers
# ----------------------------------------------------------------------------
def get_cmaps(theme: str):
    """Return (light_cmap, bold_cmap) for the chosen theme."""
    if theme == "Dark":
        return (
            ListedColormap(["#3a1f2c", "#1f2a4a", "#1f4030"]),
            ListedColormap(["#fb7185", "#60a5fa", "#34d399"]),
        )
    if theme == "Vibrant":
        return (
            ListedColormap(["#FFE5B4", "#B4E5FF", "#E5FFB4"]),
            ListedColormap(["#FF6B35", "#00B4D8", "#06FFA5"]),
        )
    return (
        ListedColormap(["#FFD8E1", "#D8E5FF", "#D7F7DC"]),
        ListedColormap(["#E11D48", "#2563EB", "#16A34A"]),
    )


def apply_theme(theme: str) -> None:
    """Apply a global matplotlib style based on the user's chosen theme."""
    if theme == "Dark":
        plt.style.use("dark_background")
    elif theme == "Vibrant":
        plt.style.use("seaborn-v0_8-bright")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def majority_vote(preds_2d: np.ndarray) -> np.ndarray:
    """Vectorised mode along axis=0 for integer label arrays of shape (k, m)."""
    n_classes = int(preds_2d.max()) + 1
    counts = np.zeros((n_classes, preds_2d.shape[1]), dtype=int)
    for c in range(n_classes):
        counts[c] = (preds_2d == c).sum(axis=0)
    return counts.argmax(axis=0)


def supports_evolution(technique: str) -> bool:
    return technique in {
        "Bagging", "Random Forest", "AdaBoost", "Gradient Boosting", "XGBoost",
    }


def get_max_step(model, technique: str) -> int:
    if technique in ("Bagging", "Random Forest"):
        return len(getattr(model, "estimators_", []))
    if technique == "AdaBoost":
        return len(getattr(model, "estimators_", []))
    if technique == "Gradient Boosting":
        return getattr(model, "n_estimators_", model.n_estimators)
    if technique == "XGBoost":
        return model.get_booster().num_boosted_rounds()
    return 1


def predict_at_step(model, technique: str, points: np.ndarray, k: int) -> np.ndarray:
    """Predict using only the first k learners of the ensemble."""
    if technique in ("Bagging", "Random Forest"):
        preds = np.array([est.predict(points) for est in model.estimators_[:k]], dtype=int)
        return majority_vote(preds)
    if technique in ("AdaBoost", "Gradient Boosting"):
        staged = list(model.staged_predict(points))
        idx = max(0, min(k - 1, len(staged) - 1))
        return staged[idx]
    if technique == "XGBoost":
        return model.predict(points, iteration_range=(0, max(1, k))).astype(int)
    return model.predict(points).astype(int)


def _make_mesh(X: np.ndarray, resolution: str):
    """Return (xx, yy, mesh_pts) for a 2D feature space."""
    h = 0.05 if resolution == "Fast" else 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def plot_decision_boundary(model, X, y, title: str, ax=None,
                           cmap_light=None, cmap_bold=None, resolution: str = "Detailed"):
    """Plot a 2-D decision boundary using a fine mesh."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    cmap_light = cmap_light or CMAP_LIGHT
    cmap_bold = cmap_bold or CMAP_BOLD

    xx, yy, pts = _make_mesh(X, resolution)
    Z = model.predict(pts).reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.85)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="white", s=45, linewidth=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    return ax


def plot_evolution_boundary(model, technique: str, k: int, X, y, title: str, ax,
                            cmap_light, cmap_bold, resolution: str = "Detailed"):
    """Boundary using only the first k learners — the evolution scrubber view."""
    xx, yy, pts = _make_mesh(X, resolution)
    Z = predict_at_step(model, technique, pts, k).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.85)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="white", s=45, linewidth=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    return ax


def plot_proba_heatmap(model, X, y, title: str, ax, cmap_bold, resolution: str = "Detailed"):
    """Probability heatmap. Binary → P(class=1) with diverging colormap;
    multi-class → max-prob (confidence) with viridis."""
    if not hasattr(model, "predict_proba"):
        return None
    xx, yy, pts = _make_mesh(X, resolution)
    proba = model.predict_proba(pts)
    if proba.shape[1] == 2:
        Z = proba[:, 1].reshape(xx.shape)
        cs = ax.contourf(xx, yy, Z, levels=20, cmap="RdBu_r", alpha=0.9, vmin=0, vmax=1)
        plt.colorbar(cs, ax=ax, label="P(class = 1)")
    else:
        Z = proba.max(axis=1).reshape(xx.shape)
        cs = ax.contourf(xx, yy, Z, levels=20, cmap="viridis", alpha=0.9)
        plt.colorbar(cs, ax=ax, label="max P(class)")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
               edgecolor="white", s=45, linewidth=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    return ax


def plot_entropy_heatmap(model, X, y, title: str, ax, cmap_bold, resolution: str = "Detailed"):
    """Predictive entropy: bright regions = high uncertainty."""
    if not hasattr(model, "predict_proba"):
        return None
    xx, yy, pts = _make_mesh(X, resolution)
    proba = model.predict_proba(pts)
    eps = 1e-12
    entropy = -np.sum(proba * np.log(proba + eps), axis=1)
    Z = entropy.reshape(xx.shape)
    cs = ax.contourf(xx, yy, Z, levels=20, cmap="magma", alpha=0.95)
    plt.colorbar(cs, ax=ax, label="entropy (nats)")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
               edgecolor="white", s=45, linewidth=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    return ax


def plot_classifier_votes(model, technique: str, X, y, title: str, ax,
                          cmap_bold, resolution: str = "Detailed"):
    """For ensembles of trees: heatmap of how many learners voted for class 1."""
    if not supports_evolution(technique) or technique == "XGBoost":
        return None
    xx, yy, pts = _make_mesh(X, resolution)
    if technique in ("Bagging", "Random Forest"):
        preds = np.array([est.predict(pts) for est in model.estimators_], dtype=int)
        Z = preds.mean(axis=0).reshape(xx.shape)
    else:
        staged = list(model.staged_predict(pts))
        preds = np.array(staged, dtype=int)
        Z = preds.mean(axis=0).reshape(xx.shape)
    cs = ax.contourf(xx, yy, Z, levels=20, cmap="PiYG", alpha=0.9, vmin=0, vmax=1)
    plt.colorbar(cs, ax=ax, label="fraction voting class 1")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
               edgecolor="white", s=45, linewidth=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    return ax


def plot_confusion(cm: np.ndarray, classes: list[str], ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=classes, yticklabels=classes, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    return ax


def plot_feature_importance(model, feature_names: list[str], ax=None):
    if not hasattr(model, "feature_importances_"):
        return None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:15]
    names = np.array(feature_names)[order]
    sns.barplot(
        x=importances[order],
        y=names,
        hue=names,
        ax=ax,
        palette="viridis",
        legend=False,
    )
    ax.set_title("Feature Importance", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    return ax


def plot_learning_curve(model_factory, X_train, y_train, X_test, y_test, n_max: int):
    """Plot accuracy as the number of estimators grows."""
    steps = sorted(set([1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, n_max]))
    steps = [s for s in steps if s <= n_max]

    train_scores, test_scores = [], []
    for n in steps:
        model = model_factory(n)
        model.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, model.predict(X_train)))
        test_scores.append(accuracy_score(y_test, model.predict(X_test)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, train_scores, marker="o", label="Train accuracy", color="#4f46e5")
    ax.plot(steps, test_scores, marker="s", label="Test accuracy", color="#10b981")
    ax.fill_between(steps, train_scores, test_scores, alpha=0.08, color="#6366f1")
    ax.set_xlabel("Number of estimators")
    ax.set_ylabel("Accuracy")
    ax.set_title("Effect of ensemble size on accuracy", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_individual_vs_ensemble(X_train, y_train, X_test, y_test, n_learners: int = 5, max_depth: int = 4):
    """Compare individual weak learners to their bagged ensemble."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    rng = np.random.default_rng(42)
    individual_preds = np.zeros((n_learners, len(X_test)))

    for i in range(n_learners):
        idx = rng.integers(0, len(X_train), len(X_train))
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=i)
        tree.fit(X_train[idx], y_train[idx])
        individual_preds[i] = tree.predict(X_test)

        if X_train.shape[1] == 2 and i < 5:
            plot_decision_boundary(
                tree, X_train, y_train,
                f"Tree #{i+1} (acc={accuracy_score(y_test, individual_preds[i]):.2f})",
                ax=axes[i],
            )

    ensemble_preds = (individual_preds.mean(axis=0) > 0.5).astype(int)
    ensemble_acc = accuracy_score(y_test, ensemble_preds)

    if X_train.shape[1] == 2:
        bagger = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=max_depth),
            n_estimators=n_learners, random_state=42,
        )
        bagger.fit(X_train, y_train)
        plot_decision_boundary(
            bagger, X_train, y_train,
            f"Ensemble (acc={ensemble_acc:.2f})",
            ax=axes[5],
        )
        for ax in axes[5:]:
            for spine in ax.spines.values():
                spine.set_edgecolor("#10b981")
                spine.set_linewidth(2)

    fig.suptitle("Individual weak learners vs. their ensemble", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, ensemble_acc


# ----------------------------------------------------------------------------
# Technique explanations
# ----------------------------------------------------------------------------
TECHNIQUE_INFO = {
    "Single Decision Tree (baseline)": {
        "family": "Baseline",
        "one_liner": "A single recursive splitter — the building block that ensembles improve on.",
        "reduces": "—",
        "speed": "Fast",
        "intuition": "A single tree splits the feature space recursively. It is the **base learner** that all our ensembles will improve upon. High variance, low bias.",
        "math": r"\hat{y} = \text{tree}(x)",
        "pros": ["Easy to interpret", "Fast to train", "No scaling required"],
        "cons": ["Overfits easily", "High variance", "Unstable to small data changes"],
        "when_to_use": [
            "You need a fully **explainable** model (an auditor must read it)",
            "Quick sanity check / baseline before trying anything fancier",
            "Very small datasets where ensembles would just overfit",
            "You want a rule list to ship as business logic",
        ],
        "when_to_avoid": [
            "Production systems where every accuracy point matters",
            "Noisy data — single trees latch on to noise",
            "Datasets with strong feature interactions you can't capture in one tree",
        ],
        "use_case": "Rule-of-thumb triage in healthcare, eligibility decisions, customer-segmentation cheat-sheets.",
    },
    "Bagging": {
        "family": "Parallel ensemble",
        "one_liner": "Train many independent learners on bootstrapped data and average their votes.",
        "reduces": "Variance",
        "speed": "Medium",
        "intuition": "**Bagging = Bootstrap AGGregatING.** Train many trees on random *bootstrap samples* of the data and average their votes. Reduces **variance** without increasing bias.",
        "math": r"\hat{y} = \text{mode}\Big(h_1(x), h_2(x), \dots, h_B(x)\Big), \quad h_b \sim \text{tree on bootstrap}_b",
        "pros": ["Reduces overfitting", "Easily parallelisable", "Works with any base learner"],
        "cons": ["Less interpretable", "More memory & compute", "Doesn't reduce bias"],
        "when_to_use": [
            "Your base model is **high-variance** (deep tree, KNN with k=1)",
            "You have multiple CPU cores and want to train in parallel",
            "Dataset is medium-to-large and you can afford B copies of it",
        ],
        "when_to_avoid": [
            "Base learner is already **biased / stable** (e.g. LogReg) — bagging won't help",
            "Memory or latency is tight (you'll keep B models in RAM)",
        ],
        "use_case": "Robustifying any noisy base model — e.g. bagged KNN for sensor data, bagged neural nets for images.",
    },
    "Random Forest": {
        "family": "Parallel ensemble",
        "one_liner": "Bagging + random feature subsets at each split → decorrelated trees.",
        "reduces": "Variance",
        "speed": "Medium",
        "intuition": "Bagging + a twist: at every split each tree only considers a **random subset of features**. This decorrelates the trees and gives an even bigger variance reduction.",
        "math": r"\text{Forest} = \frac{1}{B}\sum_{b=1}^{B}\text{Tree}_b(x;\;\text{features}\subset\mathcal{F})",
        "pros": ["Strong out-of-the-box performance", "Built-in feature importance", "Handles many feature types"],
        "cons": ["Slower to predict", "Harder to interpret than one tree", "Can overfit on noisy data"],
        "when_to_use": [
            "You want a **strong baseline** with almost zero tuning",
            "You need **feature importance** for free",
            "Tabular data with mixed numeric / categorical features",
            "You have many irrelevant features — RF naturally ignores them",
        ],
        "when_to_avoid": [
            "Online / very-low-latency prediction (hundreds of trees to walk)",
            "Tiny datasets (≤200 rows) — gradient boosting often wins",
            "You need extrapolation (RF can't predict outside training range)",
        ],
        "use_case": "Credit scoring, churn prediction, fraud detection, biology / ecology classifications.",
    },
    "AdaBoost": {
        "family": "Sequential boosting",
        "one_liner": "Re-weight misclassified samples and add a new weak learner that focuses on them.",
        "reduces": "Bias",
        "speed": "Medium",
        "intuition": "**Sequential** ensemble. Train a weak learner, then **up-weight the misclassified examples**, train another, and so on. Each new learner focuses on what the previous ones got wrong.",
        "math": r"H(x) = \mathrm{sign}\Big(\sum_{t=1}^{T} \alpha_t h_t(x)\Big), \quad \alpha_t = \tfrac{1}{2}\ln\tfrac{1-\varepsilon_t}{\varepsilon_t}",
        "pros": ["Reduces bias", "Often very accurate", "Few hyperparameters"],
        "cons": ["Sensitive to outliers", "Cannot be easily parallelised", "Can overfit if T is too large"],
        "when_to_use": [
            "Base learner is a **high-bias weak learner** (decision stump)",
            "Clean labels — no obvious outliers",
            "Binary classification with moderate dataset size",
        ],
        "when_to_avoid": [
            "Noisy labels or many outliers — they get over-weighted",
            "You can't tune `n_estimators` — AdaBoost can overfit if T is too large",
            "Multi-class problems where Gradient Boosting handles loss more cleanly",
        ],
        "use_case": "Classic face detection (Viola-Jones), text classification with weak rules, fraud signals.",
    },
    "Gradient Boosting": {
        "family": "Sequential boosting",
        "one_liner": "Each new tree fits the gradient of the loss — boosting generalised to any loss.",
        "reduces": "Bias",
        "speed": "Slow",
        "intuition": "Each new tree fits the **gradient (residual error) of the loss** of the previous ensemble. Like AdaBoost but generalised to any differentiable loss.",
        "math": r"F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x), \quad h_m \approx -\frac{\partial L}{\partial F_{m-1}}",
        "pros": ["State-of-the-art on tabular data", "Flexible loss functions", "Handles mixed feature types"],
        "cons": ["Slower to train", "Sensitive to hyperparameters", "Risk of overfitting"],
        "when_to_use": [
            "You want top accuracy on **tabular data** and have time to tune",
            "You need a **custom loss** (Huber, quantile, ranking, custom)",
            "Datasets up to a few million rows",
        ],
        "when_to_avoid": [
            "You need millisecond inference — many sequential trees to traverse",
            "You can't iterate on hyperparameters (η, depth, n_estimators all matter)",
            "Streaming / online learning — boosting needs the full dataset",
        ],
        "use_case": "Insurance pricing, ranking on search engines, demand forecasting, click-through-rate prediction.",
    },
    "XGBoost": {
        "family": "Sequential boosting",
        "one_liner": "Hyper-optimised gradient boosting with Newton steps + L1/L2 reg + sub-sampling.",
        "reduces": "Bias + Variance",
        "speed": "Medium",
        "intuition": "**eXtreme Gradient Boosting.** A highly optimised gradient-boosting library with a **second-order (Newton) approximation**, built-in **L1/L2 regularisation**, smart **column/row sub-sampling**, parallel histogram building and clever handling of missing values. Often the top performer on tabular Kaggle competitions.",
        "math": r"\mathcal{L}^{(t)} = \sum_i \big[g_i f_t(x_i) + \tfrac12 h_i f_t(x_i)^2\big] + \Omega(f_t),\quad \Omega(f)=\gamma T + \tfrac{1}{2}\lambda\|w\|^2",
        "pros": [
            "Usually the strongest tabular baseline",
            "Built-in regularisation (γ, λ, α)",
            "Row + column sub-sampling reduces overfitting",
            "Native handling of missing values",
            "Fast histogram-based training",
        ],
        "cons": [
            "Many hyperparameters to tune",
            "External dependency (not in scikit-learn)",
            "Slower than simpler methods on tiny datasets",
            "Less interpretable than a single tree",
        ],
        "when_to_use": [
            "**Tabular Kaggle-style** problems where every fraction of accuracy matters",
            "Datasets with **missing values** you'd rather not impute",
            "You can spend time tuning η, depth, λ, subsample, colsample_bytree",
            "Production where you need predict-time speed (XGBoost is fast at inference)",
        ],
        "when_to_avoid": [
            "Tiny datasets — simpler models are less likely to overfit",
            "Pure image / text / speech — use deep learning instead",
            "You need a model with closed-form interpretability (use a single tree or LR)",
        ],
        "use_case": "Banking risk models, ad CTR / CVR prediction, recommendation re-ranking, structured ML competitions.",
    },
    "Voting (hard / soft)": {
        "family": "Heterogeneous ensemble",
        "one_liner": "Combine different kinds of models (LR, KNN, Tree, NB) with a fixed voting rule.",
        "reduces": "Bias + Variance",
        "speed": "Fast → Medium",
        "intuition": "Combine **different kinds of models** (LogReg, KNN, Tree, NB...) and vote. *Hard* = majority class, *Soft* = average probabilities.",
        "math": r"\hat{y}_{\text{soft}} = \arg\max_c \frac{1}{M}\sum_{m=1}^{M} P_m(y=c\mid x)",
        "pros": ["Simple and effective", "Diverse models reduce errors", "Easy to extend"],
        "cons": ["No learned weighting", "Worst learner can drag others", "Soft voting needs probabilities"],
        "when_to_use": [
            "You already have **several decent models** of different families",
            "You want a quick win without training a meta-model",
            "All base learners output **calibrated probabilities** (for soft voting)",
        ],
        "when_to_avoid": [
            "One of your models is much worse than the others — it'll just add noise",
            "You'd benefit from **learned** weighting → use Stacking instead",
        ],
        "use_case": "Combining a logistic regression, a tree, and a KNN for a quick robust baseline ensemble.",
    },
    "Stacking": {
        "family": "Heterogeneous ensemble",
        "one_liner": "Train base models, then a meta-learner on their predictions to learn the best mix.",
        "reduces": "Bias + Variance",
        "speed": "Slow",
        "intuition": "Train several base learners, then train a **meta-learner** on their predictions. The meta-model *learns how to combine* the base models, unlike voting which uses a fixed rule.",
        "math": r"\hat{y} = g\Big(h_1(x), h_2(x), \dots, h_M(x)\Big), \quad g \text{ is the meta-learner}",
        "pros": ["Often beats voting", "Leverages model diversity", "Learns optimal combination"],
        "cons": ["Complex pipeline", "Higher risk of leakage", "Slow to train (CV needed)"],
        "when_to_use": [
            "You're squeezing the **last bit of accuracy** from a competition",
            "Your base models make **different mistakes** (low correlation)",
            "You can afford the cross-validated training cost",
        ],
        "when_to_avoid": [
            "Time-constrained pipelines or production with frequent retraining",
            "Few diverse base models — meta-learner has nothing to learn from",
            "You haven't carefully set up the CV → leakage will inflate validation scores",
        ],
        "use_case": "Top-of-leaderboard Kaggle solutions, AutoML systems blending dozens of base models.",
    },
}


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
def sidebar_controls():
    st.sidebar.markdown("## Configuration")

    dataset_name = st.sidebar.selectbox(
        "Dataset",
        [
            "Moons (2D, non-linear)",
            "Circles (2D, concentric)",
            "Classification (2D blobs)",
            "Iris (classic)",
            "Wine",
            "Breast Cancer",
        ],
        help="2D datasets allow decision-boundary visualisation.",
    )

    is_synth = "2D" in dataset_name
    n_samples = st.sidebar.slider("Sample size", 100, 2000, 500, 50, disabled=not is_synth)
    noise = st.sidebar.slider("Noise / class overlap", 0.0, 0.6, 0.25, 0.05, disabled=not is_synth)

    test_size = st.sidebar.slider("Test set size (%)", 10, 50, 25, 5) / 100.0
    seed = st.sidebar.number_input("Random seed", 0, 10_000, 42, 1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Ensemble technique")

    available_techniques = [
        t for t in TECHNIQUE_INFO.keys() if t != "XGBoost" or XGBOOST_AVAILABLE
    ]
    technique = st.sidebar.selectbox(
        "Choose a technique",
        available_techniques,
    )
    if not XGBOOST_AVAILABLE:
        st.sidebar.caption(
            "ℹ️ Install `xgboost` (`pip install xgboost`) to enable the XGBoost demo."
        )

    st.sidebar.markdown("### Hyper-parameters")
    params: dict = {"random_state": seed}

    if technique == "Single Decision Tree (baseline)":
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 20, 5)

    elif technique == "Bagging":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 1, 300, 50, 1)
        params["max_depth"] = st.sidebar.slider("base tree max_depth", 1, 20, 5)
        params["max_samples"] = st.sidebar.slider("max_samples (bootstrap %)", 0.1, 1.0, 1.0, 0.05)

    elif technique == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 1, 500, 100, 1)
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 30, 8)

    elif technique == "AdaBoost":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 1, 300, 50, 1)
        params["learning_rate"] = st.sidebar.slider("learning_rate", 0.01, 2.0, 1.0, 0.05)

    elif technique == "Gradient Boosting":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 1, 500, 100, 1)
        params["learning_rate"] = st.sidebar.slider("learning_rate", 0.01, 1.0, 0.1, 0.01)
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 10, 3)

    elif technique == "XGBoost":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 1, 500, 200, 1)
        params["learning_rate"] = st.sidebar.slider("learning_rate (η)", 0.01, 1.0, 0.1, 0.01)
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 12, 4)
        params["subsample"] = st.sidebar.slider("subsample (rows %)", 0.3, 1.0, 1.0, 0.05)
        params["colsample_bytree"] = st.sidebar.slider("colsample_bytree", 0.3, 1.0, 1.0, 0.05)
        params["reg_lambda"] = st.sidebar.slider("L2 reg (λ)", 0.0, 5.0, 1.0, 0.1)

    elif technique == "Voting (hard / soft)":
        params["voting"] = st.sidebar.radio("Voting type", ["hard", "soft"], horizontal=True)
        params["max_depth"] = st.sidebar.slider("tree max_depth", 1, 15, 5)

    elif technique == "Stacking":
        params["max_depth"] = st.sidebar.slider("base tree max_depth", 1, 15, 5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("## View options")

    theme = st.sidebar.radio(
        "Plot theme",
        ["Light", "Dark", "Vibrant"],
        horizontal=True,
        help="Switches matplotlib style and colour palette across every chart.",
    )
    resolution = st.sidebar.radio(
        "Mesh resolution",
        ["Fast", "Detailed"],
        index=1,
        horizontal=True,
        help="Detailed = 0.02 step (smoother but slower). Fast = 0.05 step.",
    )

    compare_with = st.sidebar.checkbox(
        "Compare with another technique", value=False,
        help="Side-by-side decision-boundary, metrics & confusion matrix in a new tab.",
    )
    technique_b = None
    if compare_with:
        options_b = [t for t in TECHNIQUE_INFO.keys()
                     if t != technique and (t != "XGBoost" or XGBOOST_AVAILABLE)]
        technique_b = st.sidebar.selectbox("Second technique (B)", options_b)

    show_compare = st.sidebar.checkbox("Show full comparison of ALL techniques", value=False)
    show_individual = st.sidebar.checkbox("Show 'individual learners vs ensemble'", value=True)

    return {
        "dataset_name": dataset_name,
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "seed": seed,
        "technique": technique,
        "params": params,
        "theme": theme,
        "resolution": resolution,
        "compare_with": compare_with,
        "technique_b": technique_b,
        "show_compare": show_compare,
        "show_individual": show_individual,
    }


def default_params(seed: int = 42) -> dict:
    """Sensible default hyper-parameters used by the 'Compare' and 'Full ranking' modes."""
    return {
        "random_state": seed,
        "max_depth": 5,
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_samples": 1.0,
        "voting": "soft",
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
    }


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def run_full_comparison(X_train, X_test, y_train, y_test, max_depth: int = 5):
    """Train every technique and return a results dataframe."""
    techniques = [t for t in TECHNIQUE_INFO.keys() if t != "XGBoost" or XGBOOST_AVAILABLE]
    rows = []
    progress = st.progress(0.0, text="Training all techniques...")
    for i, t in enumerate(techniques, 1):
        params = {
            "random_state": 42,
            "max_depth": max_depth,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_samples": 1.0,
            "voting": "soft",
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
        }
        model = build_model(t, params)
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        pred = model.predict(X_test)
        rows.append({
            "Technique": t,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1": f1_score(y_test, pred, average="weighted", zero_division=0),
            "Train time (s)": train_time,
        })
        progress.progress(i / len(techniques), text=f"Trained: {t}")
    progress.empty()
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Techniques overview renderer  (NEW: descriptive "which to use & why" tab)
# ----------------------------------------------------------------------------
FAMILY_BADGE_CLASS = {
    "Baseline": "badge-fam-baseline",
    "Parallel ensemble": "badge-fam-parallel",
    "Sequential boosting": "badge-fam-sequential",
    "Heterogeneous ensemble": "badge-fam-heterogeneous",
}

SPEED_BADGE_CLASS = {
    "Fast": "badge-speed-fast",
    "Medium": "badge-speed-medium",
    "Slow": "badge-speed-slow",
    "Fast → Medium": "badge-speed-medium",
}


def _technique_card_html(name: str, info: dict) -> str:
    fam_cls = FAMILY_BADGE_CLASS.get(info["family"], "badge-fam-baseline")
    spd_cls = SPEED_BADGE_CLASS.get(info["speed"], "badge-speed-medium")
    use_items = "".join(f"<li>{x}</li>" for x in info["when_to_use"])
    avoid_items = "".join(f"<li>{x}</li>" for x in info["when_to_avoid"])
    return f"""
    <div class="technique-card">
        <h4>{name}</h4>
        <span class="technique-badge {fam_cls}">{info["family"]}</span>
        <span class="technique-badge {spd_cls}">Speed: {info["speed"]}</span>
        <span class="technique-badge">Reduces: {info["reduces"]}</span>
        <p class="one-liner">{info["one_liner"]}</p>
        <span class="section-label">✅ Use it when</span>
        <ul>{use_items}</ul>
        <span class="section-label">❌ Avoid it when</span>
        <ul>{avoid_items}</ul>
        <span class="section-label">💡 Real-world example</span>
        <p style="font-size:0.88rem; color:#1f2937; margin:0.2rem 0 0 0;">{info["use_case"]}</p>
    </div>
    """


def render_techniques_overview(current_technique: str | None = None) -> None:
    """Always-on tab that explains every ensemble technique side-by-side
    and tells the user *which one to pick and why*."""

    st.markdown("## What are ensemble techniques?")
    st.markdown(
        "An **ensemble** combines multiple models so that their collective prediction "
        "is more accurate than any single model. Different ensembles attack different "
        "problems — some reduce **variance** (bagging), some reduce **bias** (boosting), "
        "and some combine **diverse model families** (voting / stacking)."
    )

    # ---------------- Quick comparison table ----------------
    st.markdown("### At-a-glance comparison")
    table = pd.DataFrame([
        {
            "Technique": name,
            "Family": info["family"],
            "Reduces": info["reduces"],
            "Speed": info["speed"],
            "One-liner": info["one_liner"],
        }
        for name, info in TECHNIQUE_INFO.items()
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)

    # ---------------- Decision guide ----------------
    st.markdown("### Decision guide — *which technique should I pick?*")
    st.markdown(
        """
| If your priority is… | Pick |
|---|---|
| **Maximum interpretability** (audit-friendly rules) | Single Decision Tree |
| **Strong out-of-the-box accuracy with no tuning** | Random Forest |
| **Reducing the variance** of a high-variance base model | Bagging |
| **Reducing the bias** of weak learners (focus on hard cases) | AdaBoost |
| **State-of-the-art on tabular data, custom losses** | Gradient Boosting |
| **Top-of-leaderboard tabular performance + missing-value handling** | XGBoost |
| **Quick win combining models you already trained** | Voting (soft) |
| **Squeezing out the last drop of accuracy with diverse models** | Stacking |
"""
    )

    st.markdown("### Bias vs variance map")
    st.markdown(
        """
```
              ┌────────────────────────────────────────────┐
   high       │                       AdaBoost · GBM       │
   bias       │    Logistic regression  XGBoost            │
              │                                            │
              │      Voting (depends on members)           │
              │                                            │
   low        │   Random Forest        Stacking            │
   bias       │   Bagging              (best blend)        │
              └────────────────────────────────────────────┘
                low variance   ─────────────►  high variance
```
- **Bagging family** (Bagging, Random Forest) ⇒ lowers **variance**.
- **Boosting family** (AdaBoost, GBM, XGBoost) ⇒ lowers **bias**.
- **Heterogeneous family** (Voting, Stacking) ⇒ lowers **both** by combining different mistakes.
"""
    )

    # ---------------- Cards (2 per row) ----------------
    st.markdown("### Technique cards")
    techniques = list(TECHNIQUE_INFO.keys())
    for i in range(0, len(techniques), 2):
        c1, c2 = st.columns(2)
        for col, name in zip(
            [c1, c2], techniques[i : i + 2]
        ):
            with col:
                if name == current_technique:
                    st.markdown(
                        f"<div style='font-weight:700;color:#4f46e5;margin-bottom:0.2rem;'>"
                        f"⬇ Currently selected"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown(_technique_card_html(name, TECHNIQUE_INFO[name]),
                            unsafe_allow_html=True)

    # ---------------- Bottom-line cheat sheet ----------------
    st.markdown("### One-sentence cheat sheet")
    st.markdown(
        """
- **Bagging** = many models, *averaged*, to **stop overfitting**.
- **Random Forest** = bagging trees with random features so the trees disagree more.
- **AdaBoost** = each new learner focuses on what the previous ones got wrong.
- **Gradient Boosting** = each new tree fits the *residual error* of the loss.
- **XGBoost** = gradient boosting + Newton step + L1/L2 reg + clever engineering.
- **Voting** = several different models, fixed combination rule.
- **Stacking** = several different models, **learned** combination rule.
"""
    )


def main():
    st.markdown('<div class="main-title">Ensemble Techniques Playground</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">An interactive tour of <b>Bagging</b>, '
        '<b>Random Forest</b>, <b>AdaBoost</b>, <b>Gradient Boosting</b>, '
        '<b>Voting</b> and <b>Stacking</b> — see how each one works and compare them side-by-side.</div>',
        unsafe_allow_html=True,
    )

    cfg = sidebar_controls()

    # Apply user-chosen theme to all matplotlib figures
    apply_theme(cfg["theme"])
    global CMAP_LIGHT, CMAP_BOLD
    CMAP_LIGHT, CMAP_BOLD = get_cmaps(cfg["theme"])

    # --- Load data
    ds = load_dataset(cfg["dataset_name"], cfg["n_samples"], cfg["noise"], cfg["seed"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(ds.X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, ds.y, test_size=cfg["test_size"], random_state=cfg["seed"], stratify=ds.y
    )

    # --- Header info
    info = TECHNIQUE_INFO[cfg["technique"]]
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fam_cls = FAMILY_BADGE_CLASS.get(info["family"], "badge-fam-baseline")
        spd_cls = SPEED_BADGE_CLASS.get(info["speed"], "badge-speed-medium")
        st.markdown(f"### Selected technique: `{cfg['technique']}`")
        st.markdown(
            f"<span class='technique-badge {fam_cls}'>{info['family']}</span>"
            f"<span class='technique-badge {spd_cls}'>Speed: {info['speed']}</span>"
            f"<span class='technique-badge'>Reduces: {info['reduces']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(info["intuition"])
        st.latex(info["math"])

        st.markdown(
            f"<div class='why-callout'>"
            f"<b>💡 Pick this when:</b> {info['when_to_use'][0].lower().rstrip('.') }. "
            f"<br><b>🎯 Real-world example:</b> {info['use_case']}"
            f"</div>",
            unsafe_allow_html=True,
        )

        with st.expander("✅ When to use it / ❌ When to avoid it"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**✅ Use when**")
                for p in info["when_to_use"]:
                    st.markdown(f"- {p}")
            with c2:
                st.markdown("**❌ Avoid when**")
                for p in info["when_to_avoid"]:
                    st.markdown(f"- {p}")
        with st.expander("Pros & Cons"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Pros**")
                for p in info["pros"]:
                    st.markdown(f"- {p}")
            with c2:
                st.markdown("**Cons**")
                for c in info["cons"]:
                    st.markdown(f"- {c}")
        st.caption("📖 Open the **Techniques overview** tab to compare all 8 techniques side-by-side.")
    with col_b:
        st.markdown(f"### Dataset: `{ds.name}`")
        st.caption(ds.description)
        st.markdown(
            f"<div class='metric-card'>"
            f"<b>Shape:</b> {ds.X.shape[0]} samples × {ds.X.shape[1]} features<br>"
            f"<b>Classes:</b> {len(ds.target_names)} → {', '.join(ds.target_names[:4])}"
            f"{'...' if len(ds.target_names) > 4 else ''}<br>"
            f"<b>Train / Test:</b> {len(X_train)} / {len(X_test)}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- Train selected model
    with st.spinner(f"Training {cfg['technique']}..."):
        model = build_model(cfg["technique"], cfg["params"])
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        y_pred = model.predict(X_test)

    # --- Metrics row
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1 score", f"{f1:.3f}")
    m5.metric("Train time", f"{train_time*1000:.0f} ms")

    st.markdown("---")

    # --- Tabs
    tabs = ["Decision boundary", "Confusion matrix", "Feature importance",
            "Learning curve", "How it works", "📚 Techniques overview"]
    if cfg["show_individual"]:
        tabs.insert(1, "Weak learners → Ensemble")
    if cfg["compare_with"] and cfg["technique_b"]:
        tabs.append("Compare A vs B")
    if cfg["show_compare"]:
        tabs.append("Full comparison")

    tab_objs = st.tabs(tabs)
    tab_map = dict(zip(tabs, tab_objs))

    # ---- Decision boundary (with view-mode toggle + evolution scrubber)
    with tab_map["Decision boundary"]:
        if X_train.shape[1] == 2:
            view_options = ["Hard boundary", "Probability heatmap", "Uncertainty (entropy)"]
            if supports_evolution(cfg["technique"]) and cfg["technique"] != "XGBoost":
                view_options.append("Per-classifier votes")

            view_mode = st.radio(
                "Visualisation mode",
                view_options,
                horizontal=True,
                help=(
                    "**Hard boundary** = argmax class. "
                    "**Probability heatmap** = `predict_proba` over the plane. "
                    "**Uncertainty** = entropy of the predicted distribution. "
                    "**Per-classifier votes** = average vote of every base learner."
                ),
                key="view_mode_main",
            )

            scrub_step = None
            if supports_evolution(cfg["technique"]):
                max_step = max(get_max_step(model, cfg["technique"]), 1)
                if max_step > 1 and view_mode == "Hard boundary":
                    scrub_step = st.slider(
                        f"Show ensemble at step k (1 → {max_step})",
                        1, max_step, max_step, 1,
                        help="Drag left to see what the boundary looked like with fewer learners.",
                        key="evo_step_main",
                    )

            fig, ax = plt.subplots(figsize=(8, 6))
            title = f"{cfg['technique']} — train set"

            if view_mode == "Hard boundary":
                if scrub_step is not None and scrub_step < get_max_step(model, cfg["technique"]):
                    plot_evolution_boundary(
                        model, cfg["technique"], scrub_step, X_train, y_train,
                        title=f"{cfg['technique']} after {scrub_step} learners",
                        ax=ax, cmap_light=CMAP_LIGHT, cmap_bold=CMAP_BOLD,
                        resolution=cfg["resolution"],
                    )
                else:
                    plot_decision_boundary(
                        model, X_train, y_train, title, ax=ax,
                        cmap_light=CMAP_LIGHT, cmap_bold=CMAP_BOLD,
                        resolution=cfg["resolution"],
                    )
                ax.scatter(
                    X_test[:, 0], X_test[:, 1], c=y_test, cmap=CMAP_BOLD,
                    edgecolor="black", s=70, marker="X", linewidth=0.7, label="test samples",
                )
                ax.legend(loc="upper right")
            elif view_mode == "Probability heatmap":
                ok = plot_proba_heatmap(
                    model, X_train, y_train,
                    f"{cfg['technique']} — probability surface",
                    ax=ax, cmap_bold=CMAP_BOLD, resolution=cfg["resolution"],
                )
                if ok is None:
                    st.warning(f"`{cfg['technique']}` does not expose `predict_proba`.")
            elif view_mode == "Uncertainty (entropy)":
                ok = plot_entropy_heatmap(
                    model, X_train, y_train,
                    f"{cfg['technique']} — predictive entropy",
                    ax=ax, cmap_bold=CMAP_BOLD, resolution=cfg["resolution"],
                )
                if ok is None:
                    st.warning(f"`{cfg['technique']}` does not expose `predict_proba`.")
            else:
                ok = plot_classifier_votes(
                    model, cfg["technique"], X_train, y_train,
                    f"{cfg['technique']} — fraction of learners voting class 1",
                    ax=ax, cmap_bold=CMAP_BOLD, resolution=cfg["resolution"],
                )
                if ok is None:
                    st.warning("This view is only available for tree-ensemble techniques "
                               "with accessible base estimators.")

            st.pyplot(fig, clear_figure=True)

            captions = {
                "Hard boundary": (
                    "Coloured regions show the model's argmax class for every point in the plane. "
                    "Drag the **step** slider above to watch the boundary evolve as the ensemble grows."
                ),
                "Probability heatmap": (
                    "Smooth probability surface: red ≈ class 1, blue ≈ class 0 (or viridis = max-prob "
                    "for multi-class). Crisper transitions ⇒ more confident model."
                ),
                "Uncertainty (entropy)": (
                    "Bright (yellow/white) regions = high entropy ⇒ the model is unsure here. "
                    "Look for narrow uncertainty bands along class boundaries."
                ),
                "Per-classifier votes": (
                    "Each base learner votes; the heat shows the *fraction* voting class 1. "
                    "Saturated regions ⇒ unanimous; mid-tones ⇒ disagreement among learners."
                ),
            }
            st.caption(captions[view_mode])
        else:
            st.info(
                "Decision boundary plots are only available for **2D** datasets. "
                "Switch to *Moons*, *Circles* or *Classification (2D blobs)* in the sidebar."
            )

    # ---- Weak learners → Ensemble
    if cfg["show_individual"] and "Weak learners → Ensemble" in tab_map:
        with tab_map["Weak learners → Ensemble"]:
            if X_train.shape[1] == 2:
                fig, ens_acc = plot_individual_vs_ensemble(
                    X_train, y_train, X_test, y_test,
                    n_learners=5,
                    max_depth=cfg["params"].get("max_depth", 4),
                )
                st.pyplot(fig, clear_figure=True)
                st.success(
                    f"Each individual tree is trained on a different bootstrap sample and "
                    f"makes different mistakes. The ensemble averages them out and reaches "
                    f"**{ens_acc:.2%}** accuracy — usually better than any single tree."
                )
            else:
                st.info("This visualisation requires a 2D dataset.")

    # ---- Confusion matrix
    with tab_map["Confusion matrix"]:
        cm = confusion_matrix(y_test, y_pred)
        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            plot_confusion(cm, ds.target_names, ax=ax)
            st.pyplot(fig, clear_figure=True)
        with col2:
            st.markdown("**Classification report**")
            report = classification_report(
                y_test, y_pred, target_names=ds.target_names, zero_division=0, output_dict=True
            )
            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    # ---- Feature importance
    with tab_map["Feature importance"]:
        ax = plot_feature_importance(model, ds.feature_names)
        if ax is not None:
            st.pyplot(ax.figure, clear_figure=True)
            st.caption(
                "Tree-based ensembles measure how much each feature reduces impurity, "
                "averaged across all trees. Bigger bar = more useful feature."
            )
        else:
            st.info(
                f"`{cfg['technique']}` does not expose `feature_importances_`. "
                "Try Random Forest, Gradient Boosting, AdaBoost or Bagging."
            )

    # ---- Learning curve
    with tab_map["Learning curve"]:
        if cfg["technique"] in ("Bagging", "Random Forest", "AdaBoost", "Gradient Boosting", "XGBoost"):
            n_max = max(cfg["params"]["n_estimators"], 50)
            n_max = min(n_max, 200)

            def factory(n: int):
                p = dict(cfg["params"])
                p["n_estimators"] = n
                return build_model(cfg["technique"], p)

            with st.spinner("Sweeping ensemble size..."):
                fig = plot_learning_curve(factory, X_train, y_train, X_test, y_test, n_max)
            st.pyplot(fig, clear_figure=True)
            st.caption(
                "Watch how train/test accuracy evolve as we add more estimators. "
                "Bagging-style methods plateau, boosting methods can keep improving — "
                "but may eventually overfit if you go too far."
            )
        else:
            st.info(
                "The learning curve over `n_estimators` only applies to bagging / boosting "
                "ensembles. Pick *Bagging*, *Random Forest*, *AdaBoost*, *Gradient Boosting* "
                "or *XGBoost*."
            )

    # ---- How it works
    with tab_map["How it works"]:
        st.markdown(f"### {cfg['technique']}")
        st.markdown(info["intuition"])
        st.latex(info["math"])
        st.markdown("#### Algorithm in plain English")

        steps = {
            "Single Decision Tree (baseline)": [
                "Find the feature & threshold that best splits the data (Gini / entropy).",
                "Recurse on each child until `max_depth` or purity is reached.",
                "Predict by walking a sample down the tree to its leaf.",
            ],
            "Bagging": [
                "Draw `B` bootstrap samples (with replacement) from the training set.",
                "Fit one base learner (e.g. tree) on each sample — independently, in parallel.",
                "Predict by majority vote / averaging across all `B` learners.",
            ],
            "Random Forest": [
                "Same as bagging, plus: at each split only consider a random feature subset.",
                "Trees become more decorrelated → variance drops further.",
                "Aggregate predictions by majority vote.",
            ],
            "AdaBoost": [
                "Start with equal weights for every training point.",
                "Train a weak learner; up-weight the points it got wrong.",
                "Repeat T times. Final prediction = weighted vote of all weak learners.",
            ],
            "Gradient Boosting": [
                "Start with a constant prediction (e.g. log-odds of the target).",
                "Compute the negative gradient of the loss for every sample.",
                "Fit a small tree to those gradients and add it to the ensemble (scaled by η).",
                "Repeat M times.",
            ],
            "XGBoost": [
                "Initialise predictions to a constant (e.g. log-odds).",
                "For every sample compute the gradient g and Hessian h of the loss.",
                "Build a tree that maximises the *similarity gain* using both g and h (Newton step), with γ as the minimum gain to split and λ as L2 leaf-weight regularisation.",
                "At each node consider only a random column subsample (`colsample_bytree`).",
                "Optionally subsample rows (`subsample`) for stochastic boosting.",
                "Add the new tree scaled by the learning rate η to the ensemble.",
                "Repeat for M boosting rounds (with early stopping in real-world use).",
            ],
            "Voting (hard / soft)": [
                "Train K *different* models on the same data (LogReg, KNN, Tree, NB...).",
                "Hard voting: predict the class chosen by most models.",
                "Soft voting: average their predicted probabilities, pick argmax.",
            ],
            "Stacking": [
                "Train K base learners using cross-validation to get out-of-fold predictions.",
                "Use those predictions as features for a meta-model (e.g. Logistic Regression).",
                "At inference, base models predict and the meta-model combines them.",
            ],
        }
        for i, step in enumerate(steps[cfg["technique"]], 1):
            st.markdown(f"**{i}.** {step}")

    # ---- Techniques overview (always visible)
    with tab_map["📚 Techniques overview"]:
        render_techniques_overview(current_technique=cfg["technique"])

    # ---- Compare A vs B
    if cfg["compare_with"] and cfg["technique_b"] and "Compare A vs B" in tab_map:
        with tab_map["Compare A vs B"]:
            tech_a = cfg["technique"]
            tech_b = cfg["technique_b"]
            params_b = default_params(cfg["seed"])

            with st.spinner(f"Training {tech_b} for comparison..."):
                model_b = build_model(tech_b, params_b)
                t0 = time.time()
                model_b.fit(X_train, y_train)
                train_time_b = time.time() - t0
                y_pred_b = model_b.predict(X_test)

            acc_b = accuracy_score(y_test, y_pred_b)
            f1_b = f1_score(y_test, y_pred_b, average="weighted", zero_division=0)
            prec_b = precision_score(y_test, y_pred_b, average="weighted", zero_division=0)
            rec_b = recall_score(y_test, y_pred_b, average="weighted", zero_division=0)

            st.markdown(
                f"<span class='technique-badge'>A · {tech_a}</span>"
                f"<span class='technique-badge'>B · {tech_b}</span>",
                unsafe_allow_html=True,
            )

            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"**A · {tech_a}** *(your settings)*")
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Accuracy", f"{acc:.3f}")
                a2.metric("Precision", f"{prec:.3f}")
                a3.metric("Recall", f"{rec:.3f}")
                a4.metric("F1", f"{f1:.3f}")
            with colB:
                st.markdown(f"**B · {tech_b}** *(default hparams)*")
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Accuracy", f"{acc_b:.3f}", delta=f"{(acc_b-acc)*100:+.1f} pp")
                b2.metric("Precision", f"{prec_b:.3f}", delta=f"{(prec_b-prec)*100:+.1f} pp")
                b3.metric("Recall", f"{rec_b:.3f}", delta=f"{(rec_b-rec)*100:+.1f} pp")
                b4.metric("F1", f"{f1_b:.3f}", delta=f"{(f1_b-f1)*100:+.1f} pp")

            st.markdown("---")

            if X_train.shape[1] == 2:
                st.markdown("#### Decision boundaries")
                fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
                plot_decision_boundary(
                    model, X_train, y_train, f"A · {tech_a}",
                    ax=axes[0], cmap_light=CMAP_LIGHT, cmap_bold=CMAP_BOLD,
                    resolution=cfg["resolution"],
                )
                plot_decision_boundary(
                    model_b, X_train, y_train, f"B · {tech_b}",
                    ax=axes[1], cmap_light=CMAP_LIGHT, cmap_bold=CMAP_BOLD,
                    resolution=cfg["resolution"],
                )
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("Decision-boundary side-by-side requires a 2D dataset.")

            st.markdown("#### Confusion matrices")
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            plot_confusion(confusion_matrix(y_test, y_pred), ds.target_names, ax=axes[0])
            axes[0].set_title(f"A · {tech_a}", fontsize=11, fontweight="bold")
            plot_confusion(confusion_matrix(y_test, y_pred_b), ds.target_names, ax=axes[1])
            axes[1].set_title(f"B · {tech_b}", fontsize=11, fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

            disagree = int(np.sum(y_pred != y_pred_b))
            agree_pct = 1.0 - disagree / len(y_test)
            winner = tech_a if acc >= acc_b else tech_b
            st.success(
                f"**Agreement** between A and B on the test set: {agree_pct:.1%} "
                f"({disagree} differing predictions out of {len(y_test)}). "
                f"Higher-accuracy model: **{winner}** "
                f"(A {acc:.3f} vs B {acc_b:.3f}, B trained in {train_time_b*1000:.0f} ms)."
            )

    # ---- Full comparison
    if cfg["show_compare"] and "Full comparison" in tab_map:
        with tab_map["Full comparison"]:
            df = run_full_comparison(X_train, X_test, y_train, y_test)
            df_sorted = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

            st.markdown("#### Results table")
            st.dataframe(
                df_sorted.style.background_gradient(subset=["Accuracy", "F1"], cmap="Greens")
                              .background_gradient(subset=["Train time (s)"], cmap="Reds"),
                use_container_width=True,
            )

            fig, ax = plt.subplots(figsize=(10, 5))
            metrics_long = df.melt(
                id_vars="Technique",
                value_vars=["Accuracy", "Precision", "Recall", "F1"],
                var_name="Metric", value_name="Score",
            )
            sns.barplot(
                data=metrics_long, x="Technique", y="Score", hue="Metric",
                palette="viridis", ax=ax,
            )
            ax.set_ylim(0, 1.05)
            ax.set_title("All techniques compared", fontsize=13, fontweight="bold")
            ax.set_xlabel("")
            plt.xticks(rotation=25, ha="right")
            ax.legend(loc="lower right", ncol=4)
            st.pyplot(fig, clear_figure=True)

            best = df_sorted.iloc[0]
            st.success(
                f"**Winner on this dataset:** `{best['Technique']}` — "
                f"accuracy {best['Accuracy']:.3f}, F1 {best['F1']:.3f}, "
                f"trained in {best['Train time (s)']*1000:.0f} ms."
            )

    # --- Footer
    st.markdown(
        '<div class="footer">'
        'Built with Streamlit · scikit-learn · matplotlib · seaborn — '
        'deploy-ready for Render via GitHub.'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
