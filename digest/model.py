"""Positive-unlabeled classifier: liked papers vs. random arXiv background.

The background is 'unlabeled', not 'disliked' — it contains some papers Liz would like, which
pulls the scores of genuinely good papers down a little. The score is the class-balanced
probability P(liked-like | x): 50% means "as much like the liked list as like generic arXiv".
(Elkan-Noto rescaling by c = E[g | liked] was tried and dropped: it clips half of all good
papers to 100% and destroys the ranking at the top, which is the part that matters.)
"""
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .authors import AuthorFeatures
from .data import STATE, Paper, load_config
from .embed import embed

MODEL_PATH = STATE / "model.joblib"
DISLIKE_WEIGHT = 10.0


def _clf():
    return LogisticRegression(C=1.0,class_weight="balanced", max_iter=5000)


def _features(X, A, w_authors: float, w_established: float):
    """Embedding + author block. The two weights set how much authors count relative to the text."""
    return sp.hstack([sp.csr_matrix(X), w_authors * A[:, :-1], w_established * A[:, -1]]).tocsr()


def _display(p, sharpen: float):
    """Monotone rescale of the raw probability: sigmoid(sharpen * logit(p)). 50% stays 50% and the
    ranking is unchanged; >1 pushes confident scores toward 0/100. The regularised classifier is
    under-confident (a Platt fit on held-out scores gives a slope of ~1.9), but full calibration
    squeezes every good paper into 98-100%, so `score_sharpen` in config.yaml sits in between."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return 1.0 / (1.0 + np.exp(-sharpen * np.log(p / (1 - p))))


def train(liked: list[Paper], background: list[Paper], disliked: list[Paper], model_name: str) -> dict:
    papers = liked + background + disliked
    E = embed([p.text for p in papers], model_name)
    acfg = load_config().get("authors") or {}
    w_a, w_e = float(acfg.get("weight", 0.0)), float(acfg.get("established_weight", 0.0))
    sharpen = float(load_config().get("score_sharpen", 1.0))
    af = AuthorFeatures(acfg.get("established")).fit([p.authors for p in papers])
    X = _features(E, af.transform([p.authors for p in papers]), w_a, w_e)
    y = np.r_[np.ones(len(liked)), np.zeros(len(background) + len(disliked))]
    w = np.r_[np.ones(len(liked) + len(background)), np.full(len(disliked), DISLIKE_WEIGHT)]

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(_clf(), X, y, cv=cv, method="predict_proba", params={"sample_weight": w})[:, 1]
    p = _display(p, sharpen)
    pos, bg = p[y == 1], p[y == 0]
    report = {
        "embedding_model": model_name,
        "w_authors": w_a, "w_established": w_e, "score_sharpen": sharpen,
        "n_liked": len(liked), "n_background": len(background), "n_disliked": len(disliked),
        "cv_auc": float(roc_auc_score(y, p)),
        "cv_avg_precision": float(average_precision_score(y, p)),
        "held_out_liked_median": float(np.median(pos)),
        "held_out_liked_frac_ge_80": float(np.mean(pos >= 0.8)),
        "held_out_liked_frac_ge_50": float(np.mean(pos >= 0.5)),
        "background_frac_ge_80": float(np.mean(bg >= 0.8)),
        "background_frac_ge_50": float(np.mean(bg >= 0.5)),
    }

    clf = _clf().fit(X, y, sample_weight=w)
    STATE.mkdir(exist_ok=True)
    joblib.dump({
        "clf": clf, "embedding_model": model_name,
        "authors": af, "w_authors": w_a, "w_established": w_e, "score_sharpen": sharpen,
        "liked_X": E[: len(liked)],
        "liked_titles": [p.title for p in liked],
        "liked_ids": [p.id for p in liked],
        "report": report,
    }, MODEL_PATH)
    report["_cv_scores"] = (p, y)  # for inspection by the caller; not persisted
    return report


def score(papers: list[Paper]) -> list[dict]:
    """Score + two nearest liked papers for each input paper."""
    m = joblib.load(MODEL_PATH)
    X = embed([p.text for p in papers], m["embedding_model"])
    F = X
    if "authors" in m:  # models trained before author features score on text alone
        F = _features(X, m["authors"].transform([p.authors for p in papers]), m["w_authors"], m["w_established"])
    probs = _display(m["clf"].predict_proba(F)[:, 1], m.get("score_sharpen", 1.0))
    sims = X @ m["liked_X"].T
    out = []
    for i in range(len(papers)):
        nn = np.argsort(-sims[i])[:2]
        out.append({
            "score": float(probs[i]),
            "similar_to": [{"title": m["liked_titles"][j], "id": m["liked_ids"][j],
                            "similarity": float(sims[i, j])} for j in nn],
        })
    return out
