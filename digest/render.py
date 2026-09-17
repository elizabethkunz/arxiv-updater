import json
import re
from datetime import date, timedelta
from itertools import groupby

from jinja2 import Environment, FileSystemLoader

from .data import ROOT, STATE, load_csv

SITE = ROOT / "site"
DAYS = STATE / "days"

env = Environment(loader=FileSystemLoader(ROOT / "templates"),
                  autoescape=lambda name: ".html" in (name or ""), trim_blocks=True, lstrip_blocks=True)


def short_authors(authors: list[str]) -> str:
    """First two and last author, per the email spec."""
    if len(authors) <= 3:
        return ", ".join(authors) + "."
    return f"{authors[0]}, {authors[1]}, ..., {authors[-1]}."


env.filters["short_authors"] = short_authors
env.filters["pct"] = lambda s: f"{100 * s:.1f}%"

# score colour ramp, red -> amber -> green: (percent, hue, saturation %, lightness offset %, punch %).
# Punch is applied as darker in light mode and brighter in dark mode (--score-sign in base.html.j2).
_TONE_STOPS = [(30, 4, 70, 8, 0), (60, 28, 80, 5, 0), (70, 46, 85, 3, 0),
               (80, 96, 72, 1, 2), (90, 128, 80, 0, 5), (100, 148, 92, 0, 8)]


def score_tone(score: float) -> str:
    """Inline custom properties that colour a .score / .bigscore element."""
    v = min(100.0, max(30.0, 100 * score))
    a, b = next((a, b) for a, b in zip(_TONE_STOPS, _TONE_STOPS[1:]) if v <= b[0])
    t = (v - a[0]) / (b[0] - a[0])
    h, s, off, punch = (a[k] + (b[k] - a[k]) * t for k in range(1, 5))
    return f"--h:{h:.0f};--s:{s:.0f}%;--l:calc(var(--score-l) + {off:.1f}% + {punch:.1f}% * var(--score-sign))"


env.filters["score_tone"] = score_tone

_TEX = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ",
        "eta": "η", "theta": "θ", "kappa": "κ", "lambda": "λ", "mu": "µ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
        "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ",
        "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω", "hbar": "ħ",
        "times": "×", "cdot": "·", "approx": "≈", "sim": "~", "simeq": "≃", "propto": "∝", "pm": "±", "leq": "≤",
        "le": "≤", "geq": "≥", "ge": "≥", "ll": "≪", "gg": "≫", "neq": "≠", "to": "→", "rightarrow": "→",
        "infty": "∞", "dagger": "†", "langle": "⟨", "rangle": "⟩", "ell": "ℓ", "partial": "∂", "%": "%"}
_SUP = str.maketrans("0123456789+-=()n*", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ*")
_SUB = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")


def _script(m: re.Match) -> str:
    table, body = _SUP if m[1] == "^" else _SUB, m[2] or m[3]
    if all(ord(c) in table for c in body):
        return body.translate(table)
    # most letters have no Unicode sub/superscript: T_1 -> T₁, but omega_q stays ω_q
    return m[1] + (body if len(body) == 1 or body.isalnum() else f"({body})")


def detex(text: str) -> str:
    """Readable plain text for mail clients, which cannot run KaTeX: $T_1 = 50$ -> T₁ = 50."""
    def math(m: re.Match) -> str:
        s = re.sub(r"\\(?:mathrm|text|mathbf|mathcal|mathit|operatorname)\s*\{([^{}]*)\}", r"\1", m[1])
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"√(\1)", s)
        s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
        s = re.sub(r"\\([A-Za-z]+|%)", lambda c: _TEX.get(c[1], c[1]), s)
        s = re.sub(r"\\[,;:! ]", " ", s)
        s = re.sub(r"([\^_])(?:\{([^{}]*)\}|(\S))", _script, s)
        return s.replace("{", "").replace("}", "")
    return re.sub(r"\$\$?(.+?)\$\$?", math, text)


env.filters["detex"] = detex


def render_email(day: dict, cfg: dict) -> tuple[str, str, str]:
    """(subject, plain text, html) for the papers flagged emailed."""
    papers = [p for p in day["papers"] if p["emailed"]]
    ctx = {"day": day, "papers": papers, "base_url": cfg["site"]["base_url"].rstrip("/"),
           "site_min": round(100 * cfg["site"]["min_score"])}
    # email.style: "styled" (site look, serif) or "plain" (bold titles, otherwise unformatted)
    html = "email_plain.html.j2" if cfg["email"].get("style") == "plain" else "email.html.j2"
    if papers:
        subject = f"arXiv digest {day['date']} — {len(papers)} papers (top: {100 * papers[0]['score']:.0f}%)"
    else:
        subject = f"arXiv digest {day['date']} — nothing above threshold"
    return (subject,
            env.get_template("email.txt.j2").render(**ctx),
            env.get_template(html).render(**ctx))


def _histogram(scores: list[float], min_score: float, bins: int = 20, height: int = 28) -> list[dict]:
    """Bar heights for the per-day score histogram (sqrt-scaled: most papers sit near 0%)."""
    counts = [0] * bins
    for s in scores:
        counts[min(int(s * bins), bins - 1)] += 1
    top = max(counts, default=0) or 1
    return [{"h": max(1, round(height * (c / top) ** 0.5)) if c else 0, "hi": i / bins >= min_score}
            for i, c in enumerate(counts)]


def _first_sentence(abstract: str) -> str:
    m = re.search(r"(?<!et al)(?<!i\.e)(?<!e\.g)(?<!Fig)\.\s+(?=[A-Z])", abstract)
    return abstract[: m.start() + 1] if m else abstract


def _database(cfg: dict) -> list[dict]:
    """Liked papers ranked by the model's own score for them; empty until `train` has run."""
    from . import model
    if not model.MODEL_PATH.exists():
        return []
    liked = load_csv(ROOT / cfg["liked_csv"])  # scored through model.score so author features apply
    rows = [{"title": p.title, "id": p.id, "score": s["score"], "first": _first_sentence(p.abstract)}
            for p, s in zip(liked, model.score(liked))]
    return sorted(rows, key=lambda r: -r["score"])


def _by_month(days: list[dict]) -> list[dict]:
    """Newest-first days grouped into [{key: '2026-09', label: 'September 2026', days: [...]}]."""
    out = []
    for key, group in groupby(days, key=lambda d: d["date"][:7]):
        out.append({"key": key, "label": date.fromisoformat(key + "-01").strftime("%B %Y"), "days": list(group)})
    return out


# windows offered by the "Top" sort on the index: (key, label, days)
_RANGES = [("week", "Week", 7), ("month", "Month", 30), ("half", "6 months", 182), ("year", "Year", 365)]


def _top(days: list[dict], window: int, n: int = 50) -> tuple[list[dict], int]:
    """(best n papers, papers considered) among those announced in the `window` days up to the newest day."""
    if not days:
        return [], 0
    cutoff = (date.fromisoformat(days[0]["date"]) - timedelta(days=window - 1)).isoformat()
    seen = {}  # newest day first, so a paper seen on two days keeps its latest entry
    for d in days:
        if d["date"] >= cutoff:
            for p in d["papers"]:
                seen.setdefault(p["id"], {**p, "day": d["date"]})
    return sorted(seen.values(), key=lambda p: -p["score"])[:n], len(seen)


def _vectors(day: dict, model_name: str) -> dict:
    """{paper id: embedding} for one day, kept in state/vectors/<date>.npz.

    state/emb_cache is too big to commit, so without these small per-day files the GitHub Action
    would re-embed every paper on the site at each build. A day's file only changes if the day does.
    """
    import numpy as np
    from .embed import embed
    path = STATE / "vectors" / f"{day['date']}.npz"
    have = {}
    if path.exists():
        with np.load(path) as z:
            if str(z["model"]) == model_name:
                have = dict(zip(z["ids"].tolist(), z["vectors"].astype(np.float32)))
    ids = [p["id"] for p in day["papers"]]
    missing = [p for p in day["papers"] if p["id"] not in have]
    if missing:
        have.update(zip((p["id"] for p in missing),
                        embed([f"{p['title']}. {p['abstract']}" for p in missing], model_name)))
    if ids and (missing or len(have) != len(ids)):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, model=model_name, ids=np.array(ids), vectors=np.stack([have[i] for i in ids]).astype(np.float16))
    return {i: have[i] for i in ids}


def _similar(days: list[dict], model_name: str, n: int = 5) -> dict:
    """{paper id: its n nearest papers on the site (cosine similarity of the title+abstract embeddings),
    ordered by match score}."""
    import numpy as np
    papers, vectors = {}, {}  # newest day first, so a paper seen on two days keeps its latest entry
    for d in days:
        for pid, v in _vectors(d, model_name).items():
            vectors.setdefault(pid, v)
        for p in d["papers"]:
            papers.setdefault(p["id"], {"id": p["id"], "title": p["title"], "day": d["date"], "score": p["score"]})
    if len(papers) < 2:
        return {}
    ids = list(papers)
    V = np.stack([vectors[i] for i in ids]).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)  # float16 storage leaves them slightly off unit length
    S = V @ V.T
    np.fill_diagonal(S, -1)
    # the n nearest neighbours, then shown best match first
    return {pid: sorted(({**papers[ids[j]], "similarity": float(S[i, j])} for j in np.argsort(-S[i])[:n]),
                        key=lambda s: -s["score"])
            for i, pid in enumerate(ids)}


def build_site(cfg: dict) -> None:
    """Regenerate the whole static site from state/days/*.json."""
    days = sorted((json.loads(f.read_text(encoding="utf-8")) for f in DAYS.glob("*.json")),
                  key=lambda d: d["date"], reverse=True)
    min_score = cfg["site"]["min_score"]
    similar = _similar(days, cfg["embedding_model"])
    for d in days:
        # the index only lists papers >= min_score; each day page lists every emailed paper too
        d["listed"] = [p for p in d["papers"] if p["score"] >= min_score]
        d["hist"] = _histogram(d.get("all_scores", []), min_score)
        d["label"] = date.fromisoformat(d["date"]).strftime("%a %d")
    tops = []
    for key, label, window in _RANGES:
        papers, total = _top(days, window)
        tops.append({"key": key, "label": label, "papers": papers, "total": total})
    ctx = {"site": {"page_size": 100, **cfg["site"]}, "min_score": min_score,
           "months": _by_month(days), "tops": tops}

    (SITE / "days").mkdir(parents=True, exist_ok=True)
    # the index lists every day, by month; its script reveals site.page_size papers at a time
    (SITE / "index.html").write_text(env.get_template("index.html.j2").render(root="", **ctx), encoding="utf-8")
    # the archive page was folded into the index; keep old links working
    (SITE / "archive.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Moved</title><meta http-equiv="refresh" content="0; url=index.html">'
        '<link rel="canonical" href="index.html"><a href="index.html">The archive is now part of the index.</a>',
        encoding="utf-8")
    (SITE / "database.html").write_text(
        env.get_template("database.html.j2").render(papers=_database(cfg), root="", **ctx), encoding="utf-8")

    search = {}  # newest day first, so a paper seen on two days keeps its latest entry
    for d in days:
        d["listed"] = d["papers"]
        (SITE / "days" / f"{d['date']}.html").write_text(
            env.get_template("day.html.j2").render(day=d, root="../", **ctx), encoding="utf-8")
        for p in d["papers"]:
            (SITE / p["id"]).mkdir(exist_ok=True)
            (SITE / p["id"] / "index.html").write_text(
                env.get_template("paper.html.j2").render(p=p, day=d, similar=similar.get(p["id"], []), root="../", **ctx), encoding="utf-8")
            search.setdefault(p["id"], {"id": p["id"], "t": p["title"], "a": ", ".join(p["authors"]),
                                        "d": d["date"], "s": round(100 * p["score"], 1),
                                        "x": p["summary"]["summary"]})
    # a script rather than JSON so the search also works when the site is opened from file://
    blob = json.dumps(list(search.values()), ensure_ascii=False).replace("</", "<\\/")
    (SITE / "search.js").write_text(f"window.PAPERS = {blob};", encoding="utf-8")
