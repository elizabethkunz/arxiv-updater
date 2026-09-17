import json
import os

import requests
from bs4 import BeautifulSoup

from .data import STATE, Paper

SUMMARY_DIR = STATE / "summaries"

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Exactly two sentences summarizing the whole paper."},
        "motivation": {"type": "string", "description": "One or two sentences: why this work was done."},
        "method": {"type": "string", "description": "One or two sentences: what was done and how."},
        "result": {"type": "string", "description": "One or two sentences: the key quantitative result."},
    },
    "required": ["summary", "motivation", "method", "result"],
    "additionalProperties": False,
}

SYSTEM = (
    "You write digest entries for an experimental physicist who works on superconducting qubits "
    "and also follows atomic physics, waveguide QED and open-quantum-systems theory. "
    "Write for an expert: be specific and quantitative (numbers, materials, fidelities, platforms), "
    "skip generic framing like 'quantum computers are promising', and never invent results that "
    "are not in the text. 'summary' is exactly two sentences; the other fields are one or two sentences each. "
    # the site typesets $...$ with KaTeX; the emails convert it to plain text (render.detex)
    "Write every symbol and mathematical expression as inline LaTeX between single dollar signs, as in a paper: "
    r"$T_1 = 50$ µs, $\gamma = \omega_q/\alpha$, $\eta_\mathrm{max} \propto \gamma^{0.61}$, $3\times10^{-4}$, "
    r"$g/2\pi \approx 3$ MHz, NbSe$_2$, $^{28}$Si. Never spell out Greek letters or write x_y, x^y, 1e-5 or sqrt() "
    "in plain text. Plain numbers with units and percentages stay outside math, and never use a dollar sign "
    "for anything other than delimiting math."
)


def fetch_full_text(arxiv_id: str) -> str | None:
    """Plain text of arXiv's HTML rendering (no bibliography), or None if unavailable."""
    try:
        r = requests.get(f"https://arxiv.org/html/{arxiv_id}", timeout=60)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    article = BeautifulSoup(r.text, "html.parser").find("article")
    if article is None:
        return None
    for tag in article.select(".ltx_bibliography, nav, script, style, math annotation, .ltx_authors"):
        tag.decompose()
    text = " ".join(article.get_text(" ").split())
    return text if len(text) > 3000 else None


def summarize(paper: Paper, cfg: dict) -> dict | None:
    """Cached structured summary. Returns None when no API credentials are configured."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / f"{paper.id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None

    import anthropic

    s = cfg["summarizer"]
    body = fetch_full_text(paper.id)
    source = "full text"
    if body is None or len(body) > s["max_chars"]:
        if body is not None:
            print(f"  {paper.id}: full text too long ({len(body)} chars), using abstract")
        body, source = paper.abstract, "abstract"

    try:
        response = anthropic.Anthropic().messages.create(
            model=s["model"],
            max_tokens=2000,
            system=SYSTEM,
            messages=[{"role": "user", "content":
                       f"Title: {paper.title}\n\nAbstract: {paper.abstract}\n\n"
                       f"Paper ({source}):\n{body}\n\nWrite the digest entry for this paper."}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
    except anthropic.RateLimitError as e:
        print(f"  {paper.id}: rate limited, skipping summary ({e.message})")
        return None
    except anthropic.APIStatusError as e:
        print(f"  {paper.id}: API error {e.status_code}, skipping summary ({e.message})")
        return None
    except anthropic.APIConnectionError:
        print(f"  {paper.id}: network error, skipping summary")
        return None
    if response.stop_reason != "end_turn":
        print(f"  {paper.id}: summary stopped with {response.stop_reason}, skipping")
        return None

    data = json.loads(next(b.text for b in response.content if b.type == "text"))
    data.update(source=source, generated_by=s["model"])
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def fallback_summary(paper: Paper) -> dict:
    """Used when no summary could be generated: first two sentences of the abstract."""
    sentences = paper.abstract.replace("et al. ", "et al. ").split(". ")
    text = ". ".join(sentences[:2]).rstrip(".") + "."
    return {"summary": text, "motivation": "", "method": "", "result": "",
            "source": "abstract", "generated_by": "none (abstract excerpt)"}
