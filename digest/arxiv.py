"""arXiv metadata via OAI-PMH (https://oaipmh.arxiv.org).

The search API (export.arxiv.org/api) rate-limits and 503s heavily; OAI-PMH is the interface
arXiv recommends for harvesting. Records are datestamped with the day they were announced or
replaced, so "new on day D" = datestamp D and created within the previous week (a quick v2 of
a fresh paper also matches; seen.json stops it being reported twice).
"""
import json
import random
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import requests

from .data import Paper, STATE, clean

OAI = "https://oaipmh.arxiv.org/oai"
NS = {"o": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}
DELAY = 3.1
NEW_WITHIN_DAYS = 7


def _set_spec(category: str) -> str:
    """'quant-ph' -> 'physics:quant-ph', 'cond-mat.supr-con' -> 'physics:cond-mat:supr-con'."""
    return "physics:" + category.replace(".", ":")


def _get(params: dict) -> ET.Element:
    for attempt in range(5):
        try:
            r = requests.get(OAI, params=params, timeout=300)
            if r.status_code == 200:
                time.sleep(DELAY)
                return ET.fromstring(r.content)
            status = r.status_code
        except requests.RequestException as e:
            status = type(e).__name__
        print(f"    arXiv OAI {status}, retry {attempt + 1}/5", flush=True)
        time.sleep(DELAY * 2 ** (attempt + 1))
    raise RuntimeError(f"arXiv OAI-PMH kept failing for {params}")


def _parse(rec: ET.Element) -> Paper:
    m = rec.find("o:metadata/a:arXiv", NS)
    authors = [clean(f"{a.findtext('a:forenames', '', NS)} {a.findtext('a:keyname', '', NS)}")
               for a in m.findall("a:authors/a:author", NS)]
    paper = Paper(
        id=m.findtext("a:id", "", NS),
        title=clean(m.findtext("a:title", "", NS)),
        authors=authors,
        abstract=clean(m.findtext("a:abstract", "", NS)),
        categories=m.findtext("a:categories", "", NS).split(),
        published=m.findtext("a:created", "", NS),
    )
    return paper


def _is_new(paper: Paper, start: date) -> bool:
    """Replacements of old papers share the datestamp; keep only genuinely new submissions.

    <created> alone is unreliable (some replaced papers carry a recent date), so also require
    the ID's YYMM to match the announcement month (or the month before, across a month boundary).
    """
    months = {f"{start:%y%m}", f"{start - timedelta(days=NEW_WITHIN_DAYS):%y%m}"}
    try:
        recent = date.fromisoformat(paper.published) >= start - timedelta(days=NEW_WITHIN_DAYS)
    except ValueError:
        return False
    return recent and paper.id[:4] in months


def fetch(categories: list[str], start: date, end: date, new_only: bool = True) -> list[Paper]:
    """Papers announced between start and end (inclusive); cross-lists deduped."""
    papers = {}
    for c in categories:
        params = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": _set_spec(c),
                  "from": start.isoformat(), "until": end.isoformat()}
        while True:
            root = _get(params)
            for rec in root.findall("o:ListRecords/o:record", NS):
                if rec.find("o:metadata", NS) is None:  # deleted record
                    continue
                p = _parse(rec)
                if not new_only or _is_new(p, start):
                    papers.setdefault(p.id, p)
            token = root.findtext("o:ListRecords/o:resumptionToken", "", NS).strip()
            if not token:
                break
            params = {"verb": "ListRecords", "resumptionToken": token}
    return list(papers.values())


def load_background(cfg: dict, exclude_ids: set[str]) -> list[Paper]:
    """Random past papers used as 'not liked' background. Cached on disk, resumable."""
    cache = STATE / "background.json"
    STATE.mkdir(exist_ok=True)
    days = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    b = cfg["background"]
    rng = random.Random(len(days))
    while len(days) < b["n_days"]:
        d = date.today() - timedelta(days=rng.randint(10, b["months_back"] * 30))
        if d.weekday() >= 5 or d.isoformat() in days:
            continue
        # replacements are fine here: any paper works as background
        got = fetch(cfg["categories"], d, d + timedelta(days=b["window_days"] - 1), new_only=False)
        print(f"  background {d}: {len(got)} papers", flush=True)
        days[d.isoformat()] = [p.to_dict() for p in got]
        cache.write_text(json.dumps(days), encoding="utf-8")
    papers = {d["id"]: Paper(**d) for day in days.values() for d in day}
    papers = [p for i, p in sorted(papers.items()) if i not in exclude_ids]
    random.Random(0).shuffle(papers)
    return papers[: b["max_papers"]]
