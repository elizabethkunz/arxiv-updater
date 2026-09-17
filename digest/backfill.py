"""Index past days without sending email:  python -m digest.backfill 2026-03-09 2026-03-20

`run --date` finds papers by OAI datestamp, which moves every time a paper is revised, so for
days more than a few weeks back it silently loses everything that later got a v2. This uses the
search API's submittedDate (the v1 date, which never changes) and files each paper under the day
it would have been announced. The search API is flaky under load, hence the slow, retried paging.
"""
import json
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date, datetime, timedelta

import requests
from dotenv import load_dotenv

from . import model, render, summarize
from .data import ROOT, Paper, clean, load_config

API = "http://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
PAGE = 200


def _cutoff_shift(t: datetime) -> timedelta:
    """Hours that move the 14:00 US Eastern cutoff to midnight: 18:00 UTC in summer, 19:00 UTC in winter."""
    def nth_sunday(month: int, n: int) -> datetime:
        first = datetime(t.year, month, 1)
        return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))
    dst = nth_sunday(3, 2) <= t < nth_sunday(11, 1)
    return timedelta(hours=6 if dst else 5)


def announce_day(published_utc: datetime) -> date:
    """Day a submission shows up in the digest (OAI datestamp = the day after the 20:00 ET mailing)."""
    batch = (published_utc + _cutoff_shift(published_utc)).date()      # the cutoff that closes this submission's batch
    while batch.weekday() >= 5:                        # weekend submissions join Monday's batch
        batch += timedelta(days=1)
    out = batch + timedelta(days=1)
    while out.weekday() >= 5:                          # Friday's batch is mailed Sunday night
        out += timedelta(days=1)
    return out


def _page(query: str, start: int) -> list[ET.Element]:
    params = {"search_query": query, "start": start, "max_results": PAGE,
              "sortBy": "submittedDate", "sortOrder": "ascending"}
    for attempt in range(6):
        try:
            r = requests.get(API, params=params, timeout=120)
            if r.status_code == 200:
                entries = ET.fromstring(r.content).findall("a:entry", ATOM)
                if entries or attempt >= 2:  # the API sometimes returns an empty page by mistake
                    time.sleep(4)
                    return entries
            print(f"    arXiv API {r.status_code}/{'empty'}, retry {attempt + 1}", flush=True)
        except requests.RequestException as e:
            print(f"    arXiv API {type(e).__name__}, retry {attempt + 1}", flush=True)
        time.sleep(8 * (attempt + 1))
    raise RuntimeError("arXiv search API kept failing")


def fetch(categories: list[str], first: date, last: date) -> dict[date, list[Paper]]:
    lo = datetime.combine(first - timedelta(days=5), datetime.min.time())
    hi = datetime.combine(last, datetime.min.time())
    cats = " OR ".join(f"cat:{c}" for c in categories)
    query = f"({cats}) AND submittedDate:[{lo:%Y%m%d%H%M} TO {hi:%Y%m%d%H%M}]"
    by_day, seen, start = defaultdict(list), set(), 0
    while True:
        entries = _page(query, start)
        for e in entries:
            pid = e.findtext("a:id", "", ATOM).rsplit("/abs/", 1)[-1].rsplit("v", 1)[0]
            published = e.findtext("a:published", "", ATOM)
            day = announce_day(datetime.fromisoformat(published.replace("Z", "")))
            if pid in seen or not first <= day <= last:
                continue
            seen.add(pid)
            by_day[day].append(Paper(
                id=pid, title=clean(e.findtext("a:title", "", ATOM)),
                authors=[clean(a.findtext("a:name", "", ATOM)) for a in e.findall("a:author", ATOM)],
                abstract=clean(e.findtext("a:summary", "", ATOM)),
                categories=[c.get("term") for c in e.findall("a:category", ATOM)],
                published=published[:10]))
        print(f"  fetched {start + len(entries)} records", flush=True)
        if len(entries) < PAGE:
            return by_day
        start += PAGE


def main():
    load_dotenv(ROOT / ".env")
    cfg = load_config()
    first, last = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2])
    e, site_min = cfg["email"], cfg["site"]["min_score"]
    render.DAYS.mkdir(parents=True, exist_ok=True)
    for day, papers in sorted(fetch(cfg["categories"], first, last).items()):
        scored = sorted(zip(papers, model.score(papers)), key=lambda t: -t[1]["score"])
        top = {p.id for p, s in scored[: e["max_papers"]] if s["score"] >= e["min_score"]}
        records = []
        for p, s in scored:  # same cut as `run`: the day's top papers plus anything above the site bar
            if p.id in top or s["score"] >= site_min:
                summ = summarize.summarize(p, cfg) or summarize.fallback_summary(p)
                records.append({**p.to_dict(), **s, "summary": summ, "emailed": p.id in top})
        print(f"{day}: {len(papers)} papers, {len(records)} kept; top: "
              + "; ".join(f"{100 * r['score']:.0f}% {r['title'][:50]}" for r in records[:3]), flush=True)
        (render.DAYS / f"{day}.json").write_text(json.dumps(
            {"date": day.isoformat(), "n_fetched": len(papers), "papers": records,
             "all_scores": [round(s["score"], 4) for _, s in scored]}, indent=2, ensure_ascii=False), encoding="utf-8")
    render.build_site(cfg)


if __name__ == "__main__":
    main()
