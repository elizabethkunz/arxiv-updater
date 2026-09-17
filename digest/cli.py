import argparse
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
from dotenv import load_dotenv

from . import arxiv, mailer, model, render, summarize
from .data import ROOT, STATE, load_config, load_csv

SEEN = STATE / "seen.json"


def cmd_train(args, cfg):
    liked = load_csv(ROOT / cfg["liked_csv"])
    dis_path = ROOT / cfg["disliked_csv"]
    disliked = load_csv(dis_path) if dis_path.exists() else []
    exclude = {p.id for p in liked + disliked if p.id}
    print(f"{len(liked)} liked, {len(disliked)} disliked; loading background ...")
    background = arxiv.load_background(cfg, exclude)
    report = model.train(liked, background, disliked, args.embedding_model or cfg["embedding_model"])
    p, y = report.pop("_cv_scores")
    print(json.dumps(report, indent=2))
    print("\nLowest-scoring liked papers (held out):")
    for i in np.argsort(p[: len(liked)])[:8]:
        print(f"  {100 * p[i]:5.1f}%  {liked[i].title[:90]}")
    print("\nHighest-scoring background papers (held out) — should look like things you'd like:")
    bg_p = p[len(liked): len(liked) + len(background)]
    for i in np.argsort(-bg_p)[:12]:
        print(f"  {100 * bg_p[i]:5.1f}%  {background[i].title[:90]}")


def cmd_run(args, cfg):
    seen = set(json.loads(SEEN.read_text())) if SEEN.exists() else set()
    if args.date:
        # Replay a single announcement day; ignores and does not update seen.json.
        day = date.fromisoformat(args.date)
        papers = arxiv.fetch(cfg["categories"], day, day)
    else:
        # Scan a short lookback window and rely on seen.json, so a missed or failed run
        # (weekend, outage) is caught up automatically without repeats.
        day = datetime.now(timezone.utc).date()
        papers = arxiv.fetch(cfg["categories"], day - timedelta(days=cfg["run"]["lookback_days"]), day)
        papers = [p for p in papers if p.id not in seen]
    print(f"{day}: {len(papers)} new papers")

    scored = sorted(zip(papers, model.score(papers)), key=lambda t: -t[1]["score"]) if papers else []
    e, site_min = cfg["email"], cfg["site"]["min_score"]
    emailed = {p.id for p, s in scored[: e["max_papers"]] if s["score"] >= e["min_score"]}

    records = []
    for p, s in scored:
        if p.id not in emailed and s["score"] < site_min:
            continue
        summ = summarize.summarize(p, cfg) or summarize.fallback_summary(p)
        records.append({**p.to_dict(), **s, "summary": summ, "emailed": p.id in emailed})
        print(f"  {100 * s['score']:5.1f}%  {p.title[:90]}")

    print("  --- next 20 below the cut (check for misses; add good ones to the CSV) ---")
    for p, s in scored[len(records): len(records) + 20]:
        print(f"  {100 * s['score']:5.1f}%  {p.id}  {p.title[:90]}")

    day_record = {"date": day.isoformat(), "n_fetched": len(papers), "papers": records,
                  "all_scores": [round(s["score"], 4) for _, s in scored]}
    render.DAYS.mkdir(parents=True, exist_ok=True)
    path = render.DAYS / f"{day}.json"
    stored = day_record
    if not args.date and path.exists():
        # A second run on the same day only sees what seen.json let through (usually nothing);
        # add that to the stored day instead of replacing it. The email still covers only the new papers.
        old = json.loads(path.read_text(encoding="utf-8"))
        new_ids = {r["id"] for r in records}
        stored = {"date": day.isoformat(), "n_fetched": old["n_fetched"] + len(papers),
                  "papers": sorted([r for r in old["papers"] if r["id"] not in new_ids] + records,
                                   key=lambda r: -r["score"]),
                  "all_scores": sorted(old.get("all_scores", []) + day_record["all_scores"], reverse=True)}
    path.write_text(json.dumps(stored, indent=2, ensure_ascii=False), encoding="utf-8")
    render.build_site(cfg)

    subject, text, html = render.render_email(day_record, cfg)
    if args.dry_run:
        out = ROOT / "out"
        out.mkdir(exist_ok=True)
        (out / f"email_{day}.html").write_text(html, encoding="utf-8")
        (out / f"email_{day}.txt").write_text(f"Subject: {subject}\n\n{text}", encoding="utf-8")
        print(f"dry run: wrote out/email_{day}.html/.txt and site/")
    else:
        if records:
            print(f"email sent to {mailer.send(subject, text, html)} recipient(s)")
        if not args.date:
            SEEN.write_text(json.dumps(sorted(seen | {p.id for p in papers})))


def cmd_resummarize(args, cfg):
    """Refill summaries for a stored day (e.g. after adding an API key) and re-render, no refetch."""
    from .data import Paper
    path = render.DAYS / f"{args.date}.json"
    day = json.loads(path.read_text(encoding="utf-8"))
    fields = ("id", "title", "authors", "abstract", "categories", "published")
    for rec in day["papers"]:
        if rec["summary"]["generated_by"].startswith("none"):
            p = Paper(**{k: rec[k] for k in fields})
            rec["summary"] = summarize.summarize(p, cfg) or rec["summary"]
    path.write_text(json.dumps(day, indent=2, ensure_ascii=False), encoding="utf-8")
    render.build_site(cfg)
    subject, text, html = render.render_email(day, cfg)
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    (out / f"email_{args.date}.html").write_text(html, encoding="utf-8")
    (out / f"email_{args.date}.txt").write_text(f"Subject: {subject}\n\n{text}", encoding="utf-8")
    print(f"re-rendered site/ and out/email_{args.date}.*")


def cmd_send(args, cfg):
    """(Re)send the email for a stored day as-is: no refetch, no rescoring."""
    day = json.loads((render.DAYS / f"{args.date}.json").read_text(encoding="utf-8"))
    if args.style:  # trying out a look: override the configured style and say so in the subject
        cfg["email"]["style"] = args.style
    subject, text, html = render.render_email(day, cfg)
    n = mailer.send(subject + (f" [{args.style}]" if args.style else ""), text, html)
    print(f"email sent to {n} recipient(s)")


def main():
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(prog="digest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="(re)train the classifier from the liked-papers CSV")
    t.add_argument("--embedding-model")
    r = sub.add_parser("run", help="fetch, score, summarize, build site, send email")
    r.add_argument("--date", help="replay one submission day (YYYY-MM-DD)")
    r.add_argument("--dry-run", action="store_true", help="write the email to out/ instead of sending")
    sub.add_parser("site", help="rebuild site/ from state/days")
    s = sub.add_parser("resummarize", help="fill in missing AI summaries for a stored day and re-render")
    s.add_argument("date")
    m = sub.add_parser("send", help="(re)send the email for a stored day without refetching")
    m.add_argument("date")
    m.add_argument("--style", choices=["styled", "plain"], help="override email.style for this send")
    args = ap.parse_args()
    cfg = load_config()
    {"train": cmd_train, "run": cmd_run, "resummarize": cmd_resummarize, "send": cmd_send,
     "site": lambda a, c: render.build_site(c)}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
