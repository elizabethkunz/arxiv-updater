# arXiv digest

Daily email of up to 10 new arXiv papers ranked by % match with `Interesting papers - Sheet1.csv`,
plus a static site (`site/`) listing everything at 80%+ with a fuller AI summary.

## How it works

1. **Fetch** newly announced papers in the categories in `config.yaml` via arXiv's OAI-PMH
   endpoint (the search API rate-limits too aggressively). Slow: ~1 min per category.
2. **Score** each title+abstract: `bge-base` embeddings → class-balanced logistic regression
   trained on your liked papers vs. 2,500 random background papers.
   The % is "how much does this look like the liked list rather than generic quant-ph".
3. **Summarize** the top papers with Claude Haiku 4.5 from arXiv's full-text HTML (abstract as fallback).
4. **Email** the top ≤10 above 50%; **site** gets everything ≥80%.

## Commands

```bash
uv run python -m digest.cli train                         # retrain after editing the CSV
uv run python -m digest.cli run --date 2026-09-15 --dry-run   # replay a day, no email
uv run python -m digest.cli run                           # the real daily run (sends email)
uv run python -m digest.cli resummarize 2026-09-15        # fill in missing AI summaries, no refetch
uv run python -m digest.cli site                          # rebuild site/ only
```

## Teaching it

- Add rows to `Interesting papers - Sheet1.csv` (Title, Authors, Abstract, Link), then `train`.
- Optional: create `disliked.csv` with the same columns for papers it keeps surfacing that you
  don't want. They count 5× as negatives.
- Commit `state/model.joblib` after retraining so the GitHub Action uses the new model.

Without an API key the run still works; summaries fall back to the first two abstract sentences.
