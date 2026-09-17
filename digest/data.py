import csv
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"

ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", re.I)


@dataclass
class Paper:
    id: str | None          # arXiv ID without version, None for non-arXiv rows
    title: str
    authors: list[str]
    abstract: str
    categories: list[str] = field(default_factory=list)
    published: str = ""

    @property
    def text(self) -> str:
        return f"{self.title}. {self.abstract}"

    def to_dict(self) -> dict:
        return asdict(self)


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def load_csv(path: Path) -> list[Paper]:
    papers = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            title = clean(row.get("Title"))
            if not title:
                continue
            m = ARXIV_ID_RE.search(row.get("Link") or "")
            papers.append(Paper(
                id=m.group(1) if m else None,
                title=title,
                authors=[clean(a) for a in (row.get("Authors") or "").split(",") if clean(a)],
                abstract=clean(row.get("Abstract")),
            ))
    return papers
