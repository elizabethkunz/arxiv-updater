"""Author features appended to the text embedding.

Each author who appears on >= 2 training papers becomes a feature; rows are scaled by
1/sqrt(n_authors) so a 200-author collaboration paper doesn't swamp a 4-author one. One extra
feature marks papers with an author from `authors.established` in config.yaml, so groups with
few or no liked papers still count.

Names: old papers list "F. Yan", new ones "Fei Yan". A full-name author also matches the
initial-form key ("f yan") only when the training data never saw a *different* full name under
that key — so "Fei Yan" matches "F. Yan" but "Kai Li" and "Ke Li" don't match each other.
"""
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp

MIN_PAPERS = 2


def _tokens(name: str) -> list[str]:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.findall(r"[a-z]+", s.replace("-", ""))


def keys(name: str) -> tuple[str, str | None]:
    """(initial key, full key or None when only initials are given)."""
    t = _tokens(name)
    if len(t) < 2:
        return (t[0] if t else "", None)
    return f"{t[0][0]} {t[-1]}", (f"{t[0]} {t[-1]}" if len(t[0]) > 1 else None)


class AuthorFeatures:
    def __init__(self, established: list[str] | None = None):
        self.established = [keys(n) for n in (established or [])]
        self.vocab: dict[str, int] = {}
        self.canon: dict[str, str | None] = {}

    def _emit(self, authors: list[str]) -> set[str]:
        out = set()
        for a in authors:
            ini, full = keys(a)
            if not ini:
                continue
            if full:
                out.add(full)
                if self.canon.get(ini, full) in (None, full):
                    out.add(ini)
            else:
                out.add(ini)
                if self.canon.get(ini):
                    out.add(self.canon[ini])
        return out

    def _is_established(self, authors: list[str]) -> bool:
        for a in authors:
            ini, full = keys(a)
            for e_ini, e_full in self.established:
                if ini == e_ini and (full is None or e_full is None or full == e_full):
                    return True
        return False

    def fit(self, author_lists: list[list[str]]) -> "AuthorFeatures":
        fulls = defaultdict(set)
        for authors in author_lists:
            for a in authors:
                ini, full = keys(a)
                if ini and full:
                    fulls[ini].add(full)
        # initial key -> its one full name, None if only initials were seen; ambiguous keys get ""
        self.canon = {i: (next(iter(f)) if len(f) == 1 else "") for i, f in fulls.items()}
        cnt = Counter(k for authors in author_lists for k in self._emit(authors))
        self.vocab = {k: j for j, k in enumerate(sorted(k for k, c in cnt.items() if c >= MIN_PAPERS))}
        return self

    def transform(self, author_lists: list[list[str]]) -> sp.csr_matrix:
        M = sp.lil_matrix((len(author_lists), len(self.vocab) + 1))
        for i, authors in enumerate(author_lists):
            cols = [self.vocab[k] for k in self._emit(authors) if k in self.vocab]
            if cols:
                M[i, cols] = 1.0 / np.sqrt(max(len(authors), 1))
            if self._is_established(authors):
                M[i, len(self.vocab)] = 1.0
        return M.tocsr()
