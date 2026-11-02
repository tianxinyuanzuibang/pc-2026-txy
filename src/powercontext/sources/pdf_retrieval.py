# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline page ranking for complete financial reports.

The ranker works on the pages that already live in a scoped Memory: every page
of the report is an entry whose text is ``DOCUMENT_ID\\nPAGE_TEXT``.  It scores
those pages with a small, deterministic lexical model and a set of structural
priors taken from the shape of a Form 10-K (financial statements, item
sections), then returns exact page citations.  No model, index or network
service is required, and nothing is invented: the return value only ever
references entries that the Memory already stores.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

WORD = re.compile(r"[a-z]+|\d+")
_ALNUM = re.compile(r"[^a-z0-9]+")

# Function words that carry no retrieval signal in a natural-language question.
_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those there their they we you i it its
    of to in on at by for with from as is are was were be been being do does did done has
    have had having what which who whom whose how why when where much many more most other
    others such any all both each few no nor not only own same so too very can could should
    would will shall may might must please based using use used show given according state
    response answer question following details detail shown rely relying assume perspective
    analyst report reported reports company companies
    """.split()
)

# Vocabulary that annual reports use for the ideas a question may phrase loosely.
# Keys are matched against the stem of a question term.
_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "geograph": ("region", "countri", "territori", "worldwid", "global"),
    "legal": ("litig", "lawsuit", "proceed", "claim", "regulatori", "court", "settlement"),
    "custom": ("client", "consumer", "retail", "distributor"),
    "employ": ("headcount", "workforc", "personnel"),
    "product": ("brand", "portfolio", "servic"),
    "competit": ("competitor", "rival", "marketshar"),
    "cyclical": ("season", "cycl", "econom"),
}

# Statement headings, in the wording different filers use.
_STATEMENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("income", re.compile(r"statements? of (income|operations|earnings)")),
    ("balance", re.compile(r"(balance sheets?|statements? of financial position)")),
    ("cash", re.compile(r"statements? of cash flows?")),
    ("equity", re.compile(r"statements? of (changes in )?(stockholders|shareholders)")),
    ("comprehensive", re.compile(r"statements? of comprehensive")),
)

_STATEMENT_HEADINGS = {name: pattern for name, pattern in _STATEMENT_PATTERNS}

# Form 10-K items and the words that introduce them.
_ITEMS: dict[str, str] = {
    "1": "business",
    "1a": "risk factors",
    "2": "properties",
    "3": "legal proceedings",
    "5": "market for",
    "7": "management",
    "7a": "quantitative and qualitative",
    "8": "financial statements",
    "9": "changes in",
}
_ITEM_HEADING = re.compile(r"(?i)\bitem\s+(\d{1,2})([ab])?\s*[\.\:]\s*([a-z][a-z ,'&\-]{2,60})")
_ITEM_REFERENCE = re.compile(r"(?i)(see\s+(also\s+)?|refer\s+to\s+|in\s+|under\s+|per\s+)$")

# Words that mark a question as being about a reported figure.
_METRIC_TERMS = frozenset(
    """
    margin ebitda capex revenue income profit profitability cost expense liability asset
    liquidity ratio rate tax cash debt capital growth percentage amount balance sheet
    flow operation investing financing dividend earning
    """.split()
)

# Ranking knobs.  They are constants rather than configuration because the
# acceptance run fixes the resource profile of the product.
_LEXICAL_WEIGHT = 1.0
_PASSAGE_WEIGHT = 0.6
_PHRASE_WEIGHT = 0.8
_STATEMENT_WEIGHT = 1.8
_ITEM_WEIGHT = 1.4
_HEADING_WEIGHT = 0.6
_ITEM_SPAN = 3

def stem(word: str) -> str:
    """Reduce one lower-case English word to a comparison key."""

    if len(word) <= 3 or word.isdigit():
        return word
    s = word
    if s.endswith("ies") and len(s) > 4:
        s = f"{s[:-3]}y"
    elif s.endswith(("ses", "xes", "zes", "ches", "shes")):
        s = s[:-2]
    elif s.endswith("s") and not s.endswith(("ss", "us", "is")):
        s = s[:-1]
    for suffix in ("ing", "ed"):
        if s.endswith(suffix) and len(s) - len(suffix) >= 4:
            base = s[: -len(suffix)]
            if len(base) >= 2 and base[-1] == base[-2] and base[-1] not in "lsz":
                base = base[:-1]
            s = base
            break
    for suffix, replacement in (
        ("ations", "ate"),
        ("ation", "ate"),
        ("itions", "ite"),
        ("ition", "ite"),
        ("ical", "ic"),
        ("ally", "al"),
        ("ities", "ity"),
        ("ness", ""),
        ("ment", ""),
        ("ity", ""),
        ("ive", ""),
        ("ous", ""),
        ("ful", ""),
        ("ly", ""),
    ):
        if s.endswith(suffix) and len(s) - len(suffix) >= 4:
            s = f"{s[: -len(suffix)]}{replacement}"
            break
    if s.endswith("e") and len(s) > 4:
        s = s[:-1]
    return s


def _key(word: str) -> str:
    if word.isdigit():
        return word
    if len(word) <= 3:
        return word
    reduced = stem(word)
    return reduced[:6] if len(reduced) > 6 else reduced


def content_terms(question: str) -> list[str]:
    """Comparison keys of the question, in order, without function words."""

    seen: set[str] = set()
    terms: list[str] = []
    for word in WORD.findall(question.lower()):
        if word in _STOPWORDS or (len(word) == 1 and not word.isdigit()):
            continue
        key = _key(word)
        if key in seen:
            continue
        seen.add(key)
        terms.append(key)
    return terms


def _phrase_pairs(question: str) -> list[tuple[str, str]]:
    """Adjacent content-word pairs, so a page that keeps the wording wins."""

    keys: list[str] = []
    for word in _ALNUM.split(question.lower()):
        if not word or (len(word) == 1 and not word.isdigit()):
            continue
        keys.append("" if word in _STOPWORDS else _key(word))
    return [(a, b) for a, b in zip(keys, keys[1:]) if a and b]


@dataclass(frozen=True)
class Page:
    """One stored page that can be cited."""

    document_id: str
    page_number: int
    citation: object
    text: str


class Corpus:
    """Lexical statistics for one set of stored pages."""

    def __init__(self, pages: list[Page]) -> None:
        self.pages = pages
        self.keys = [(page.document_id, page.page_number) for page in pages]
        self._tokens = [_page_tokens(page.text) for page in pages]
        self._counts = [Counter(tokens) for tokens in self._tokens]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average = (sum(self._lengths) / len(pages)) if pages else 1.0
        self._document_frequency: dict[str, int] = {}
        self._pair_sets = [set(zip(tokens, tokens[1:])) for tokens in self._tokens]

    # -- statistics -----------------------------------------------------
    def document_frequency(self, term: str) -> int:
        cached = self._document_frequency.get(term)
        if cached is None:
            cached = sum(1 for counts in self._counts if term in counts)
            self._document_frequency[term] = cached
        return cached

    def inverse_document_frequency(self, term: str) -> float:
        total = len(self.pages)
        frequency = self.document_frequency(term)
        return math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))

    def terms_in(self, index: int, terms: set[str]) -> set[str]:
        return terms.intersection(self._counts[index])

    # -- scoring --------------------------------------------------------
    def bm25(self, weights: dict[str, float], allowed: list[int]) -> list[float]:
        scores = [0.0] * len(self.pages)
        average = self._average or 1.0
        for term, weight in weights.items():
            if weight <= 0.0:
                continue
            idf = self.inverse_document_frequency(term)
            if idf <= 0.0:
                continue
            for index in allowed:
                frequency = self._counts[index].get(term, 0)
                if frequency:
                    norm = 1.0 - 0.75 + 0.75 * self._lengths[index] / average
                    scores[index] += weight * idf * (frequency * 2.2) / (frequency + 1.2 * norm)
        return scores

    def passage(self, terms: set[str], allowed: list[int], window: int = 90, step: int = 30) -> list[float]:
        """Best idf coverage of the question inside one short span of a page."""

        scores = [0.0] * len(self.pages)
        weights = {term: self.inverse_document_frequency(term) for term in terms}
        for index in allowed:
            tokens = self._tokens[index]
            if not tokens:
                continue
            positions: dict[str, list[int]] = {}
            for position, token in enumerate(tokens):
                if token in weights:
                    positions.setdefault(token, []).append(position)
            if not positions:
                continue
            best = 0.0
            last = max(1, len(tokens) - window + 1)
            for start in range(0, last, step):
                stop = start + window
                coverage = 0.0
                for term, found in positions.items():
                    for position in found:
                        if start <= position < stop:
                            coverage += weights[term]
                            break
                if coverage > best:
                    best = coverage
            scores[index] = best
        return scores

    def phrases(self, pairs: list[tuple[str, str]], allowed: list[int]) -> list[float]:
        scores = [0.0] * len(self.pages)
        for first, second in pairs:
            weight = self.inverse_document_frequency(first) + self.inverse_document_frequency(second)
            for index in allowed:
                if (first, second) in self._pair_sets[index]:
                    scores[index] += weight
        return scores

    def document_of(self, index: int) -> str:
        return self.keys[index][0]

    def documents(self) -> list[str]:
        seen: list[str] = []
        for document_id, _ in self.keys:
            if document_id not in seen:
                seen.append(document_id)
        return seen

    def text(self, index: int) -> str:
        return self.pages[index].text


def _page_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for word in WORD.findall(text.lower()):
        if word in _STOPWORDS:
            continue
        tokens.append(_key(word))
    return tokens


@dataclass
class _Signals:
    """Per-page evidence collected for one question."""

    scores: dict[int, float] = field(default_factory=dict)

    def add(self, index: int, value: float) -> None:
        self.scores[index] = self.scores.get(index, 0.0) + value


def _company_key(document_id: str) -> str:
    return _ALNUM.sub("", document_id.split("_")[0].lower())


def resolve_document(corpus: Corpus, question: str, lexical: list[float]) -> str | None:
    """Pick the report a question is about, by name first and by period second."""

    flat = _ALNUM.sub("", question.lower())
    documents = corpus.documents()
    if not documents:
        return None
    named = [document_id for document_id in documents if _names(flat, document_id)]
    return _best_period(named or documents, question, lexical, corpus)


def _names(flat_question: str, document_id: str) -> bool:
    key = _company_key(document_id)
    return len(key) >= 3 and key in flat_question


_YEAR = re.compile(r"(?:19|20)\d{2}")


def _best_period(documents: list[str], question: str, lexical: list[float], corpus: Corpus) -> str:
    """Break a tie between several reports of the same company by period."""

    years = set(_YEAR.findall(question))
    best = documents[0]
    best_value = (-1, -1.0)
    for document_id in documents:
        shared = len(years.intersection(_YEAR.findall(document_id)))
        tops = sorted(
            (score for index, score in enumerate(lexical) if corpus.document_of(index) == document_id),
            reverse=True,
        )[:10]
        value = (shared, sum(tops))
        if value > best_value:
            best_value = value
            best = document_id
    return best


def _expand(terms: list[str]) -> dict[str, float]:
    weights = {term: 1.0 for term in terms}
    for term in terms:
        for prefix, related in _EXPANSIONS.items():
            if term.startswith(prefix) or prefix.startswith(term[:6]):
                for word in related:
                    weights.setdefault(_key(word), 0.45)
    return weights


def _normalize(values: list[float]) -> list[float]:
    top = max(values) if values else 0.0
    if top <= 0.0:
        return [0.0] * len(values)
    return [value / top for value in values]


def _numeric_tokens(text: str) -> int:
    return sum(1 for word in WORD.findall(text.lower()) if word.isdigit())


def statement_pages(corpus: Corpus, document_id: str) -> dict[str, int]:
    """First page of each primary financial statement of one report."""

    pages: dict[str, int] = {}
    indexes = sorted(
        (index for index in range(len(corpus.pages)) if corpus.document_of(index) == document_id),
        key=lambda index: corpus.keys[index][1],
    )
    last_page = int(len(indexes) * 0.7)
    for index in indexes:
        if corpus.keys[index][1] > last_page:
            break
        flat = " ".join(corpus.text(index).split()).lower()
        if flat.startswith("table of contents"):
            flat = flat[len("table of contents") :]
        head = flat[:200]
        if head.startswith("index to"):
            continue
        if _numeric_tokens(flat) < 40:
            continue
        found = [name for name, pattern in _STATEMENT_PATTERNS if pattern.search(flat)]
        if len(found) >= 2:
            continue  # a listing page rather than the statement itself
        for name in found:
            if name in pages:
                continue
            if _STATEMENT_HEADINGS[name].search(head):
                pages[name] = corpus.keys[index][1]
    return pages


def item_pages(corpus: Corpus, document_id: str) -> dict[str, int]:
    """First page of each Form 10-K item of one report."""

    pages: dict[str, int] = {}
    indexes = sorted(
        (index for index in range(len(corpus.pages)) if corpus.document_of(index) == document_id),
        key=lambda index: corpus.keys[index][1],
    )
    for index in indexes:
        flat = " ".join(corpus.text(index).split())
        flat = re.sub(r"(?i)^table of contents\s*", "", flat)
        hits: list[str] = []
        for match in _ITEM_HEADING.finditer(flat):
            key = f"{match.group(1)}{match.group(2) or ''}"
            body = match.group(3).lower().strip()
            expected = _ITEMS.get(key)
            if not expected:
                continue
            if not any(word in body for word in expected.split()[:2]):
                continue
            before = flat[max(0, match.start() - 25) : match.start()].lower().strip()
            if _ITEM_REFERENCE.search(before):
                continue
            hits.append(key)
        if len(hits) >= 4:
            continue  # the table of contents, not the item itself
        for key in hits:
            pages.setdefault(key, corpus.keys[index][1])
    return pages


def _statement_order(terms: set[str]) -> list[str]:
    if terms.intersection({"liability", "asset", "liquid", "ratio", "balance", "receivable", "inventori"}):
        return ["balance", "income", "cash", "comprehensive", "equity"]
    if terms.intersection({"cash", "capex", "flow", "invest", "financ", "dividend", "depreci", "operat"}):
        return ["cash", "income", "balance", "comprehensive", "equity"]
    return ["income", "cash", "balance", "comprehensive", "equity"]


def _is_metric_question(terms: set[str]) -> bool:
    return bool(terms.intersection(_METRIC_TERMS))


def heading_bonus(corpus: Corpus, allowed: list[int], terms: set[str]) -> list[float]:
    """Reward pages whose opening heading names what the question is about."""

    scores = [0.0] * len(corpus.pages)
    strong = {term for term in terms if corpus.inverse_document_frequency(term) >= 2.5}
    if not strong:
        return scores
    for index in allowed:
        opening = corpus.text(index).split()[:16]
        head = {_key(word) for word in WORD.findall(" ".join(opening).lower())}
        shared = strong.intersection(head)
        if shared:
            scores[index] = sum(corpus.inverse_document_frequency(term) for term in shared)
    return scores


def rank(question: str, corpus: Corpus, limit: int) -> list[Page]:
    """Return the pages of a report that best support one question."""

    if not corpus.pages or limit < 1:
        return []
    terms = content_terms(question)
    if not terms:
        return corpus.pages[:limit]
    term_set = set(terms)

    lexical = corpus.bm25(_expand(terms), list(range(len(corpus.pages))))
    document_id = resolve_document(corpus, question, lexical)
    if document_id is None:
        return []
    allowed = [index for index in range(len(corpus.pages)) if corpus.document_of(index) == document_id]

    weighted = corpus.bm25(_expand(terms), allowed)
    passages = corpus.passage(term_set, allowed)
    phrases = corpus.phrases(_phrase_pairs(question), allowed)
    headings = heading_bonus(corpus, allowed, term_set)
    signals = _Signals()
    lex_norm = _normalize(weighted)
    pass_norm = _normalize(passages)
    phrase_norm = _normalize(phrases)
    head_norm = _normalize(headings)
    for index in allowed:
        signals.add(
            index,
            _LEXICAL_WEIGHT * lex_norm[index]
            + _PASSAGE_WEIGHT * pass_norm[index]
            + _PHRASE_WEIGHT * phrase_norm[index]
            + _HEADING_WEIGHT * head_norm[index],
        )

    # Structural priors: a reported figure lives in the statements of the report,
    # a description of the business lives in the item that introduces it.
    page_numbers = {corpus.keys[index][1]: index for index in allowed}
    statements = statement_pages(corpus, document_id)
    items = item_pages(corpus, document_id)

    def boost(page_number: int | None, weight: float) -> None:
        if page_number is not None and page_number in page_numbers:
            signals.add(page_numbers[page_number], weight)

    if _is_metric_question(term_set):
        for position, name in enumerate(_statement_order(term_set)):
            if position >= 4:
                break
            boost(statements.get(name), _STATEMENT_WEIGHT - 0.2 * position)
    starts: list[int] = []
    if term_set.intersection({"legal", "litig", "battl", "claim"}):
        starts.append(items.get("3"))
    starts.extend([items.get("1"), items.get("7")])
    added = 0
    for start in [page for page in starts if page]:
        for offset in range(_ITEM_SPAN):
            if added >= 2 * _ITEM_SPAN:
                break
            boost(start + offset, _ITEM_WEIGHT - 0.1 * added)
            added += 1

    ordered = sorted(signals.scores, key=lambda index: (-signals.scores[index], corpus.keys[index]))
    return [corpus.pages[index] for index in ordered[:limit]]


__all__ = ["Corpus", "Page", "content_terms", "rank", "stem", "resolve_document"]
