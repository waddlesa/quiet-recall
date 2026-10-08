from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata


_ASCII_WORD = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")


def lexical_tokens(text: str) -> list[str]:
    """Tokenize Chinese without a dictionary dependency.

    Character bi/tri-grams preserve distinctive short cues such as 自画像、网页
    and 人机恋; ASCII terms such as GPT and NEARFIELD remain whole tokens.
    """

    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = _ASCII_WORD.findall(normalized)
    for run in _CJK_RUN.findall(normalized):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
        if len(run) >= 3:
            tokens.extend(run[index : index + 3] for index in range(len(run) - 2))
    return tokens


def bm25_scores(
    query: str,
    documents: list[str],
    *,
    k1: float = 1.2,
    b: float = 0.75,
    max_query_terms: int = 12,
) -> list[float]:
    """Return standard positive-IDF BM25 scores for an in-memory corpus."""

    tokenized = [lexical_tokens(document) for document in documents]
    query_terms = list(dict.fromkeys(lexical_tokens(query)))
    if not documents or not query_terms:
        return [0.0 for _ in documents]
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / len(lengths) if lengths else 1.0
    frequencies = [Counter(tokens) for tokens in tokenized]
    document_frequency = {
        term: sum(term in frequency for frequency in frequencies) for term in query_terms
    }
    total = len(documents)
    # Conversational queries contain many glue words.  Keep the highest-IDF
    # terms that actually occur in the corpus rather than letting a long casual
    # sentence win by accumulating weak overlaps.
    query_terms = sorted(
        (term for term in query_terms if document_frequency[term] > 0),
        key=lambda term: (
            math.log(
                1.0
                + (total - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            ),
            len(term),
            term,
        ),
        reverse=True,
    )[:max_query_terms]
    scores: list[float] = []
    for length, frequency in zip(lengths, frequencies):
        score = 0.0
        normalization = k1 * (1.0 - b + b * length / max(average_length, 1.0))
        for term in query_terms:
            count = frequency.get(term, 0)
            if not count:
                continue
            df = document_frequency[term]
            idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            score += idf * (count * (k1 + 1.0)) / (count + normalization)
        scores.append(score)
    return scores


def reciprocal_rank(rank: int, *, k: int = 60) -> float:
    return 1.0 / (k + rank)


def matched_cues(query: str, document: str, *, limit: int = 3) -> tuple[str, ...]:
    """Return compact query terms actually present in a candidate document."""

    document_terms = set(lexical_tokens(document))
    matches = {
        term
        for term in lexical_tokens(query)
        if term in document_terms and (len(term) >= 2 or term.isascii())
    }
    ordered = sorted(matches, key=lambda term: (len(term), term.isascii(), term), reverse=True)
    selected: list[str] = []
    for term in ordered:
        if any(term in existing for existing in selected):
            continue
        selected.append(term)
        if len(selected) >= limit:
            break
    return tuple(selected)
