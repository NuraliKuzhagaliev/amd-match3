"""Hybrid BM25 / dense retrieval with identifier links across documents."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import math
import re
import unicodedata


@dataclass
class Chunk:
    id: str
    path: str
    locator: str
    text: str


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    found = re.findall(r"[^\W_]+(?:[-_.][^\W_]+)*", text, re.UNICODE)
    result = []
    for value in found:
        result.append(value)
        if re.search(r"[-_.]", value):
            result.extend(re.split(r"[-_.]", value))
            result.append(re.sub(r"[-_.]", "", value))
    return result


def identifiers(text: str) -> set[str]:
    return {re.sub(r"[-_.]", "", value.upper()) for value in re.findall(
        r"\b[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+\b|\b[A-Z]{1,6}\d{3,}\b", text)}


def required_identifiers(query: str) -> set[str]:
    strict = {entity for entity in identifiers(query) if any(c.isdigit() for c in entity)}
    strict.update(re.sub(r'[-_.]', '', raw.upper()) for raw in re.findall(r'\b[A-Za-z]\w*_\w+\b', query))
    return strict


def chunk_blocks(path: str, blocks: list[dict], size: int = 2400, overlap: int = 350) -> list[Chunk]:
    result = []
    buffer, locators = "", []

    def emit() -> None:
        text = buffer.strip()
        if not text:
            return
        identity = hashlib.sha256((path + "\0" + text).encode()).hexdigest()[:16]
        locator = "; ".join(dict.fromkeys(locators))[:300]
        if not result or result[-1].id != identity:
            result.append(Chunk(identity, path, locator, text))

    for block in blocks:
        text = block["text"].strip()
        if not text:
            continue
        for line in text.splitlines():
            while len(line) > size:
                if buffer:
                    emit()
                    buffer, locators = "", []
                buffer, locators = line[:size], [block["locator"]]
                emit()
                buffer, locators = "", []
                line = line[max(1, size - overlap):]
            if len(buffer) + len(line) + 1 > size:
                emit()
                buffer = buffer[-overlap:] if overlap else ""
                locators = locators[-1:]
            buffer += "\n" + line
            locators.append(block["locator"])
    emit()
    return result


class SearchIndex:
    def __init__(self, chunks: list[Chunk], vectors=None):
        self.chunks = chunks
        self.vectors = vectors
        self.counts = [Counter(tokens(c.path + " " + c.text)) for c in chunks]
        self.lengths = [sum(x.values()) for x in self.counts]
        self.average = sum(self.lengths) / max(1, len(chunks))
        self.df = Counter(k for row in self.counts for k in row)
        self.entities = [identifiers(c.text) for c in chunks]
        self.links = defaultdict(list)
        for i, ids in enumerate(self.entities):
            for value in ids:
                self.links[value].append(i)

    def missing_identifiers(self, query: str) -> set[str]:
        from .evidence import value_in_text
        # An exact product suffix or signal name cannot be silently replaced by
        # a nearby identifier. Permit harmless spacing/punctuation variants.
        missing = set()
        for entity in required_identifiers(query):
            if entity not in self.links and not any(value_in_text(entity, c.path + '\n' + c.text) for c in self.chunks):
                missing.add(entity)
        return missing

    def lexical(self, query: str) -> list[int]:
        query_tokens = set(tokens(query))
        scores = []
        total = len(self.chunks)
        for i, row in enumerate(self.counts):
            score = 0.0
            for term in query_tokens:
                frequency = row.get(term, 0)
                if frequency:
                    inverse = math.log(1 + (total - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    denominator = frequency + 1.5 * (0.25 + 0.75 * self.lengths[i] / max(1, self.average))
                    score += inverse * frequency * 2.5 / denominator
            if score > 0:
                scores.append((score, i))
        return [i for _, i in sorted(scores, reverse=True)]

    def retrieve(self, query: str, vector=None, limit: int = 24) -> list[Chunk]:
        if not self.chunks:
            return []
        rankings = [self.lexical(query)[:60]]
        if self.vectors is not None and vector is not None:
            import numpy as np
            similarities = self.vectors @ np.asarray(vector, dtype=np.float32)
            rankings.append(np.argsort(-similarities).tolist()[:60])
        fused = defaultdict(float)
        for ranking in rankings:
            for rank, i in enumerate(ranking):
                fused[i] += 1 / (40 + rank)
        query_ids = identifiers(query)
        historical = bool(re.search(r"\b(previous|old|withdrawn|superseded|historical|revision\s+1)\b", query, re.I))
        for i in list(fused):
            if query_ids & self.entities[i]:
                fused[i] += 0.08 * len(query_ids & self.entities[i])
            if not historical and re.search(r"withdrawn|superseded|obsolete", self.chunks[i].path, re.I):
                fused[i] *= 0.7
        anchors = [i for i in fused if query_ids & self.entities[i]]
        seeds = sorted(anchors or list(fused), key=fused.get, reverse=True)[:12 if anchors else 6]
        # Follow shared identifiers for two hops. Generic product IDs linking most
        # of the corpus are downweighted; distinctive ticket/error IDs dominate.
        frontier = set(seeds)
        visited = set(seeds)
        for hop in range(2):
            following = set()
            for seed in frontier:
                for entity in self.entities[seed]:
                    # Generic column/signal names link unrelated products. Only
                    # distinctive numbered bridge IDs expand the graph; query
                    # IDs already anchored retrieval and must not fan out again.
                    if entity in query_ids or not any(c.isdigit() for c in entity):
                        continue
                    neighbours = self.links[entity]
                    if len(neighbours) > max(8, len(self.chunks) // 4):
                        continue
                    for other in neighbours:
                        if other not in visited:
                            fused[other] = max(fused[other], max(0.045, fused[seed] * 0.9))
                            following.add(other)
            visited.update(following)
            frontier = following
        order = sorted(fused, key=fused.get, reverse=True)
        selected, per_file = [], Counter()
        # Initial diversity keeps a long file from burying a bridge document.
        for quota in (2, 6):
            for i in order:
                if i not in selected and per_file[self.chunks[i].path] < quota:
                    selected.append(i)
                    per_file[self.chunks[i].path] += 1
                    if len(selected) == limit:
                        return [self.chunks[j] for j in selected]
        return [self.chunks[i] for i in selected]
