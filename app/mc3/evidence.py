"""Strict response parsing and auditable evidence validation."""
from __future__ import annotations

import json
import math
import re
import unicodedata

REFUSAL = {"answer": "", "citations": [], "confidence": 0.0}


def compact(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())


def whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def value_in_text(answer: str, text: str) -> bool:
    """Match complete values, never the 94 inside 194 or C2 inside C21.

    Punctuation variation and multiword qualifiers remain accepted. This is
    literal grounding, not a proof that the value answers the question.
    """
    target = compact(answer)
    if not target:
        return False
    normalized = unicodedata.normalize('NFKC', text)
    if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', answer):
        return bool(re.search(r'(?<![\w.+-])' + re.escape(answer) + r'(?![\w.]|[,]\d)', normalized))
    words = re.findall(r'[^\W_]+(?:[-_.·][^\W_]+)*', normalized, re.UNICODE)
    for start in range(len(words)):
        value = ''
        for word in words[start:start+12]:
            value += compact(word)
            if value == target:
                return True
            if len(value) >= len(target):
                break
    return False


def value_only(answer: str, query: str) -> str:
    # Do not strip identifying prefixes/suffixes from revisions or part numbers.
    if re.search(r"temperature|junction|degrees|celsius|fahrenheit", query, re.I):
        match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*(?:°\s*)?(?:[CF]|degrees(?:\s+(?:celsius|fahrenheit))?)", answer, re.I)
        if match:
            return match[1]
    # Normalize scalar quantities only for the requested physical attribute.
    # Never truncate an arbitrary numeric-looking identifier or change units:
    # the numeric value stays exactly as extracted, including sign/decimals.
    quantities = (
        (r'voltage|supply|\bvolts?\b', r'(?:[mk]?V|volts?|millivolts?|kilovolts?)'),
        (r'\bpower\b|\bwatts?\b', r'(?:[mk]?W|watts?|milliwatts?|kilowatts?)'),
        (r'\blength\b|\bdistance\b|\bmet(?:re|er)s?\b', r'(?:[mck]?m|metres?|meters?|millimetres?|millimeters?)'),
        (r'\bseconds\b|\btimeout\b|\bduration\b|\blatency\b', r'(?:s|secs?|seconds?|ms|milliseconds?)'),
        (r'\blead\s*time\b|\bweeks?\b', r'(?:weeks?|wks?)'),
    )
    for attribute, unit in quantities:
        if re.search(attribute, query, re.I):
            match = re.fullmatch(r'([+-]?\d+(?:\.\d+)?)\s*' + unit, answer, re.I)
            if match:
                return match[1]
    return answer


def validate_selection(value: dict, sources: dict, query: str) -> tuple[dict, dict]:
    """Validate source IDs; construct evidence excerpts from originals in code.

    Citation necessity and semantic interpretation remain model judgments.
    Validation checks existence and value grounding, without asking the model
    to reproduce error-prone exact quotes in its output.
    """
    answer = value.get("answer")
    if not isinstance(answer, str):
        raise ValueError("answer must be a string")
    answer = value_only(answer.strip(), query)
    if not answer:
        return dict(REFUSAL), {"reason": "model_abstained"}
    if len(answer) > 500 or "\n" in answer:
        return dict(REFUSAL), {"reason": "answer_is_not_a_short_value"}
    chosen = value.get("sources")
    if not isinstance(chosen, list) or not chosen:
        return dict(REFUSAL), {"reason": "missing_sources"}
    if not all(isinstance(label, str) and label in sources for label in chosen):
        return dict(REFUSAL), {"reason": "unknown_source"}
    chosen = list(dict.fromkeys(chosen))
    from .retrieval import required_identifiers, identifiers
    required = required_identifiers(query)
    selected_text = '\n'.join(sources[label].path + '\n' + sources[label].text for label in chosen)
    missing = [entity for entity in required if not value_in_text(entity, selected_text)]
    if missing:
        return dict(REFUSAL), {'reason': 'question_identifier_not_in_sources', 'missing': sorted(missing)}
    if not any(value_in_text(answer, sources[label].text) for label in chosen):
        return dict(REFUSAL), {"reason": "answer_not_in_sources"}
    # A citation cannot vanish if it is the only bridge between the question
    # entity and the answer. Check file-level connectivity using distinctive IDs.
    # This is structural evidence, not proof of the edge's semantic meaning.
    files = {}
    for label in chosen:
        source = sources[label]
        files.setdefault(source.path, []).append(source.text)
    bodies = {path: '\n'.join(texts) for path, texts in files.items()}
    values = {path for path, text in bodies.items() if value_in_text(answer, text)}
    entities = {path: {x for x in identifiers(text) if any(c.isdigit() for c in x)}
                for path, text in bodies.items()}
    if required:
        connected = False
        for value_path in values:
            reached = {value_path}
            while True:
                following = {path for path in bodies if path not in reached
                             and any(entities[path] & entities[other] for other in reached)}
                if not following:
                    break
                reached.update(following)
            context = '\n'.join(path + '\n' + bodies[path] for path in reached)
            if all(value_in_text(entity, context) for entity in required):
                connected = True
                break
        if not connected:
            return dict(REFUSAL), {'reason': 'disconnected_identifier_chain'}
    confidence = value.get("confidence", 0.8)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
        confidence = 0.8
    evidence = [{"source": label, "path": sources[label].path, "locator": sources[label].locator,
                 "original_text": sources[label].text} for label in chosen]
    return {"answer": answer, "citations": sorted({sources[label].path for label in chosen}),
            "confidence": min(1.0, max(0.0, confidence))}, {"reason": "validated", "evidence": evidence}


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict) and "answer" in value:
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError("Model did not produce an answer JSON object")


def validate(value: dict, sources: dict) -> tuple[dict, dict]:
    """sources map presented chunk labels to Chunk instances.

    Evidence quotes must actually occur in the selected chunks. This detects
    fabricated quotes and source IDs; it is not a mathematical proof of entailment.
    Source necessity is decided by the model and evaluated independently in tests.
    """
    answer = value.get("answer")
    if not isinstance(answer, str):
        raise ValueError("answer must be a string")
    answer = answer.strip()
    if not answer:
        return dict(REFUSAL), {"reason": "model_abstained"}
    if len(answer) > 500 or "\n" in answer:
        return dict(REFUSAL), {"reason": "answer_is_not_a_short_value"}
    evidence = value.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return dict(REFUSAL), {"reason": "missing_evidence"}
    paths, quotes = set(), []
    for item in evidence:
        if not isinstance(item, dict):
            return dict(REFUSAL), {"reason": "invalid_evidence"}
        source, quote = sources.get(item.get("source")), item.get("quote")
        if source is None or not isinstance(quote, str) or len(quote.strip()) < 2:
            return dict(REFUSAL), {"reason": "unknown_source_or_empty_quote"}
        if whitespace(quote) not in whitespace(source.text):
            return dict(REFUSAL), {"reason": "quote_not_in_source", "source": item.get("source")}
        paths.add(source.path)
        quotes.append(quote)
    # Challenge answers are extractive values, including multipart version IDs.
    # A model may not introduce a number absent from all its supporting quotes.
    if compact(answer) not in compact(" ".join(quotes)):
        return dict(REFUSAL), {"reason": "answer_not_in_evidence"}
    confidence = value.get("confidence", 0.8)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
        confidence = 0.8
    result = {"answer": answer, "citations": sorted(paths), "confidence": min(1.0, max(0.0, confidence))}
    return result, {"evidence": evidence, "reason": "validated"}
