"""Deterministic, provenance-first investigation answer engine.

No generative text model is used. Retrieved source observations are converted into
atomic, typed claims. Every factual claim must cite one or more source chunks.
Cross-exhibit matches are leads unless their matching rule is independently
sufficient; co-occurrence and lexical overlap are never promoted to proof.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from itertools import combinations
from typing import Any

CLASSIFICATIONS = {
    "SOURCE_TEXT", "OCR_OBSERVATION", "VISION_OBSERVATION",
    "TRANSCRIPTION", "INFERENCE", "VERIFIED_RELATIONSHIP",
}
_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "was", "were",
    "are", "has", "have", "had", "into", "onto", "then", "than", "there",
    "their", "they", "them", "what", "when", "where", "which", "who",
    "show", "find", "tell", "about", "does", "did", "can", "could",
    "evidence", "exhibit", "source", "record", "image", "video", "text",
}
_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9:_/-]*[A-Za-z])(?=[A-Za-z0-9:_/-]*\d)"
    r"[A-Za-z][A-Za-z0-9]*(?:[-:/][A-Za-z0-9]+)+(?![A-Za-z0-9])"
)
_DATE_RE = re.compile(r"\b(?:\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b")
_TIME_RE = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s*[AP]M)?\b", re.I)
# Structural labels / metadata emitted by the extraction pipeline. They are never
# observations, so they must not become standalone claims.
_SECTION_LABELS = (
    "SCENE AND LAYOUT",
    "PEOPLE AND VISIBLE FEATURES",
    "OBJECTS AND POSITIONS",
    "READABLE TEXT",
    "VISIBLE ACTION",
    "VISUAL DETAILS THAT MAY HELP COMPARISON",
    "UNCERTAINTY AND LIMITATIONS",
)
_PREFIX_LABELS = ("IMAGE VISUAL OBSERVATIONS", "VISUAL OBSERVATIONS", "IMAGE OCR TEXT") + _SECTION_LABELS
_LABEL_PREFIX_RE = re.compile(
    r"(?im)^[ \t]*(?:" + "|".join(re.escape(label) for label in _PREFIX_LABELS) + r")[ \t]*:[ \t]*"
)
_METADATA_HEADER_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:VIDEO|AUDIO) METADATA:|\[FRAME \d+[^\]]*\])[^\n]*$"
)
_OCR_CONFIDENCE_RE = re.compile(r"[ \t]*\[\s*OCR confidence:[^\]\[]*\]", re.I)
_PLACEHOLDER_RE = re.compile(r"(?i)^[ \t]*\[\s*no readable text[^\]]*\]\s*$")
_FAILURE_NOTICE_RE = re.compile(
    r"(?i)\b(?:visual analys[ie]s|ocr|transcription|extraction)\s+unavailable\b"
    r"|\bunavailable:\s*(?:ValueError|NameError|RuntimeError|OSError|ImportError|TypeError|ReadTimeout)"
    r"|^(?:ValueError|NameError|RuntimeError|OSError|ImportError|TypeError|ReadTimeoutError)\b"
)
# Phrases produced by extraction instructions and status boilerplate rather than
# by the evidence itself. They never appear in genuine source observations.
_BOILERPLATE_MARKERS = (
    "requested analysis",
    "interpretation rule",
    "model output",
    "verify against original",
    "candidate observations",
    "not ground truth",
    "cannot establish identity",
    "confirm readable text",
    "review the original image",
    "one sampled video frame",
    "record visible objects",
    "separate observations from uncertainty",
    "do not infer",
    "mark blurred or ambiguous details as unclear",
    "candidate alignments",
    "not independently verified event times",
)
# Location/context anchors such as "CAM 02", "ZONE B", "GATE 3", "DOOR D-12".
_LOCATION_RE = re.compile(
    r"\b(?i:cam|camera|zone|gate|door|room|aisle|shelf|bay|dock|lane|terminal|"
    r"counter|entrance|exit|reception|vault)\s*[-#]?\s*"
    r"([A-Z]{0,3}-?\d{1,4}|[A-Z]{1,2})\b"
)
# Cross-modal forensic rules ported from correlation_diagnostic.py. A feature
# described in more than one exhibit is a candidate association, never proof.
_VEHICLE_FEATURE_RULES = {
    "black motorcycle": re.compile(r"\bblack\s+(?:sports[- ]style\s+)?motor(?:cycle|bike)\b", re.I),
    "red tank stripe": re.compile(r"\bred\s+(?:tank\s+)?stripe\b", re.I),
    "left mirror damage": re.compile(
        r"\bleft\s+mirror\b[^\n]{0,50}?\b(?:broken|broke|missing|off|damaged)\b", re.I
    ),
    "altered or obscured plate": re.compile(
        r"\b(?:over[- ]?painted|painted\s+by\s+hand|fresh\s+tape\s+residue|"
        r"plate\s+was\s+dirty|digit\s+over[- ]?painted)\b",
        re.I,
    ),
}
# Vehicle registration plates, spaced (TN 05 / B 7231) or compact (TN05AB7231).
_PLATE_RE = re.compile(r"\b[A-Z]{2}[\s/.-]*0?\d{1,2}[\s/.-]*[A-Z]{1,3}[\s/.-]*\d{3,4}\b")
# Prose markers. Extraction stores OCR as short table cells and overlays, while a
# vision model writes grammatical sentences; this separates the two.
_PROSE_RE = re.compile(
    r"(?i)\b(?:the|this|that|these|those|there|it|he|she|they|his|her|its|"
    r"a|an|is|are|was|were|has|have|had|shows?|showing|displays?|displaying|"
    r"indicates?|appears?|seems?|contains?|includes?|visible|wearing)\b"
)
# Sentences that only describe the medium ("shows a computer screen") carry no
# comparable fact: they must not outrank anchors such as zone, IDs or timestamps.
_META_DESCRIPTION_RE = re.compile(
    r"(?i)(?:shows?|showing|displays?|displaying|depicts?)\s+(?:a|an|the)?\s*"
    r"(?:computer\s+|laptop\s+)?(?:screen|monitor|screenshot|image|photo|video|frame)\b"
    r"|background of the screen"
    r"|blurred view"
)
# A request for the *full* observation set. The bare nouns ("inferences",
# "observations", "facts") are not sufficient on their own: "what inferences can be
# obtained from correlating X and Y" asks for reasoning, not for every extracted
# cell, and treating it as exhaustive flooded the answer with OCR debris.
_EXHAUSTIVE_RE = re.compile(
    r"\b(?:all|every|each|entire|complete|comprehensive|full|everything|"
    r"list|enumerate|walk\s+me\s+through)\b",
    re.I,
)
# Video metadata that qualifies how trustworthy the CCTV clock is, plus the OCR
# confidence the extraction recorded for a chunk.
_VIDEO_METADATA_LINE_RE = re.compile(r"(?im)^[ \t]*VIDEO METADATA:[^\n]*$")
_CAMERA_OFFSET_RE = re.compile(
    r"camera[_ ]?(?:clock[_ ]?)?offset[_ ]?(?:seconds)?\s*[=:]\s*(-?\d+(?:\.\d+)?)",
    re.I,
)
_OCR_CONFIDENCE_VALUE_RE = re.compile(r"\[\s*OCR confidence:\s*([0-9]*\.?[0-9]+)", re.I)
_ANCHOR_KINDS = ("identifier", "date", "time", "location")
# Comparability comes from what a cell can be matched on. An identifying anchor is
# the substance of a link, while a bare timestamp or date is supporting detail, so a
# lone "Zone B" cell must outrank a cell that only carries the timestamp of its row.
_FACT_ANCHOR_WEIGHTS = {"identifier": 3, "location": 3, "date": 1, "time": 1}


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _modality(item: dict) -> str:
    """Resolve modality from explicit metadata/extension before filename hints."""
    raw = str(item.get("modality", "unknown")).casefold().strip()
    name = str(item.get("filename", "")).casefold()
    suffix = re.search(r"\.([a-z0-9]+)$", name)
    ext = suffix.group(1) if suffix else ""
    extension_map = {
        "mp4": "video", "mov": "video", "avi": "video", "mkv": "video",
        "jpg": "image", "jpeg": "image", "png": "image", "webp": "image", "bmp": "image",
        "wav": "audio", "mp3": "audio", "m4a": "audio", "flac": "audio",
        "pdf": "document", "docx": "document", "txt": "document", "csv": "document", "json": "document",
    }
    if raw in {"video", "image", "audio", "document"}:
        return raw
    if ext in extension_map:
        return extension_map[ext]
    hints = raw + " " + name
    if "video" in hints or "cctv" in hints:
        return "video"
    if "image" in hints or "photo" in hints:
        return "image"
    if "audio" in hints:
        return "audio"
    return raw or "unknown"


def _classification(item: dict) -> str:
    explicit = str(item.get("observation_type") or item.get("classification") or "").upper()
    if explicit in CLASSIFICATIONS:
        return explicit
    method = str(item.get("extraction_method") or item.get("method") or "").casefold()
    modality = _modality(item)
    if "ocr" in method or "ocr" in str(item.get("text", "")[:100]).casefold():
        return "OCR_OBSERVATION"
    if modality in {"image", "video"}:
        return "VISION_OBSERVATION"
    if modality == "audio" or "whisper" in method or "transcri" in method:
        return "TRANSCRIPTION"
    return "SOURCE_TEXT"


def _source_ref(item: dict) -> dict:
    loc = item.get("source_location") or item.get("location") or {}
    if not isinstance(loc, dict):
        loc = {}
    return {
        "case_id": item.get("case_id"),
        "evidence_id": item.get("evidence_id"),
        "chunk_id": item.get("chunk_id"),
        "filename": item.get("filename"),
        "sha256": item.get("sha256"),
        "modality": _modality(item),
        "page": item.get("page", loc.get("page")),
        "frame": item.get("frame", loc.get("frame")),
        "timestamp": item.get("timestamp", loc.get("timestamp")),
        "excerpt": str(item.get("text", "")).strip()[:700],
    }


def _normalize_time_text(text: str) -> str:
    """Normalize OCR clock forms such as 21.14.27 or 13.44:05 to colon form.

    Camera overlays render separators inconsistently (dots, colons, or a mix),
    so every separator is normalized before the anchor patterns run.
    """
    normalized = str(text or "")
    normalized = re.sub(
        r"(?<!\d)([01]?\d|2[0-3])[.]([0-5]\d)[.]([0-5]\d)(?!\d)",
        r"\1:\2:\3",
        normalized,
    )
    # Mixed separators: hour:minute.second -> hour:minute:second
    normalized = re.sub(
        r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)[.]([0-5]\d)(?!\d)",
        r"\1:\2:\3",
        normalized,
    )
    # Mixed separators: hour.minute:second -> hour:minute:second
    normalized = re.sub(
        r"(?<!\d)([01]?\d|2[0-3])[.]([0-5]\d):([0-5]\d)(?!\d)",
        r"\1:\2:\3",
        normalized,
    )
    return normalized


def _time_seconds(value: str) -> int | None:
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", value.strip())
    if not match:
        return None
    hour, minute, second = int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)
    if hour > 23 or minute > 59 or second > 59:
        return None
    return hour * 3600 + minute * 60 + second


def _canonical_location(value: str) -> str:
    """Reduce site-qualified variants to one key so "Warehouse Zone B" == "Zone B"."""
    text = _norm(value)
    stripped = re.sub(r"^(?:the\s+)?warehouse\s+", "", text).strip()
    return stripped or text


def _anchors(text: str) -> dict[str, set[str]]:
    normalized = _normalize_time_text(text)
    lower = re.sub(r"[^a-z0-9]+", " ", normalized.casefold())
    locations = set()
    if re.search(r"\bwarehouse\s+zone\s+b\b", lower):
        locations.add("warehouse zone b")
    elif re.search(r"\bzone\s+b\b", lower):
        locations.add("zone b")
    # Generic scene/context anchors (cameras, zones, gates, doors, aisles ...).
    locations.update(_norm(match.group(0)) for match in _LOCATION_RE.finditer(normalized))
    dates = {_norm(x) for x in _DATE_RE.findall(normalized)}
    times = {_norm(x) for x in _TIME_RE.findall(normalized)}
    identifiers = {_norm(x) for x in _ID_RE.findall(normalized)} - dates - times
    return {
        "identifier": identifiers,
        "date": dates,
        "time": times,
        "location": {_canonical_location(value) for value in locations},
    }


def _has_anchor(text: str) -> bool:
    """True when a fragment carries an identifier, date, clock time or place.

    Extraction emits table cells and camera overlays as very short lines
    ("Zone B", "CAM 02", "LOG-3187"). They are the comparable facts, so they are
    kept even though they are shorter than a sentence.
    """
    return any(_anchors(text).values())


def _looks_like_ocr_line(line: str) -> bool:
    """Heuristic OCR-shape test for text that was not tagged by the extractor.

    OCR is stored as short table cells and overlay fragments; a vision model
    emits grammatical prose, which is rejected here.
    """
    stripped = str(line or "").strip()
    if not stripped or len(stripped) > 60 or re.search(r"[?!]", stripped):
        return False
    words = stripped.split()
    if len(words) > 8:
        return False
    if len(words) >= 4 and _PROSE_RE.search(stripped):
        return False
    return True


def _edit_distance(left: str, right: str) -> int:
    """Levenshtein distance between two short identifier strings."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def _identifier_parts(value: str) -> tuple[str, str]:
    """Split an identifier into its alphabetic prefix and trailing digit group."""
    compact = re.sub(r"[^A-Za-z0-9]", "", str(value or "")).upper()
    match = re.fullmatch(r"([A-Z]+)(\d+)", compact)
    if match:
        return match.group(1), match.group(2)
    return compact, ""


def _similar_identifier(candidate: str, reference: str) -> bool:
    """True when a vision-read identifier is a plausible corruption of an OCR one.

    A vision model routinely mis-reads one or two characters of an identifier it is
    shown (OCR ``PRN-2210`` becomes ``EMP-2210``). Such a value is not a second
    identifier; it is a disputed reading of the value OCR actually recorded.
    """
    candidate_compact = re.sub(r"[^A-Za-z0-9]", "", str(candidate or "")).upper()
    reference_compact = re.sub(r"[^A-Za-z0-9]", "", str(reference or "")).upper()
    if not candidate_compact or not reference_compact:
        return False
    if candidate_compact == reference_compact:
        return False
    if _edit_distance(candidate_compact, reference_compact) <= 1:
        return True
    candidate_alpha, candidate_digits = _identifier_parts(candidate_compact)
    reference_alpha, reference_digits = _identifier_parts(reference_compact)
    if not candidate_digits or not reference_digits:
        return False
    # Same numeric body with a different prefix (EMP-2210 vs PRN-2210).
    if candidate_digits == reference_digits:
        return True
    # One digit apart with a nearly identical prefix (EMM-105 vs EMP-104).
    if len(candidate_digits) == len(reference_digits) and _edit_distance(candidate_digits, reference_digits) <= 1:
        if candidate_alpha[:2] and candidate_alpha[:2] == reference_alpha[:2]:
            return True
        if candidate_alpha[-2:] and candidate_alpha[-2:] == reference_alpha[-2:]:
            return True
    return False


def _numeric_digits(digits: str) -> int:
    """Numeric value of an identifier's digit body, or -1 when there is none."""
    return int(digits) if digits and digits.isdigit() else -1


def _closest_ocr_identifier(candidate: str, ocr_values: set[str]) -> str | None:
    """Pick the OCR identifier a vision-read value is most plausibly a corruption of.

    Every OCR value the vision value resembles is a possibility, so the reference
    must be the *closest* one rather than the alphabetically first. Candidates are
    ranked by edit distance, a shared numeric body is preferred, and the remaining
    tie is broken deterministically so the reported reference is stable.
    """
    options = [value for value in ocr_values if _similar_identifier(candidate, value)]
    if not options:
        return None
    candidate_alpha, candidate_digits = _identifier_parts(candidate)

    def rank(value: str) -> tuple[int, int, int, int, int]:
        reference_alpha, reference_digits = _identifier_parts(value)
        return (
            _edit_distance(
                candidate_alpha + candidate_digits, reference_alpha + reference_digits
            ),
            0 if candidate_digits and candidate_digits == reference_digits else 1,
            _edit_distance(candidate_alpha, reference_alpha),
            _edit_distance(candidate_digits, reference_digits),
            -_numeric_digits(reference_digits),
        )

    return min(options, key=rank)


def _token_set(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", _norm(text)))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _meaningful_terms(text: str) -> set[str]:
    terms = set(re.findall(r"\b[a-z0-9]{4,}\b", text.casefold()))
    return terms - _STOP


def _sentences(text: str) -> list[str]:
    """Split extracted text, including literal escaped newlines, into usable observations.

    Extraction wrappers, section labels, OCR confidence annotations, failure
    notices and repeated camera overlays are not treated as independent claims.
    The original chunk remains available in retrieval storage and its ID is
    retained whenever a substantive observation from that chunk is used.
    """
    text = str(text or "").replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    # Remove generic model/extraction instructions, not the observation content.
    text = re.sub(r"(?is)Interpretation rule:.*$", "", text)
    text = re.sub(r"(?is)Requested analysis:.*?(?=SmolVLM2 visual observations|OCR:|IMAGE OCR TEXT:|VISUAL OBSERVATIONS|$)", "", text)
    text = re.sub(r"(?i)SmolVLM2 visual observations \(model output; verify against original\):?", "", text)
    # Structural metadata and section labels are not observations.
    text = _METADATA_HEADER_RE.sub("", text)
    text = _LABEL_PREFIX_RE.sub("", text)
    # "Text [OCR confidence: 0.99]" -> "Text" (the bracket must go with it).
    text = _OCR_CONFIDENCE_RE.sub("", text)
    parts = re.split(r"(?<=[.!?])\s+|[\r\n]+|\s+(?=(?:SCENE AND LAYOUT|PEOPLE AND VISIBLE FEATURES|OBJECTS AND POSITIONS|READABLE TEXT|VISIBLE ACTION|VISUAL DETAILS THAT MAY HELP COMPARISON|UNCERTAINTY AND LIMITATIONS):)", text)
    cleaned = []
    for part in parts:
        candidate = _LABEL_PREFIX_RE.sub("", part)
        candidate = re.sub(r"\s+", " ", candidate).strip(" \t-•[]/|")
        # A short fragment is still a usable observation when it carries an
        # identifier, place, date or clock time ("Zone B", "CAM 02", "LOG-3187").
        if len(candidate) < 12 and not _has_anchor(candidate):
            continue
        if _PLACEHOLDER_RE.match(candidate) or _FAILURE_NOTICE_RE.search(candidate):
            continue
        lowered = candidate.casefold()
        if any(marker in lowered for marker in _BOILERPLATE_MARKERS):
            continue
        compact = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
        # A repeated camera watermark/label alone is metadata, not a substantive event observation.
        if ("simulated cctv" in compact or "camera is labeled" in compact) and not (
            _DATE_RE.search(candidate) or _TIME_RE.search(_normalize_time_text(candidate))
        ):
            continue
        cleaned.append(candidate)
    return cleaned[:40]


def _claim_id(evidence_id: str, chunk_id: str, text: str) -> str:
    raw = f"{evidence_id}|{chunk_id}|{_norm(text)}".encode("utf-8")
    return "clm_" + hashlib.sha256(raw).hexdigest()[:16]


def _claim_classification(item: dict, sentence: str) -> str:
    """Classify one observation, separating OCR text from model vision output.

    A screenshot or video chunk usually carries two very different sources: the
    OCR of what is written in the exhibit, and a vision model's description of
    it. OCR is authoritative for identifiers, so the two must not share a label.
    """
    base = _classification(item)
    if base in {"VISION_OBSERVATION", "SOURCE_TEXT"} and _modality(item) in {"image", "video"}:
        if not _PROSE_RE.search(sentence) or _looks_like_ocr_line(sentence):
            return "OCR_OBSERVATION"
    return base


def build_claims(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract atomic claims and merge near-duplicate statements within an exhibit.

    Every contributing chunk remains in source_refs; deduplication never discards provenance.
    Candidate comparison is bucketed by (exhibit, classification) so the cost stays
    proportional to the number of claims sharing an exhibit rather than all claims.
    Each claim also carries the anchor set of the whole chunk it came from, because
    extraction splits one table row across several single-cell lines.
    """
    claims: list[dict[str, Any]] = []
    buckets: dict[tuple[str, str], list[tuple[frozenset[str], int]]] = defaultdict(list)
    for item in chunks:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or "").strip()
        chunk_id = str(item.get("chunk_id") or "").strip()
        filename = str(item.get("filename") or "").strip()
        text = str(item.get("text") or "").strip()
        if not (evidence_id and chunk_id and filename and text):
            continue
        chunk_anchors = {k: sorted(v) for k, v in _anchors(text).items()}
        for sentence in _sentences(text):
            classification = _claim_classification(item, sentence)
            bucket = buckets[(evidence_id, classification)]
            tokens = _token_set(sentence)
            existing_index = next(
                (index for stored, index in bucket if _jaccard(stored, tokens) >= 0.88),
                None,
            )
            if existing_index is not None:
                existing = claims[existing_index]
                existing["chunk_anchors"][chunk_id] = chunk_anchors
                if not any(r.get("chunk_id") == chunk_id for r in existing["source_refs"]):
                    existing["source_refs"].append(_source_ref(item))
                continue
            claims.append({
                "claim_id": _claim_id(evidence_id, chunk_id, sentence),
                "text": sentence,
                "classification": classification,
                "verification_status": "source_observation_unverified",
                "source_refs": [_source_ref(item)],
                "entities": sorted(_ID_RE.findall(sentence)),
                "anchors": {k: sorted(v) for k, v in _anchors(sentence).items()},
                "chunk_anchors": {chunk_id: chunk_anchors},
            })
            bucket.append((tokens, len(claims) - 1))
    return claims


def _anchor_union(claim: dict[str, Any], evidence_id: str) -> dict[str, set[str]]:
    """Claim anchors plus the anchors of the chunks of this exhibit that carry it.

    Extraction writes a table row as separate one-token lines, so "Zone B" and
    the row timestamp end up in different claims of the same chunk. A chunk is
    one document region, so its anchors are combined before cross-exhibit
    matching; the match is still cited against concrete chunks.
    """
    combined = {
        key: set((claim.get("anchors") or {}).get(key) or [])
        for key in _ANCHOR_KINDS
    }
    chunk_ids = {
        str(ref.get("chunk_id"))
        for ref in claim.get("source_refs") or []
        if ref.get("chunk_id") and str(ref.get("evidence_id")) == str(evidence_id)
    }
    for chunk_id, anchors in (claim.get("chunk_anchors") or {}).items():
        if chunk_id not in chunk_ids:
            continue
        for key in _ANCHOR_KINDS:
            combined[key].update((anchors or {}).get(key) or [])
    return combined


def _anchor_support(
    a: dict[str, Any],
    b: dict[str, Any],
    shared_id: list[str],
    shared_date: list[str],
    shared_location: list[str],
    near_times: list[tuple[str, str, int]],
) -> int:
    """How many of the shared anchors are written in the claims themselves.

    A high value means the produced link quotes the two observations directly
    rather than only their shared chunk context, so it is preferred when the
    matching rule is otherwise identical.
    """
    own_a = a.get("anchors") or {}
    own_b = b.get("anchors") or {}

    def own(key: str) -> set[str]:
        return set(own_a.get(key) or []) | set(own_b.get(key) or [])

    support = 0
    for key, values in (("identifier", shared_id), ("location", shared_location), ("date", shared_date)):
        available = own(key)
        support += sum(1 for value in values if value in available)
    own_times = own("time")
    support += sum(1 for ta, tb, _delta in near_times if ta in own_times or tb in own_times)
    return support


def _own_time_delta(a: dict[str, Any], b: dict[str, Any]) -> int:
    """Closest clock delta between the two claims' own timestamps (large if none)."""
    best = 10**9
    for ta in (a.get("anchors") or {}).get("time") or []:
        for tb in (b.get("anchors") or {}).get("time") or []:
            sa, sb = _time_seconds(ta), _time_seconds(tb)
            if sa is not None and sb is not None:
                best = min(best, abs(sa - sb))
    return best


def correlate_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create source-linked candidate associations across distinct exhibits."""
    by_evidence: dict[str, list[dict]] = defaultdict(list)
    for claim in claims:
        refs = claim.get("source_refs") or []
        evidence_ids = {str(r.get("evidence_id")) for r in refs if r.get("evidence_id")}
        for evidence_id in evidence_ids:
            by_evidence[evidence_id].append(claim)
    links = []
    term_cache: dict[str, set[str]] = {}

    def _terms(claim: dict[str, Any]) -> set[str]:
        key = claim["claim_id"]
        cached = term_cache.get(key)
        if cached is None:
            cached = _meaningful_terms(claim["text"])
            term_cache[key] = cached
        return cached

    for first_id, second_id in combinations(sorted(by_evidence), 2):
        best = None
        for a in by_evidence[first_id]:
            for b in by_evidence[second_id]:
                aa = _anchor_union(a, first_id)
                bb = _anchor_union(b, second_id)
                shared_id = sorted(aa["identifier"] & bb["identifier"])
                shared_date = sorted(aa["date"] & bb["date"])
                shared_time = sorted(aa["time"] & bb["time"])
                shared_location = sorted(aa["location"] & bb["location"])
                shared_terms = sorted(_terms(a) & _terms(b))
                near_times = []
                for ta in sorted(aa["time"]):
                    for tb in sorted(bb["time"]):
                        sa, sb = _time_seconds(ta), _time_seconds(tb)
                        if sa is not None and sb is not None and abs(sa - sb) <= 10:
                            near_times.append((ta, tb, abs(sa - sb)))
                near_times.sort(key=lambda value: (value[2], value[0], value[1]))
                if shared_id:
                    candidate = ("identifier_match", shared_id, 0.85)
                elif shared_location and shared_date and near_times:
                    pair = near_times[0]
                    candidate = ("location_and_temporal_proximity", shared_location + shared_date + [f"{pair[0]}~{pair[1]} ({pair[2]}s apart)"], 0.58)
                elif shared_date and shared_time:
                    candidate = ("date_and_time_match", shared_date + shared_time, 0.60)
                elif shared_location and shared_time:
                    candidate = ("location_and_time_match", shared_location + shared_time, 0.55)
                elif shared_location and shared_date:
                    candidate = ("location_and_date_match", shared_location + shared_date, 0.50)
                # NOTE (H1): date-only match suppressed — a shared calendar date without
                # a second anchor (time, location, or identifier) is too weak to be a
                # meaningful relationship and caused spurious cross-exhibit links.
                elif shared_time:
                    candidate = ("time_match", shared_time, 0.25)
                elif shared_location:
                    candidate = ("location_match", shared_location, 0.20)
                elif len(shared_terms) >= 4:
                    candidate = ("lexical_overlap", shared_terms[:8], 0.10)
                else:
                    continue
                kind, anchors, score = candidate
                support = _anchor_support(a, b, shared_id, shared_date, shared_location, near_times)
                total_anchors = len(shared_id) + len(shared_date) + len(shared_location) + len(near_times)
                delta_match = re.search(r"\((\d+)s apart\)", ",".join(anchors))
                delta = int(delta_match.group(1)) if delta_match else 10**9
                # Strongest rule first, then the pair that quotes the most shared
                # anchors, then the closest own clock times, then the closest pair.
                priority = (score, support, -_own_time_delta(a, b), -delta)
                if best is None or priority > best[0]:
                    best = (priority, kind, anchors, score, a, b, "claim_text" if support >= total_anchors else "claim_text_plus_chunk_context")
        if best:
            _priority, kind, anchors, score, claim_a, claim_b, match_basis = best
            refs_a = [r for r in claim_a["source_refs"] if r.get("evidence_id") == first_id]
            refs_b = [r for r in claim_b["source_refs"] if r.get("evidence_id") == second_id]
            links.append({
                "link_id": "lnk_" + hashlib.sha256(f"{first_id}|{second_id}|{kind}|{anchors}".encode()).hexdigest()[:14],
                "evidence_ids": [first_id, second_id],
                "match_type": kind,
                "shared_anchors": anchors,
                "lead_strength": score,
                "verification_status": "unverified_candidate",
                "match_basis": match_basis,
                "claim_ids": [claim_a["claim_id"], claim_b["claim_id"]],
                "source_refs": refs_a[:3] + refs_b[:3],
                "interpretation": (
                    "An exact identifier string occurs in both exhibits; verify the identifier against original records."
                    if kind == "identifier_match" else
                    "The exhibits share a normalized location/time or date anchor; this is a candidate association, not proof of a common event."
                ),
                "limitations": [
                    "String equality and temporal proximity do not establish identity, authenticity, event continuity, causation, or guilt.",
                    "Clock synchronization and source accuracy have not been independently established.",
                ],
            })
    return links


def _plate_tokens(text: str) -> set[str]:
    """Normalize plate renderings so '/', spaces and separators do not split one plate."""
    return {
        re.sub(r"[^A-Za-z0-9]", "", match.group(0)).upper()
        for match in _PLATE_RE.finditer(str(text or ""))
    }


def _forensic_lead(
    kind: str,
    anchors: list[str],
    by_evidence: dict[str, list[dict[str, Any]]],
    strength: float,
    interpretation: str,
) -> dict[str, Any]:
    evidence_ids = sorted(by_evidence)
    refs = [ref for evidence_id in evidence_ids for ref in by_evidence[evidence_id]][:6]
    return {
        "link_id": "frc_" + hashlib.sha256(
            f"{kind}|{'|'.join(anchors)}|{'|'.join(evidence_ids)}".encode()
        ).hexdigest()[:14],
        "evidence_ids": evidence_ids,
        "match_type": kind,
        "shared_anchors": anchors,
        "lead_strength": strength,
        "verification_status": "unverified_candidate",
        "claim_ids": [],
        "source_refs": refs,
        "interpretation": interpretation,
        "limitations": [
            "A shared description or text string is a candidate association, not proof of identity or a common event.",
            "Model-generated visual descriptions and OCR are unverified and may be wrong.",
        ],
    }


def forensic_correlations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cross-modal forensic leads that only exist when two exhibits agree.

    These are the diagnostic's vehicle-feature and registration-plate rules, kept
    conservative: a lead is emitted only for an attribute that appears in more
    than one evidence item, and every lead carries chunk-level provenance.
    """
    feature_hits: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    plate_hits: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for item in chunks:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or "").strip()
        chunk_id = str(item.get("chunk_id") or "").strip()
        if not (evidence_id and chunk_id):
            continue
        text = str(item.get("text") or "")
        ref = {
            "evidence_id": evidence_id,
            "chunk_id": chunk_id,
            "filename": item.get("filename"),
            "modality": _modality(item),
        }
        for feature, pattern in _VEHICLE_FEATURE_RULES.items():
            if pattern.search(text):
                feature_hits[feature][evidence_id].append(ref)
        for plate in _plate_tokens(text):
            plate_hits[plate][evidence_id].append(ref)

    leads = []
    for feature, by_evidence in feature_hits.items():
        if len(by_evidence) < 2:
            continue
        leads.append(_forensic_lead(
            "shared_vehicle_feature", [feature], by_evidence, 0.45,
            "The same vehicle feature is described in more than one exhibit. Text and "
            "model descriptions alone do not prove the exhibits show the same vehicle.",
        ))
    for plate, by_evidence in plate_hits.items():
        if len(by_evidence) < 2:
            continue
        leads.append(_forensic_lead(
            "registration_plate_match", [plate], by_evidence, 0.55,
            "The same registration plate string occurs in more than one exhibit. Plate "
            "text can be incomplete or altered; verify against the original media.",
        ))
    return leads


def _chunk_metadata(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-observation metadata the extractor recorded for a chunk, if forwarded."""
    for key in ("source_metadata", "frame_metadata"):
        values = chunk.get(key)
        if isinstance(values, list):
            found = [value for value in values if isinstance(value, dict)]
            if found:
                return found
    return []


def _ocr_confidence_values(chunk: dict[str, Any]) -> list[str]:
    """OCR confidences recorded for a chunk, from inline text or forwarded metadata."""
    values = [match.group(1) for match in _OCR_CONFIDENCE_VALUE_RE.finditer(str(chunk.get("text") or ""))]
    for meta in _chunk_metadata(chunk):
        method = " ".join(
            str(meta.get(key) or "")
            for key in ("method", "extraction_method", "observation_type", "kind", "name")
        ).casefold()
        if "ocr" not in method:
            continue
        confidence = meta.get("confidence")
        if isinstance(confidence, bool):
            continue
        if isinstance(confidence, (int, float)):
            values.append(f"{float(confidence):.4f}".rstrip("0").rstrip("."))
        elif isinstance(confidence, str) and confidence.strip():
            values.append(confidence.strip())
    return sorted(set(values))


def _index_caveats(chunks: list[dict[str, Any]]) -> list[str]:
    """Reliability caveats recorded in the index that qualify the observations.

    The CCTV clock offset and the OCR confidence decide how much weight a matched
    timestamp or identifier deserves, so they are surfaced with the answer rather
    than staying buried in chunk metadata.
    """
    offsets: list[str] = []
    confidences: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        blob = "\n".join([str(chunk.get("text") or "")] + [str(meta) for meta in _chunk_metadata(chunk)])
        offsets.extend(match.group(1) for match in _CAMERA_OFFSET_RE.finditer(blob))
        confidences.extend(_ocr_confidence_values(chunk))
    caveats: list[str] = []
    for offset in sorted(set(offsets)):
        try:
            seconds = float(offset)
        except ValueError:
            continue
        if seconds == 0:
            continue
        caveats.append(
            f"A camera clock offset of {seconds:g}s was applied to video timestamps during "
            "extraction; the resulting alignments are candidate alignments, not independently "
            "verified event times."
        )
    for confidence in sorted(set(confidences)):
        caveats.append(
            "Optical character recognition confidence for text in at least one exhibit is "
            f"{confidence}; identifiers and timestamps read by OCR are candidate readings and "
            "should be confirmed against the original media."
        )
    caveats.extend(_frame_clock_notes(chunks))
    return caveats


def _frame_clock_notes(chunks: list[dict[str, Any]]) -> list[str]:
    """Per-frame video clock readings against the adjusted timeline.

    The extractor strips its ``[FRAME n | ... | adjusted_t=...]`` header lines, so the
    camera-clock alignment reached the answer only as generic prose and was never tied
    to a frame. Rebuilding it per frame also exposes a clock that does not advance: a
    repeated on-screen reading while the adjusted timeline moves on is a real lead
    about the recording rather than cosmetic noise.
    """
    for chunk in chunks:
        if not isinstance(chunk, dict) or _modality(chunk) != "video":
            continue
        readings: list[tuple[int | None, float | None, str]] = []
        for meta in _chunk_metadata(chunk):
            if str(meta.get("observation_type") or "") != "video_frame_observation":
                continue
            ocr_text = _normalize_time_text(str(meta.get("ocr_text") or ""))
            times = sorted({value for value in _TIME_RE.findall(ocr_text)})
            if not times:
                continue
            seconds = meta.get("normalized_timeline_seconds")
            readings.append((
                meta.get("frame_index") if isinstance(meta.get("frame_index"), int) else None,
                float(seconds) if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) else None,
                times[0],
            ))
        if len(readings) < 2:
            continue
        ordered = sorted(
            readings,
            key=lambda row: (row[1] is None, row[1] if row[1] is not None else 0.0),
        )
        rendered = ", ".join(
            f"frame {frame} = {time}" + (f" at {seconds:g}s" if seconds is not None else "")
            for frame, seconds, time in ordered
        )
        notes = [
            "Video frame clock readings against the adjusted timeline: "
            f"{rendered}."
        ]
        seen: dict[str, int] = defaultdict(int)
        for _frame, _seconds, time in ordered:
            seen[time] += 1
        repeated = sorted(time for time, count in seen.items() if count > 1)
        if repeated:
            notes.append(
                f"The on-screen video clock repeats {' and '.join(repeated)} across more than one "
                "frame, so it does not advance monotonically; the adjusted timeline and the "
                "on-screen clock therefore disagree and any alignment depends on which is trusted."
            )
        return notes
    return []


def detect_identifier_conflicts(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag identifiers that only a model-generated description asserts.

    OCR of an exhibit is authoritative for identifiers. When a vision sentence
    states an identifier that OCR does not contain, but which closely resembles one
    that OCR does contain, the described value is a disputed reading rather than a
    fact and must be reported as such instead of being presented as a second ID.
    """
    grouped: dict[str, dict[str, list[tuple[str, str, dict]]]] = defaultdict(lambda: defaultdict(list))
    for claim in claims:
        classification = claim.get("classification")
        if classification not in {"OCR_OBSERVATION", "VISION_OBSERVATION"}:
            continue
        text = str(claim.get("text") or "")
        display: dict[str, str] = {}
        for value in _ID_RE.findall(text):
            display.setdefault(_norm(value), value)
        for identifier in (claim.get("anchors") or {}).get("identifier") or []:
            for ref in claim.get("source_refs") or []:
                evidence_id = str(ref.get("evidence_id") or "")
                if not evidence_id:
                    continue
                grouped[evidence_id][classification].append(
                    (identifier, display.get(identifier, identifier), claim)
                )

    conflicts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for evidence_id, groups in grouped.items():
        ocr_entries = groups.get("OCR_OBSERVATION") or []
        vision_entries = groups.get("VISION_OBSERVATION") or []
        if not ocr_entries or not vision_entries:
            continue
        ocr_values = {identifier for identifier, _display, _claim in ocr_entries}
        for identifier, shown, claim in vision_entries:
            if identifier in ocr_values:
                continue
            reference = _closest_ocr_identifier(identifier, ocr_values)
            if reference is None:
                continue
            reference_shown = next(
                (display for value, display, _claim in ocr_entries if value == reference),
                reference,
            )
            disputed = claim.setdefault("disputed_identifiers", [])
            if shown not in disputed:
                disputed.append(shown)
            claim["verification_status"] = "disputed_identifier_unverified"
            key = (evidence_id, identifier, reference)
            entry = conflicts.get(key)
            if entry is None:
                conflicts[key] = {
                    "conflict_id": "idc_" + hashlib.sha256(
                        f"{evidence_id}|{identifier}|{reference}".encode()
                    ).hexdigest()[:14],
                    "evidence_id": evidence_id,
                    "ocr_value": reference_shown,
                    "disputed_value": shown,
                    "topic": "identifier_provenance",
                    "vision_claim_ids": [claim["claim_id"]],
                    "status": "disputed_reading_requires_review",
                    "note": (
                        "OCR of this exhibit is authoritative for identifiers. A model-generated "
                        "visual description states an identifier that OCR does not contain and that "
                        "closely resembles an OCR identifier, so the description is treated as a "
                        "disputed reading, not a fact."
                    ),
                }
            elif claim["claim_id"] not in entry["vision_claim_ids"]:
                entry["vision_claim_ids"].append(claim["claim_id"])
    return list(conflicts.values())


def detect_contradictions(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect only narrow, explicit opposing assertions; avoid semantic guessing."""
    groups: dict[str, list[dict]] = defaultdict(list)
    # Contradiction rules are deliberately conservative: explicit binary states
    # attached to the same exact identifier within source statements.
    patterns = [
        (re.compile(r"\b(door|gate)\b.{0,35}\b(open|unlocked)\b", re.I), "open", "door_state"),
        (re.compile(r"\b(door|gate)\b.{0,35}\b(closed|locked)\b", re.I), "closed", "door_state"),
        (re.compile(r"\b(entry|access|permission)\b.{0,25}\b(granted|allowed|permitted|authorized)\b", re.I), "granted", "access_state"),
        (re.compile(r"\b(entry|access|permission)\b.{0,25}\b(denied|blocked|rejected|unauthorized)\b", re.I), "denied", "access_state"),
        (re.compile(r"\b(present|inside|entered|on[- ]site)\b", re.I), "present", "presence_state"),
        (re.compile(r"\b(absent|outside|exited|off[- ]site)\b", re.I), "absent", "presence_state"),
        (re.compile(r"\b(status|state)\b.{0,25}\b(active|enabled|online)\b", re.I), "active", "status"),
        (re.compile(r"\b(status|state)\b.{0,25}\b(inactive|disabled|offline)\b", re.I), "inactive", "status"),
    ]
    for claim in claims:
        ids = claim.get("anchors", {}).get("identifier", [])
        if not ids:
            continue
        for pattern, state, topic in patterns:
            if pattern.search(claim["text"]):
                for identifier in ids:
                    groups[(identifier, topic)].append((state, claim))
    out = []
    for (identifier, topic), entries in groups.items():
        states = {state for state, _ in entries}
        if len(states) < 2:
            continue
        for (s1, c1), (s2, c2) in combinations(entries, 2):
            if s1 == s2:
                continue
            refs = c1["source_refs"] + c2["source_refs"]
            if refs[0].get("evidence_id") == refs[1].get("evidence_id"):
                continue
            out.append({
                "contradiction_id": "con_" + hashlib.sha256(f"{identifier}|{topic}|{c1['claim_id']}|{c2['claim_id']}".encode()).hexdigest()[:14],
                "subject_anchor": identifier,
                "topic": topic,
                "claim_ids": [c1["claim_id"], c2["claim_id"]],
                "statements": [{"state": s1, "text": c1["text"], "source_refs": c1["source_refs"]},
                               {"state": s2, "text": c2["text"], "source_refs": c2["source_refs"]}],
                "status": "potential_conflict_requires_review",
                "note": "These source statements appear inconsistent under a narrow text rule; inspect originals and context before resolving.",
            })
    return out


def verify_claims(
    claims: list[dict[str, Any]],
    retrieved_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate provenance structure and ensure citations map to actual retrieved chunks (M4)."""
    valid, invalid = [], []
    valid_chunk_ids = (
        {str(c.get("chunk_id")) for c in retrieved_chunks if isinstance(c, dict) and c.get("chunk_id")}
        if retrieved_chunks is not None else None
    )
    for claim in claims:
        refs = claim.get("source_refs")
        ok = bool(
            claim.get("claim_id") and claim.get("text") and
            claim.get("classification") in CLASSIFICATIONS and
            isinstance(refs, list) and refs and
            all(isinstance(r, dict) and all(r.get(k) for k in ("evidence_id", "chunk_id", "filename", "excerpt")) for r in refs) and
            (valid_chunk_ids is None or all(str(r.get("chunk_id")) in valid_chunk_ids for r in refs))
        )
        (valid if ok else invalid).append(claim.get("claim_id"))
    return {
        "status": "passed" if not invalid else "failed",
        "claims_checked": len(claims),
        "valid_claims": len(valid),
        "invalid_claim_ids": invalid,
        "scope": "Structural provenance validation only; verifies citations ground into retrieved source chunks.",
    }


def _question_terms(question: str) -> set[str]:
    return _meaningful_terms(question)


def _claim_fact_score(claim: dict[str, Any]) -> int:
    """Rank observations that carry comparable facts above medium descriptions.

    Lines such as "shows a computer screen" describe the exhibit rather than its
    content, so they must not displace the zone, identifier and timestamp rows a
    comparison actually needs.
    """
    text = str(claim.get("text") or "")
    anchors = claim.get("anchors") or {}
    score = sum(
        _FACT_ANCHOR_WEIGHTS[kind] * len(anchors.get(kind) or [])
        for kind in _ANCHOR_KINDS
    )
    if claim.get("classification") == "OCR_OBSERVATION":
        score += 3
    if _META_DESCRIPTION_RE.search(text):
        score -= 5
    if claim.get("disputed_identifiers"):
        score -= 2
    return score


def _coverage_claim_ok(claim: dict[str, Any]) -> bool:
    """Whether a claim belongs in the full-observation coverage of an exhibit.

    Even an exhaustive request must not list scene filler ("The floor is made of
    concrete.") or a disputed reading: those lines carry no comparable fact, yet the
    coverage path re-admitted them because it sorted on fact score alone.
    """
    if claim.get("disputed_identifiers"):
        return False
    if _META_DESCRIPTION_RE.search(str(claim.get("text") or "")):
        return False
    anchors = claim.get("anchors") or {}
    return any(anchors.get(kind) for kind in _ANCHOR_KINDS)


def _balanced_exhibit_claims(
    claims: list[dict[str, Any]],
    per_exhibit: int = 2,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Select claims round-robin across exhibits so no single exhibit dominates.

    Within an exhibit the claims carrying comparable facts (identifiers, places,
    timestamps) are offered first, so a truncated comparison still shows the lines
    that can actually be compared.
    """
    by_evidence: dict[str, list[dict]] = defaultdict(list)
    for claim in claims:
        if not _coverage_claim_ok(claim):
            continue
        for ref in claim.get("source_refs") or []:
            evidence_id = str(ref.get("evidence_id") or "")
            if evidence_id and all(c["claim_id"] != claim["claim_id"] for c in by_evidence[evidence_id]):
                by_evidence[evidence_id].append(claim)
    for items in by_evidence.values():
        items.sort(key=lambda claim: (-_claim_fact_score(claim), claim["claim_id"]))
    selected: list[dict] = []
    seen: set[str] = set()
    for _ in range(max(1, per_exhibit)):
        for evidence_id in sorted(by_evidence):
            for claim in by_evidence[evidence_id]:
                if claim["claim_id"] in seen:
                    continue
                selected.append(claim)
                seen.add(claim["claim_id"])
                break
            if len(selected) >= limit:
                return selected
    return selected[:limit]


def _link_lines(correlations: list[dict[str, Any]]) -> list[str]:
    """Render the candidate cross-exhibit links that lead the answer body.

    P1: the linked claims are individual OCR cells (often just a bare timestamp or a
    zone label). Presented alone they are unreadable, so the link that joins them is
    stated first, naming the rule, the shared anchors and the strength; the numbered
    observations that follow are flagged as the cells the link was matched on.
    """
    lines = [f"Candidate cross-exhibit links requiring verification: {len(correlations)}."]
    for link in correlations:
        refs_text = "; ".join(
            f"{r.get('filename')} / {r.get('chunk_id')}" for r in link.get("source_refs", [])
        )
        lines.append(
            f"Candidate link ({link['match_type']}, lead strength {link.get('lead_strength', 0.0):.2f}): "
            f"shared anchors {', '.join(link['shared_anchors'])}. "
            f"Supporting chunks: {refs_text}. Status: unverified candidate."
        )
    return lines


# What each correlation rule actually supports. The engine never invents prose, but
# when the question asks what can be inferred the winning rule is a conclusion and
# has to be stated in words instead of leaving the reader to decode anchor tuples.
_RULE_INFERENCE = {
    "identifier_match": (
        "the same identifier string is recorded in both exhibits, so the two records can "
        "be read as referring to the same identifier"
    ),
    "location_and_temporal_proximity": (
        "both exhibits place activity at the same location on the same date within seconds "
        "of each other, so the two records can be read as describing the same time window"
    ),
    "date_and_time_match": (
        "both exhibits carry the same date and clock time, so the two records can be read as "
        "describing the same moment"
    ),
    "location_and_time_match": (
        "both exhibits place activity at the same location at the same clock time, so the two "
        "records can be read as describing the same place at the same time"
    ),
    "location_and_date_match": (
        "both exhibits place activity at the same location on the same date, so the two records "
        "can be read as describing the same site on the same day"
    ),
    "time_match": (
        "both exhibits carry the same clock time, which is weak on its own and needs a second "
        "anchor before it can be treated as a link"
    ),
    "location_match": (
        "both exhibits name the same location, which is weak on its own and needs a second "
        "anchor before it can be treated as a link"
    ),
    "lexical_overlap": (
        "the two exhibits only share wording, which is too weak on its own to support a link"
    ),
}


def _inference_lines(correlations: list[dict[str, Any]]) -> list[str]:
    """State deterministically what the strongest link supports.

    The question ("what can be inferred from correlating ...") asks for a conclusion,
    but the body previously offered only raw cells and anchor tuples. The winning rule
    is a conclusion the engine already computed, so it is written out explicitly and
    bounded by the same caveat that governs every lead.
    """
    if not correlations:
        return []
    lead = max(correlations, key=lambda link: link.get("lead_strength", 0.0))
    anchors = ", ".join(lead.get("shared_anchors") or []) or "no shared anchors"
    return [
        f"Most that can be inferred: {_RULE_INFERENCE.get(lead['match_type'], 'the exhibits share an anchor')} "
        f"(rule {lead['match_type']}, lead strength {lead.get('lead_strength', 0.0):.2f}). "
        f"Shared anchors: {anchors}. "
        "This is a lead to verify against the original exhibits, not proof of a common event, "
        "identity, causation or guilt."
    ]


def generate_answer(question: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Render a deterministic answer with claim-level citations and abstention."""
    claims = build_claims(chunks)
    provenance = verify_claims(claims, retrieved_chunks=chunks)
    if provenance["status"] != "passed":
        return {
            "answer": "I cannot provide a factual answer because one or more retrieved claims failed source-provenance validation.",
            "answer_status": "abstained_provenance_failure",
            "claims": claims, "correlations": [], "contradictions": [],
            "identifier_conflicts": [], "caveats": [],
            "provenance_validation": provenance,
        }
    if not claims:
        return {
            "answer": "I could not find usable, source-attributed observations that answer this question. The system is abstaining rather than guessing.",
            "answer_status": "abstained_no_evidence",
            "claims": [], "correlations": [], "contradictions": [],
            "identifier_conflicts": [], "caveats": [],
            "provenance_validation": provenance,
        }

    q_terms = _question_terms(question)
    q_anchors = _anchors(question)
    # "What are all the inferences ..." asks for the full observation set rather
    # than a short sample, so the display caps are widened for those questions.
    exhaustive = bool(_EXHAUSTIVE_RE.search(question))
    ranked = []
    for claim in claims:
        terms = _meaningful_terms(claim["text"])
        overlap = len(q_terms & terms)
        # exact identifiers/dates in question are given extra weight
        c_anchors = claim.get("anchors", {})
        anchor_hits = sum(len(set(q_anchors[k]) & set(c_anchors.get(k, []))) for k in q_anchors)
        score = overlap + anchor_hits * 5
        # Medium descriptions ("shows a computer screen") describe the exhibit, not
        # its content, so they rank below observations carrying comparable facts.
        medium_penalty = 1 if _META_DESCRIPTION_RE.search(claim["text"]) else 0
        ranked.append((score, medium_penalty, _claim_fact_score(claim), claim))
    ranked.sort(key=lambda x: (x[0], -x[1], x[2], x[3]["claim_id"]), reverse=True)
    relevant = [claim for score, _medium, _facts, claim in ranked if score > 0][: 24 if exhaustive else 12]

    correlations = correlate_claims(claims) + forensic_correlations(chunks)
    contradictions = detect_contradictions(claims)
    identifier_conflicts = detect_identifier_conflicts(claims)
    caveats = _index_caveats(chunks)
    # Claims cited by a cross-exhibit link are the ones whose context has to be
    # shown, because the link was matched on the whole chunk rather than on the
    # single extracted cell.
    linked_ids = {claim_id for link in correlations for claim_id in link.get("claim_ids", [])}
    # For cross-exhibit questions, never let lexical ranking hide the actual
    # source claims selected by the correlation rule. "same"/"both"/"differ" style
    # comparison questions must trigger this too: they rarely share vocabulary with
    # the source text even though the exhibits themselves are comparable.
    cross_exhibit_intent = bool(re.search(
        r"\b(connect|link|relat|correlat|compar|between|across|associat|"
        r"same|both|identical|similar|differ|match|consistent|contradict)\w*\b",
        question, re.I,
    ))
    if (cross_exhibit_intent or exhaustive) and correlations:
        linked_claims = [claim for claim in claims if claim["claim_id"] in linked_ids]
        selected_ids = {claim["claim_id"] for claim in linked_claims}
        relevant = linked_claims + [c for c in relevant if c["claim_id"] not in selected_ids]
        if not exhaustive:
            relevant = relevant[:8]

    # An exhaustive request asks for the full observation set. It is honoured even for
    # a question that shares no vocabulary with the cells and names no comparison,
    # which previously fell through to abstention; scene filler and disputed readings
    # are still excluded by _coverage_claim_ok. The linked claims stay in front.
    if exhaustive:
        coverage = _balanced_exhibit_claims(
            claims,
            per_exhibit=max(1, len(claims)),
            limit=24,
        )
        chosen = {c["claim_id"] for c in relevant}
        relevant = (relevant + [c for c in coverage if c["claim_id"] not in chosen])[:24]

    # A comparison question can legitimately share no vocabulary with the source
    # text. When the case really does hold two or more exhibits, answer with a
    # balanced, provenance-carrying sample of each rather than abstaining.
    comparison_fallback = False
    if not relevant and cross_exhibit_intent:
        balanced = _balanced_exhibit_claims(
            claims,
            per_exhibit=8 if exhaustive else 2,
            limit=24 if exhaustive else 8,
        )
        if len({c["source_refs"][0]["evidence_id"] for c in balanced}) >= 2:
            relevant = balanced
            comparison_fallback = True

    if not relevant:
        status = "abstained_low_relevance"
        intro = ("I could not identify a sufficiently relevant source observation for this question. "
                 "The retrieved material may be related to the case but does not support a reliable answer.")
        selected = []
    elif comparison_fallback:
        status = "answered_with_sources"
        intro = ("No verified cross-exhibit link was established for this comparison. The following are the "
                 "corresponding source-attributed observations from each exhibit, shown so they can be "
                 "compared directly. They are not automatically verified facts.")
        selected = relevant
    else:
        status = "answered_with_sources"
        intro = "The following are source-attributed observations relevant to the question. They are not automatically verified facts."
        selected = relevant

    # P1: the cross-exhibit link leads the body. The claims it cites are single OCR
    # cells (often just a bare timestamp), which are meaningless on their own, so the
    # conclusion the link supports is stated first, then the rule that produced it.
    lines = [intro]
    lines.extend(_inference_lines(correlations))
    if correlations:
        lines.extend(_link_lines(correlations))
    for idx, claim in enumerate(selected, 1):
        citations = []
        for ref in claim["source_refs"][:4]:
            loc_parts = []
            if ref.get("page") is not None: loc_parts.append(f"page {ref['page']}")
            if ref.get("frame") is not None: loc_parts.append(f"frame {ref['frame']}")
            if ref.get("timestamp") is not None: loc_parts.append(f"time {ref['timestamp']}")
            location = ", ".join(loc_parts)
            citation = f"[{ref['filename']} | evidence {ref['evidence_id']} | chunk {ref['chunk_id']}"
            if location: citation += f" | {location}"
            citations.append(citation + "]")
        flag = ""
        if claim.get("disputed_identifiers"):
            flag = f" DISPUTED identifier(s) not present in OCR: {', '.join(claim['disputed_identifiers'])}."
        linked_note = " [linked by a candidate cross-exhibit link above]" if claim["claim_id"] in linked_ids else ""
        # The cell is shown in the separator-normalized form the anchors and the link
        # line already use, so one reading is not printed as "21.14.32" in the body and
        # "21:14:32" in the link above it.
        display_text = _normalize_time_text(claim["text"])
        lines.append(f"{idx}. {display_text} {' '.join(citations)} ({claim['classification']}; unverified observation){flag}{linked_note}")
    if contradictions:
        lines.append(f"Potential conflicts requiring human review: {len(contradictions)}.")
    if identifier_conflicts:
        lines.append(f"Disputed identifiers requiring review: {len(identifier_conflicts)}.")
        for conflict in identifier_conflicts:
            lines.append(
                f"- OCR for {conflict['evidence_id']} reads '{conflict['ocr_value']}'; the model-generated "
                f"visual description instead states '{conflict['disputed_value']}'. OCR is authoritative for "
                "identifiers, so the described value is treated as a disputed reading, not a fact."
            )
    if caveats:
        lines.append("Reliability caveats recorded in the index:")
        for caveat in caveats:
            lines.append(f"- {caveat}")
    lines.append("Limitations: extracted text, OCR, transcription and model-generated visual descriptions may contain errors. Matching identifiers or timestamps does not by itself prove identity, event continuity, causation or guilt. Inspect the original exhibits before drawing conclusions.")

    return {
        "answer": "\n".join(lines),
        "answer_status": status,
        "claims": selected,
        "all_claims": claims,
        "correlations": correlations,
        "contradictions": contradictions,
        "identifier_conflicts": identifier_conflicts,
        "caveats": caveats,
        "provenance_validation": provenance,
        "answer_generation_method": "deterministic_claims_with_provenance",
        "reasoning_model": "deterministic_evidence_rules",
    }
