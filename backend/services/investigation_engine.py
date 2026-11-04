
import csv
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

EVIDENCE_ROOT = Path(
    os.getenv(
        "SHERLOCK_EVIDENCE_DIR",
        str(Path(__file__).resolve().parents[2] / "evidence_storage"),
    )
).resolve()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
TEXT_EXTENSIONS = {".txt", ".csv", ".json"}
TIMESTAMP_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\b"
    r"|\b\d{1,2}:\d{2}:\d{2}\b"
    r"|\b\d{1,2}:\d{2}\s?(?:AM|PM)\b",
    re.IGNORECASE,
)


def resolve_evidence_path(storage_path: str) -> Path:
    relative = Path(storage_path)

    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid evidence storage path.")

    path = (EVIDENCE_ROOT / relative).resolve()

    if not path.is_relative_to(EVIDENCE_ROOT) or not path.is_file():
        raise FileNotFoundError("Evidence file not found in local storage.")

    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def extract_text(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(
            encoding="utf-8-sig", errors="replace"
        ), "text_extraction"

    if suffix == ".csv":
        with path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            rows = list(csv.reader(file))
        return "\n".join(" | ".join(row) for row in rows), "csv_extraction"

    if suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
        return json.dumps(data, indent=2, ensure_ascii=False), "json_extraction"

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [
            f"[PDF page {index}]\n{page.extract_text() or ''}"
            for index, page in enumerate(reader.pages, start=1)
        ]
        return "\n\n".join(pages), "pdf_text_extraction"

    if suffix in IMAGE_EXTENSIONS:
        import pytesseract
        from PIL import Image

        with Image.open(path) as image:
            return pytesseract.image_to_string(image), "image_ocr"

    raise ValueError(f"Unsupported text extraction type: {suffix}")


def extract_video(path: Path, max_frames: int = 12) -> tuple[str, str]:
    import cv2
    import pytesseract
    from PIL import Image

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise ValueError("Unable to open the CCTV video.")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)

        if frame_count <= 0:
            raise ValueError("The video contains no readable frames.")

        count = min(max_frames, frame_count)
        indices = sorted({
            round(i * (frame_count - 1) / max(count - 1, 1))
            for i in range(count)
        })

        observations = [
            f"Video frame count: {frame_count}; FPS: {fps:.3f}"
        ]

        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()

            if not ok:
                continue

            seconds = index / fps if fps > 0 else 0
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(
                Image.fromarray(rgb)
            ).strip()

            observations.append(
                f"[Video-relative time {seconds:.2f}s; frame {index}] "
                f"{text or 'No readable text detected'}"
            )

        return "\n".join(observations), "video_frame_ocr"

    finally:
        capture.release()


def describe_image_bytes(image_bytes: bytes, instruction: str) -> str:
    """Call the shared SmolVLM2 helper without duplicating model loading."""
    try:
        from backend.services.hybrid_rag import _vision_describe_bytes
    except ImportError:
        # Supports deployments where backend/services is directly on sys.path.
        from hybrid_rag import _vision_describe_bytes
    return _vision_describe_bytes(image_bytes, instruction)


def analyze_evidence(record: dict) -> dict:
    path = resolve_evidence_path(record["storage_path"])
    expected_hash = record.get("sha256", "")
    actual_hash = sha256_file(path)

    if not expected_hash or actual_hash.lower() != expected_hash.lower():
        raise ValueError("SHA-256 integrity check failed.")

    extension = Path(
        record.get("original_filename", path.name)
    ).suffix.lower()

    if extension in VIDEO_EXTENSIONS:
        content, method = extract_video(path)
    else:
        content, method = extract_text(path)
        if extension in IMAGE_EXTENSIONS:
            try:
                visual = describe_image_bytes(
                    path.read_bytes(),
                    "Describe visible objects, distinguishing marks, text, and damage. "
                    "State uncertainty; do not identify people or infer guilt.",
                )
                content = "IMAGE VISUAL OBSERVATIONS (unverified):\n" + visual + "\n\nIMAGE OCR TEXT:\n" + content
                method = "smolvlm2_visual_plus_ocr"
            except Exception as exc:
                # Keep OCR results usable if the optional visual model is unavailable.
                content = (
                    "IMAGE VISUAL ANALYSIS UNAVAILABLE: "
                    f"{type(exc).__name__}: {exc}\n\nIMAGE OCR TEXT:\n{content}"
                )
                method = "image_ocr_visual_model_unavailable"

    timestamps = sorted(set(TIMESTAMP_PATTERN.findall(content)))

    return {
        "evidence_id": record["evidence_id"],
        "filename": record.get("original_filename", path.name),
        "modality": record.get("modality", "unknown"),
        "sha256_verified": True,
        "extraction_method": method,
        "extracted_text": content[:50000],
        "extracted_character_count": len(content),
        "candidate_timestamps": timestamps,
        "analysis_status": "completed",
        "limitations": [
            "OCR may omit or misread content.",
            "Video-relative timestamps are not CCTV wall-clock timestamps.",
            "Text extraction does not identify people or establish guilt.",
        ],
    }


def normalize_text(value: str) -> set[str]:
    stop_words = {
        "the", "and", "that", "this", "with", "from", "were",
        "was", "are", "for", "has", "have", "had", "not",
        "into", "before", "after", "then", "there", "their",
    }

    words = set(re.findall(r"\b[a-z0-9]{3,}\b", value.lower()))
    return words - stop_words


def correlate_evidence(analyses: list[dict]) -> dict:
    """Produce transparent lexical leads, not definitive conclusions."""
    leads = []
    usable = [
        item for item in analyses
        if item.get("analysis_status") == "completed"
    ]

    for index, first in enumerate(usable):
        first_terms = normalize_text(first.get("extracted_text", ""))

        for second in usable[index + 1:]:
            second_terms = normalize_text(second.get("extracted_text", ""))
            shared = sorted(first_terms & second_terms)

            # Ignore generic matches; require a few shared terms.
            if len(shared) >= 3:
                leads.append({
                    "evidence_ids": [
                        first["evidence_id"],
                        second["evidence_id"],
                    ],
                    "type": "shared_terms",
                    "shared_terms": shared[:30],
                    "interpretation": (
                        "These evidence extracts share terms. "
                        "Review the original files to determine whether "
                        "the overlap is meaningful."
                    ),
                    "confidence": "unscored",
                })

    return {
        "correlation_method": "lexical_overlap",
        "evidence_analyzed": len(usable),
        "potential_leads": leads,
        "limitations": [
            "Shared words do not prove a causal relationship.",
            "No person identification is performed.",
            "No clock correction is applied without absolute timestamps.",
        ],
    }


def run_investigation(evidence_records: list[dict]) -> dict:
    analyses = []
    failures = []

    for record in evidence_records:
        try:
            analyses.append(analyze_evidence(record))
        except Exception as exc:
            failures.append({
                "evidence_id": record.get("evidence_id", "unknown"),
                "error": "Evidence analysis failed. Check server logs for details.",
                "analysis_status": "failed",
            })

    correlation = correlate_evidence(analyses)

    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "total_evidence": len(evidence_records),
        "completed": len(analyses),
        "failed": len(failures),
        "evidence_analyses": analyses,
        "correlation": correlation,
        "failures": failures,
        "overall_status": (
            "completed" if not failures
            else "partial" if analyses
            else "failed"
        ),
    }
