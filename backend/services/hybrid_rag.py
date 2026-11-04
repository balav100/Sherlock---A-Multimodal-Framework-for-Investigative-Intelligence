"""Mr. Sherlock: multimodal evidence extraction, retrieval and deterministic answers.

Vision: SmolVLM2 (local Transformers)
Answer generation: deterministic evidence rules (no text-generation model dependency)
Retrieval: FAISS
Evidence relationships: Neo4j
Evidence metadata: Firebase Realtime Database
Original evidence: local filesystem
"""

from __future__ import annotations

import csv
import contextlib
import hashlib
import json
import os
import traceback
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


# =========================================================
# 1. CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVIDENCE_ROOT = Path(
    os.getenv(
        "SHERLOCK_EVIDENCE_DIR",
        str(PROJECT_ROOT / "evidence_storage"),
    )
).resolve()

INDEX_ROOT = Path(
    os.getenv(
        "SHERLOCK_INDEX_DIR",
        str(PROJECT_ROOT / "investigation_indexes"),
    )
).resolve()

VISION_MODEL_NAME = os.getenv(
    "SHERLOCK_VISION_MODEL",
    os.getenv(
        "SHERLOCK_FLORENCE_MODEL",
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    ),
)

# Answer generation is deterministic and evidence-grounded; no text LLM is used.
REASONING_MODEL_NAME = "deterministic_evidence_rules"

EMBED_MODEL = os.getenv(
    "SHERLOCK_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

NEO4J_URI = os.getenv(
    "NEO4J_URI",
    "bolt://127.0.0.1:7687",
)

NEO4J_USER = os.getenv(
    "NEO4J_USERNAME",
    "neo4j",
)

NEO4J_PASSWORD = os.getenv(
    "NEO4J_PASSWORD",
    "",
)

NEO4J_DATABASE = os.getenv(
    "NEO4J_DATABASE",
    "neo4j",
)

# Only configure a non-zero offset after checking the camera clock.
CAMERA_CLOCK_OFFSET_SECONDS = int(
    os.getenv(
        "SHERLOCK_CAMERA_OFFSET_SECONDS",
        "0",
    )
)

# Keep the default low for CPU-only laptops.
MAX_PDF_OCR_PAGES = max(0, min(int(os.getenv("SHERLOCK_MAX_PDF_OCR_PAGES", "8")), 50))
PDF_OCR_DPI = max(72, min(int(os.getenv("SHERLOCK_PDF_OCR_DPI", "120")), 200))

MAX_VIDEO_FRAMES = max(
    1,
    min(
        int(
            os.getenv(
                "SHERLOCK_MAX_VIDEO_FRAMES",
                "3",
            )
        ),
        60,
    ),
)

# Incremental-index contract. Bump these when extraction/chunking semantics change.
INDEX_SCHEMA_VERSION = 2
EXTRACTION_PIPELINE_VERSION = "multimodal-extraction-v5"
CHUNKING_PIPELINE_VERSION = "boundary-chunks-850-120-v2"
EMBEDDING_PIPELINE_VERSION = "normalized-sentence-transformers-v1"
_LOCK_GUARD = threading.Lock()
_CASE_LOCKS: dict[str, threading.RLock] = {}
LOCK_TIMEOUT_SECONDS = max(30, int(os.getenv("SHERLOCK_INDEX_LOCK_TIMEOUT", "900")))
CACHE_ROOT = INDEX_ROOT / "_cache"

PIPELINE_NAME = "smolvlm2-faiss-neo4j-deterministic-evidence-rules"


def analysis_record_from_result(
    result: dict,
    *,
    ingested_evidence_ids=None,
    live_evidence_ids=None,
) -> dict:
    """Build the `cases/<case_id>/analysis` update written after an index run.

    Both ingest entry points persist this record, so it is built in one place to keep
    them from drifting apart. Pass the evidence ids the run actually indexed alongside
    the ids present now to catch an index that a concurrent upload overtook: otherwise
    the run reports "ready" while the exhibit it never saw stays unsearchable, and the
    ask route would answer from an incomplete index instead of asking for a rebuild.
    """
    record = {
        "pipeline": PIPELINE_NAME,
        "last_indexed_at": datetime.now(timezone.utc).isoformat(),
        "indexed_chunks": result.get("indexed_chunks", 0),
        "vision_model": result.get("vision_model"),
        "reasoning_model": result.get("reasoning_model"),
        "camera_clock_offset_seconds": result.get("camera_clock_offset_seconds"),
        "index_status": (
            "ready"
            if result.get("published")
            else "stale"
            if result.get("preserved_previous_generation")
            else "failed"
        ),
        "active_generation": result.get("generation"),
        "evidence_manifest_sha256": (result.get("manifest") or {}).get("evidence_manifest_sha256"),
        "graph_status": result.get("graph_status", "unknown"),
    }
    if ingested_evidence_ids is not None and live_evidence_ids is not None:
        if set(ingested_evidence_ids) != set(live_evidence_ids):
            record["index_status"] = "stale"
            record["stale_reason"] = "evidence_changed_during_index"
            record["stale_at"] = datetime.now(timezone.utc).isoformat()
    return record


@contextlib.contextmanager
def _timed_stage(stage: str, **fields):
    """Emit one machine-readable duration record for an indexing stage."""
    started = time.perf_counter()
    status = "ok"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        record = {
            "stage": stage,
            "seconds": round(time.perf_counter() - started, 3),
            "status": status,
            **{key: value for key, value in fields.items() if value is not None},
        }
        print("[INDEX_TIMING] " + json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)


# =========================================================
# 2. EVIDENCE PATHS AND SHA-256
# =========================================================

def _safe_path(storage_path: str) -> Path:
    """Resolve a stored relative path without allowing traversal."""

    relative = Path(storage_path)

    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid evidence storage path.")

    resolved = (EVIDENCE_ROOT / relative).resolve()

    if not resolved.is_relative_to(EVIDENCE_ROOT):
        raise ValueError("Evidence path escapes storage root.")

    return resolved


def _sha256_file(path: Path) -> str:
    """Calculate SHA-256 without loading the entire file into RAM."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


# =========================================================
# 3. LOCAL SMOLVLM2 VISUAL UNDERSTANDING
# =========================================================



@lru_cache(maxsize=1)
def _vision_components():
    """Load SmolVLM2 once for CPU inference."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"[VISION] Loading model: {VISION_MODEL_NAME}")

    # Avoid uncontrolled CPU thread oversubscription on laptop deployments.
    torch_threads = max(1, min(int(os.getenv("SHERLOCK_TORCH_THREADS", "4")), 16))
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only allows this before parallel work has started.
        pass

    try:
        with _timed_stage("vision_model_load", model=VISION_MODEL_NAME, torch_threads=torch_threads):
            processor = AutoProcessor.from_pretrained(VISION_MODEL_NAME)
            # Keep the weights in fp32. The bf16 checkpoint halves resident memory,
            # but PyTorch's CPU bf16 GEMM falls back to an unvectorised path on CPUs
            # without AVX512-BF16 or AMX, which measured ~2x slower per generated
            # token than fp32 here. Extraction throughput matters more than the
            # headroom, so the slower-but-safer memory trade is not taken.
            model = AutoModelForImageTextToText.from_pretrained(
                VISION_MODEL_NAME,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            model.to("cpu")
            model.eval()

        print("[VISION] Model and processor loaded successfully.")
        return processor, model

    except Exception as exc:
        print(f"[VISION LOAD ERROR] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise




def _vision_describe_bytes(image_bytes: bytes, instruction: str) -> str:
    """Describe image bytes with SmolVLM2; output is an unverified lead."""
    import io
    import torch
    from PIL import Image

    stage = "validate image"

    try:
        if not image_bytes:
            raise ValueError("Cannot analyze an empty image.")

        stage = "load vision model"
        processor, model = _vision_components()

        stage = "decode image"
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")

        try:
            stage = "build chat prompt"
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": (
                        "You are a careful visual evidence examiner. Inspect only pixels visible "
                        "in the supplied image. This is one still image, not proof of an event. "
                        "Do not use case context to fill visual gaps. Do not identify a person, "
                        "guess a name, or infer intent, guilt, ownership, or relationships.\n\n"
                        "Answer with one plain paragraph of about six complete sentences. In "
                        "that paragraph cover, in order: the setting and layout; how many "
                        "people are clearly visible and where each person is in the image with "
                        "their visible clothing and items; salient objects and their approximate "
                        "positions; the visible posture of any person; where legible text "
                        "appears (do not transcribe the characters, because a separate OCR pass "
                        "records them); and any uncertainty, blur, glare or occlusion. Do not "
                        "use headings or lists. Finish the final sentence.\n\n"
                        "Distinguish what is clearly visible from what is uncertain. User task: "
                        + instruction
                    )},
                ],
            }]

            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            stage = "preprocess image and prompt"
            with _timed_stage("vision_preprocess"):
                inputs = processor(
                    text=prompt,
                    images=[image],
                    return_tensors="pt",
                )
            inputs = {
                key: value.to("cpu") if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

            print(
                "[VISION DEBUG] Input shapes:",
                {
                    key: tuple(value.shape)
                    for key, value in inputs.items()
                    if hasattr(value, "shape")
                },
            )

            stage = "generate visual description"
            # 256 tokens is ample for the ~6-sentence paragraph the prompt asks
            # for; a natural stop was observed at 99 tokens, so this only guards
            # against a runaway generation. Greedy decoding on a small CPU model
            # can fall into a repeated-phrase loop, so a 4-gram repeat block plus
            # a stronger repetition penalty keeps the paragraph readable.
            with _timed_stage("vision_inference", max_new_tokens=256):
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        num_beams=1,
                        repetition_penalty=1.15,
                        no_repeat_ngram_size=4,
                    )

            stage = "decode generated description"
            input_ids = inputs.get("input_ids")

            if input_ids is not None:
                generated_ids = output_ids[:, input_ids.shape[1]:]
            else:
                generated_ids = output_ids

            caption = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            if "Assistant:" in caption:
                caption = caption.rsplit("Assistant:", 1)[-1].strip()

            if not caption:
                raise RuntimeError(
                    "SmolVLM2 returned an empty description."
                )

            # Return the raw model observations only. Headings already carry the
            # meaning; wrapper sentences and interpretation notes are excluded so
            # they never become searchable evidence text.
            return caption

        finally:
            image.close()

    except Exception as exc:
        print(
            f"[VISION ERROR] Stage: {stage} | "
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise



def _vision_describe(path: Path) -> str:
    """Describe an image file using the shared SmolVLM2 model."""
    return _vision_describe_bytes(
        path.read_bytes(),
        "Record visible objects, their clearly discernible attributes and positions, "
        "visible damage, and any legible text. Separate observations from uncertainty; "
        "do not infer identity, ownership, intent, or cause.",
    )


# =========================================================
# 4. OCR
# =========================================================

_EASYOCR_READER = None


def _get_easyocr_reader():
    """Initialize EasyOCR once and reuse it."""

    global _EASYOCR_READER

    if _EASYOCR_READER is None:
        import easyocr

        with _timed_stage("easyocr_model_load", languages="en", device="cpu"):
            _EASYOCR_READER = easyocr.Reader(
                ["en"],
                gpu=False,
                verbose=False,
            )

    return _EASYOCR_READER



def _image_ocr(path: Path) -> tuple[str, float | None]:
    """Extract text and return it clean of inline metadata.

    Returns the joined OCR text plus a mean detection confidence. Confidence is
    reported as structured metadata, never as inline ``[...]`` text, so it cannot
    fragment sentences or leak into searchable evidence claims.
    """

    reader = _get_easyocr_reader()

    results = reader.readtext(
        str(path),
        detail=1,
        paragraph=False,
        batch_size=1,
    )

    extracted: list[str] = []
    confidences: list[float] = []

    for bbox, text, confidence in results:
        text = text.strip()

        if not text:
            continue

        extracted.append(text)
        try:
            confidences.append(float(confidence))
        except (TypeError, ValueError):
            continue

    mean_confidence = (
        round(sum(confidences) / len(confidences), 4) if confidences else None
    )
    return "\n".join(extracted), mean_confidence



# =========================================================
# 5. MULTIMODAL EVIDENCE EXTRACTION
# =========================================================

@lru_cache(maxsize=1)
def _whisper_model():
    """Load one CPU/int8 Faster-Whisper model per worker process."""
    from faster_whisper import WhisperModel

    model_name = os.getenv("SHERLOCK_WHISPER_MODEL", "base").strip() or "base"
    with _timed_stage("whisper_model_load", model=model_name, device="cpu", compute_type="int8"):
        return WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=max(1, int(os.getenv("SHERLOCK_WHISPER_THREADS", "4"))))


def _pdf_page_ocr(path: Path, page_index: int) -> tuple[str, float | None]:
    """OCR one rendered PDF page. Optional dependencies fail explicitly.

    Returns the joined page text plus a mean detection confidence, reported as
    structured metadata rather than inline ``[...]`` text.
    """
    import fitz
    document = fitz.open(str(path))
    try:
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(dpi=PDF_OCR_DPI, alpha=False)
        image_bytes = pixmap.tobytes("png")
        import io
        from PIL import Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        reader = _get_easyocr_reader()
        results = reader.readtext(image, detail=1, paragraph=False, batch_size=1)
        lines: list[str] = []
        confidences: list[float] = []
        for _bbox, text, confidence in results:
            text = str(text).strip()
            if not text:
                continue
            lines.append(text)
            try:
                confidences.append(float(confidence))
            except (TypeError, ValueError):
                continue
        mean_confidence = (
            round(sum(confidences) / len(confidences), 4) if confidences else None
        )
        return "\n".join(lines), mean_confidence
    finally:
        document.close()


def _observation(kind: str, evidence_location: dict[str, Any], text: str, method: str, confidence: float | None = None, status: str = "extracted") -> dict[str, Any]:
    """Create a consistent, provenance-bearing extraction observation."""
    return {
        "schema_version": 1,
        "observation_type": kind,
        "source_location": evidence_location,
        "text": text,
        "extraction_method": method,
        "confidence": confidence,
        "verification_status": "unverified_machine_observation",
        "processing_status": status,
    }

def _extract(
    path: Path,
    filename: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Extract searchable observations from supported evidence."""

    extension = Path(filename).suffix.lower()

    # -----------------------------------------------------
    # Plain text
    # -----------------------------------------------------

    if extension in {".txt", ".log"}:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        docs = [_observation("text_document", {"line_range": "whole_file"}, text, "utf8_text_reader", status="extracted" if text else "empty")]
        return text, docs

    # -----------------------------------------------------
    # CSV
    # -----------------------------------------------------

    if extension == ".csv":
        with path.open(
            encoding="utf-8-sig",
            newline="",
            errors="replace",
        ) as file:
            rows = csv.reader(file)

            extracted = "\n".join(
                " | ".join(row)
                for row in rows
            )

        docs = [_observation("tabular_text", {"row_range": "whole_file"}, extracted, "csv_reader", status="extracted" if extracted else "empty")]
        return extracted, docs

    # -----------------------------------------------------
    # JSON
    # -----------------------------------------------------

    if extension == ".json":
        data = json.loads(
            path.read_text(encoding="utf-8-sig")
        )

        extracted = json.dumps(data, ensure_ascii=False, indent=2)
        docs = [_observation("structured_json", {"json_path": "$"}, extracted, "json_parser", status="extracted")]
        return extracted, docs

    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    if extension == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        page_texts: list[str] = []
        page_docs: list[dict[str, Any]] = []
        for page_index, page in enumerate(reader.pages):
            native_text = (page.extract_text() or "").strip()
            method = "pypdf_text"
            text = native_text
            page_confidence: float | None = None
            if len(native_text) < 20 and page_index < MAX_PDF_OCR_PAGES:
                try:
                    with _timed_stage("pdf_page_ocr", filename=filename, page=page_index + 1):
                        ocr_text, page_confidence = _pdf_page_ocr(path, page_index)
                    if ocr_text.strip():
                        text = ocr_text
                        method = "pdf_render_easyocr_fallback"
                except Exception as exc:
                    # Keep native extraction and expose the OCR failure as status metadata.
                    page_docs.append(_observation(
                        "pdf_page_ocr_failure", {"page": page_index + 1},
                        f"{type(exc).__name__}: {exc}", "pdf_ocr_fallback", status="failed"
                    ))
            page_texts.append(f"[PAGE {page_index + 1}]\n{text}")
            page_docs.append(_observation(
                "document_page_text", {"page": page_index + 1}, text,
                method, confidence=page_confidence, status="extracted" if text else "empty"
            ))
        return "\n\n".join(page_texts), page_docs

    # -----------------------------------------------------
    # Images: vision model + OCR
    # -----------------------------------------------------

    if extension in {".jpg", ".jpeg", ".png", ".webp"}:
        parts = []
        visual = ""
        ocr_text = ""
        ocr_confidence: float | None = None
        visual_failed = False
        ocr_failed = False

        try:
            with _timed_stage("image_visual_understanding", filename=filename):
                visual = _vision_describe(path)

            if visual:
                parts.append(visual)

        except Exception as exc:
            visual_failed = True
            import traceback
            traceback.print_exc()

            print(f"VISION ERROR: {type(exc).__name__}: {exc}")

        try:
            with _timed_stage("image_ocr", filename=filename):
                ocr_text, ocr_confidence = _image_ocr(path)

            if ocr_text:
                parts.append(ocr_text)

        except Exception as exc:
            ocr_failed = True

        if visual_failed and ocr_failed:
            raise RuntimeError(
                "Both image visual analysis and OCR failed."
            )

        image_observations = [
            _observation("image_visual_description", {"image": 1}, visual, "smolvlm2", status="extracted" if not visual_failed else "failed"),
            _observation("image_ocr_text", {"image": 1}, ocr_text if not ocr_failed else "", "easyocr", confidence=ocr_confidence, status="extracted" if not ocr_failed else "failed"),
        ]
        return "\n\n".join(parts), image_observations
    
    # -----------------------------------------------------
    # Video: sample frames + vision + OCR
    # -----------------------------------------------------

    if extension in {".mp4", ".mov", ".avi", ".mkv"}:
        import cv2

        capture = cv2.VideoCapture(str(path))

        if not capture.isOpened():
            raise RuntimeError(
                "OpenCV could not open this video."
            )

        frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = float(
            capture.get(cv2.CAP_PROP_FPS) or 0
        )

        if frame_count <= 0:
            capture.release()

            raise RuntimeError(
                "Video contains no readable frames."
            )

        sample_count = min(
            MAX_VIDEO_FRAMES,
            frame_count,
        )

        indices = sorted({
            round(
                index * (frame_count - 1)
                / max(sample_count - 1, 1)
            )
            for index in range(sample_count)
        })

        frame_docs: list[dict[str, Any]] = []

        try:
            for frame_index in indices:
                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    frame_index,
                )

                ok, frame = capture.read()

                if not ok:
                    continue

                video_seconds = (
                    frame_index / fps
                    if fps > 0
                    else None
                )

                adjusted_seconds = (
                    video_seconds
                    + CAMERA_CLOCK_OFFSET_SECONDS
                    if video_seconds is not None
                    else None
                )

                # OCR is independently attempted. Failures are recorded as
                # structured metadata, never as searchable body text.
                ocr_text = ""
                ocr_error = ""
                ocr_failed = False

                try:
                    with _timed_stage("video_frame_ocr", filename=filename, frame_index=frame_index):
                        reader = _get_easyocr_reader()
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = reader.readtext(rgb_frame, detail=0, paragraph=True)
                        ocr_text = "\n".join(item.strip() for item in results if item.strip())

                except Exception as exc:
                    ocr_failed = True
                    ocr_error = f"{type(exc).__name__}: {exc}"

                # Visual analysis is independently attempted.
                visual_text = ""
                vision_error = ""
                visual_failed = False

                try:
                    success, buffer = cv2.imencode(
                        ".jpg",
                        frame,
                        [
                            int(cv2.IMWRITE_JPEG_QUALITY),
                            85,
                        ],
                    )

                    if not success:
                        raise RuntimeError(
                            "Could not encode sampled video frame."
                        )

                    with _timed_stage("video_frame_visual_understanding", filename=filename, frame_index=frame_index):
                        visual_text = _vision_describe_bytes(
                            buffer.tobytes(),
                            (
                            "This is one sampled video frame, not the full video. Describe "
                            "only visible objects, clothing colours when clear, approximate "
                            "positions, scene layout, visible interactions, and legible text. "
                            "Do not infer movement from this single frame, identify a person, "
                            "or infer intent or guilt. Mark blurred or ambiguous details as unclear."
                        ),
                    )

                except Exception as exc:

                    import traceback
                    traceback.print_exc()

                    print(f"VISION ERROR: {type(exc).__name__}: {exc}")

                    visual_failed = True
                    vision_error = f"{type(exc).__name__}: {exc}"

                extraction_errors = []
                if ocr_failed:
                    extraction_errors.append(f"ocr: {ocr_error}")
                if visual_failed:
                    extraction_errors.append(f"vision: {vision_error}")

                if ocr_failed and visual_failed:
                    processing_status = "failed"
                elif extraction_errors:
                    processing_status = "partial"
                else:
                    processing_status = "extracted"

                frame_docs.append({
                    "schema_version": 1,
                    "observation_type": "video_frame_observation",
                    "source_location": {"frame_index": frame_index, "video_timestamp_seconds": video_seconds},
                    "extraction_method": "opencv+smolvlm2+easyocr",
                    "verification_status": "unverified_machine_observation",
                    "processing_status": processing_status,
                    "extraction_errors": extraction_errors,
                    "frame_index": frame_index,
                    "video_timestamp_seconds": video_seconds,
                    "normalized_timeline_seconds": adjusted_seconds,
                    "camera_clock_offset_seconds_applied": (
                        CAMERA_CLOCK_OFFSET_SECONDS
                    ),
                    "timestamp_basis": (
                        "Video-relative timestamp plus configured "
                        "offset. This is not independently verified."
                    ),
                    "ocr_text": ocr_text,
                    "visual_observations": visual_text,
                })

        finally:
            capture.release()

        if not frame_docs:
            raise RuntimeError(
                "No video frames could be sampled."
            )

        header = (
            f"VIDEO METADATA: frame_count={frame_count}; "
            f"fps={fps}; sampled_frames={len(frame_docs)}; "
            f"camera_offset_seconds="
            f"{CAMERA_CLOCK_OFFSET_SECONDS}. "
            "Adjusted timestamps are candidate alignments, "
            "not independently verified event times."
        )

        def _stamp(value: float | None) -> str:
            return f"{round(value, 2)}s" if value is not None else "unknown"

        frame_blocks = []
        for item in frame_docs:
            lines = [
                f"[FRAME {item['frame_index']} | "
                f"video_t={_stamp(item['video_timestamp_seconds'])} | "
                f"adjusted_t={_stamp(item['normalized_timeline_seconds'])}]"
            ]
            if item["visual_observations"]:
                lines.append(item["visual_observations"])
            if item["ocr_text"]:
                lines.append(item["ocr_text"])
            frame_blocks.append("\n".join(lines))

        body = "\n\n".join(frame_blocks)

        return header + "\n\n" + body, frame_docs

    # -----------------------------------------------------
    # Audio
    # -----------------------------------------------------

    if extension in {".wav", ".mp3"}:
        try:
            model = _whisper_model()
            with _timed_stage("audio_transcription", filename=filename, whisper_model=os.getenv("SHERLOCK_WHISPER_MODEL", "base")):
                segments, info = model.transcribe(
                    str(path),
                    beam_size=1,
                    vad_filter=True,
                    word_timestamps=False,
                )
            transcript_parts: list[str] = []
            audio_docs: list[dict[str, Any]] = []
            for segment in segments:
                segment_text = str(segment.text).strip()
                if not segment_text:
                    continue
                location = {"start_seconds": round(float(segment.start), 3), "end_seconds": round(float(segment.end), 3)}
                transcript_parts.append(f"[AUDIO {location['start_seconds']:.2f}-{location['end_seconds']:.2f}s] {segment_text}")
                audio_docs.append(_observation(
                    "audio_transcript_segment", location, segment_text,
                    f"faster_whisper:{os.getenv('SHERLOCK_WHISPER_MODEL', 'base')}",
                    confidence=None,
                ))
            if not transcript_parts:
                raise RuntimeError("Audio transcription produced no speech segments.")
            language = getattr(info, "language", None)
            header = f"AUDIO METADATA: detected_language={language or 'unknown'}; source_duration_seconds={getattr(info, 'duration', None)}"
            return header + "\n\n" + "\n".join(transcript_parts), audio_docs
        except ImportError as exc:
            raise RuntimeError("Audio transcription requires faster-whisper. Install it in the active Python environment.") from exc

    raise ValueError(
        f"Unsupported evidence type: {extension}"
    )


# =========================================================
# 6. TEXT CHUNKING
# =========================================================

def _chunks(
    text: str,
    size: int = 850,
    overlap: int = 120,
) -> list[str]:
    """Split extracted evidence into overlapping text chunks.

    Chunk boundaries are snapped to paragraph, line, or sentence ends whenever a
    suitable one exists near the target size. This prevents retrieval chunks from
    beginning or ending mid-word or mid-token, while retaining the exact
    character-window behaviour for text that has no internal boundaries.
    """

    if size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= size:
        raise ValueError(
            "Chunk overlap must be non-negative and smaller than size."
        )

    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        raise ValueError("No machine-readable evidence text was extracted; refusing to index a placeholder.")

    length = len(text)
    inner_boundaries = sorted({
        match.end()
        for match in re.finditer(r"\n+|(?<=[.!?])\s+", text)
        if 0 < match.end() < length
    })
    minimum_chunk = max(1, size // 2)

    result = []
    start = 0

    while start < length:
        end = min(length, start + size)

        if end < length and inner_boundaries:
            floor = start + minimum_chunk
            candidates = [
                boundary for boundary in inner_boundaries
                if floor <= boundary <= end
            ]
            if candidates:
                end = candidates[-1]

        result.append(text[start:end])

        if end >= length:
            break

        next_start = end - overlap
        if inner_boundaries:
            boundary = next(
                (point for point in inner_boundaries if point >= next_start),
                None,
            )
            if boundary is not None and boundary < end:
                next_start = boundary

        start = max(next_start, start + 1)

    return result


# =========================================================
# 7. EMBEDDING MODEL
# =========================================================

@lru_cache(maxsize=1)
def _embedder():
    """Load the sentence embedding model once per process."""

    from sentence_transformers import SentenceTransformer

    with _timed_stage("embedding_model_load", model=EMBED_MODEL):
        return SentenceTransformer(
            EMBED_MODEL,
            local_files_only=True,
        )


# =========================================================
# 8. CASE-SPECIFIC INDEX DIRECTORY
# =========================================================

def _case_dir(case_id: str) -> Path:
    """Return a safe, case-specific index directory."""

    if not case_id or not case_id.strip():
        raise ValueError("Case ID cannot be empty.")

    safe_case_id = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        case_id,
    )

    folder = (INDEX_ROOT / safe_case_id).resolve()

    if not folder.is_relative_to(INDEX_ROOT):
        raise ValueError("Invalid case index path.")

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    return folder


# =========================================================
# 9. NEO4J CONNECTION
# =========================================================

def _neo4j_driver():
    """Return the process-wide Neo4j driver (created lazily)."""
    from backend.services.graph_rag import get_driver
    return get_driver()


# =========================================================
# 10. CANDIDATE ENTITY EXTRACTION
# =========================================================

# Candidate-entity vocabulary for graph linking. Keyword matching is scoped
# case-insensitive while identifier/label/name bodies keep their case, so plain
# prose cannot become an entity. Without plates, day-first dates and location
# labels the graph never contained the terms case questions actually use.
_ENTITY_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        # Exhibit and record identifiers: EMP-104, EVD-12, CASE-..., TXN-..., ATM-...
        r"\b(?i:emp|evd|case|txn|atm)[-_ ]?[A-Z0-9]{2,}\b",
        # Prefixed report identifiers: LOG-3187, LPT-8876, JW-2026-101, CR-2026-088.
        r"\b[A-Z]{2,6}[-/]\d{3,6}(?:[-/]\d{2,4})*\b",
        # Vehicle registration plates, spaced (TN 05 / B 7231) or compact (TN05AB7231).
        r"\b[A-Z]{2}[\s/.-]*0?\d{1,2}[\s/.-]*[A-Z]{1,3}[\s/.-]*\d{3,4}\b",
        r"\b[A-Z]{2}0?\d{1,2}[A-Z]{1,3}\d{3,4}\b",
        # Clock times, ISO dates, and day-first dates as case documents write them.
        r"\b\d{1,2}:\d{2}(?::\d{2})?\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        # Camera/location labels: CAM 02, ZONE B, GATE 3, DOOR D-12.
        r"\b(?i:cam|camera|zone|gate|door|room|aisle|shelf|bay|dock|lane|terminal|"
        r"counter|entrance|exit|reception|vault)"
        r"\s*[-#]?\s*(?i:[A-Z]{1,3}[- ]?\d{1,4}|\d{1,4}|[A-Z])(?![A-Za-z])",
        # Standalone place names used in case narratives.
        r"\b(?i:showroom|storeroom|warehouse|reception|vault|parking lot|loading dock)\b",
        # Titled or role-labelled person names: "Mr Ravi Kumar", "Owner: S. Rao".
        r"\b(?:Mr|Mrs|Ms|Dr|Shri|Smt)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b",
        r"\b(?i:complainant|witness|owner|suspect|accused|driver|rider|informant)"
        r"\s*[:=-]?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b",
        # Named card/terminal/account/device references (legacy rule kept).
        r"\b(?i:card|terminal|transaction|account|device)\s*(?:id|no\.?|number|#)?"
        r"\s*[:=-]?\s*[A-Z0-9-]{3,}\b",
    )
)


def _entities(text: str) -> list[str]:
    """Extract candidate identifiers for graph linking.

    These are possible references, not verified facts. The vocabulary covers
    registration plates, case-document dates, camera/location labels, report
    identifiers and titled names so a question's terms can reach the graph.
    """

    found: dict[str, str] = {}

    for pattern in _ENTITY_PATTERNS:
        for match in pattern.finditer(str(text or "")):
            raw = re.sub(r"\s+", " ", match.group(0)).strip()
            if raw:
                found.setdefault(raw.casefold(), raw)

    return sorted(found.values())[:100]


# =========================================================
# 11. NEO4J GRAPH STORAGE
# =========================================================

def _write_graph(
    case_id: str,
    evidence_id: str,
    chunk_id: str,
    filename: str,
    modality: str,
    text: str,
    sha256: str,
) -> None:
    """Compatibility wrapper; graph writes are batched by evidence in ingest_case."""
    from backend.services.graph_rag import upsert_evidence_chunks
    upsert_evidence_chunks(case_id, evidence_id, [{
        "chunk_id": chunk_id, "filename": filename, "modality": modality,
        "text": text, "sha256": sha256, "entities": _entities(text),
    }])


# =========================================================
# 12. FAISS INDEX PERSISTENCE
# =========================================================

def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with temporary.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _case_lock(case_id: str):
    """Cross-thread and cross-process case lock using an atomic lock directory."""
    import contextlib

    @contextlib.contextmanager
    def manager():
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", case_id)
        with _LOCK_GUARD:
            local_lock = _CASE_LOCKS.setdefault(safe, threading.RLock())
        with local_lock:
            lock_dir = _case_dir(case_id) / ".index.lock"
            started = time.monotonic()
            while True:
                try:
                    lock_dir.mkdir()
                    _atomic_write_json(lock_dir / "owner.json", {
                        "pid": os.getpid(), "created_at": time.time()
                    })
                    break
                except FileExistsError:
                    # Recover only clearly stale locks; never steal a live lock.
                    try:
                        age = time.time() - lock_dir.stat().st_mtime
                        if age > LOCK_TIMEOUT_SECONDS * 2:
                            shutil.rmtree(lock_dir, ignore_errors=True)
                            continue
                    except OSError:
                        pass
                    if time.monotonic() - started > LOCK_TIMEOUT_SECONDS:
                        raise TimeoutError(f"Timed out waiting for index lock for case {case_id}.")
                    time.sleep(0.25)
            try:
                yield
            finally:
                shutil.rmtree(lock_dir, ignore_errors=True)
    return manager()


def _active_generation(case_id: str) -> tuple[Path, dict | None]:
    folder = _case_dir(case_id)
    pointer = folder / "active.json"
    if pointer.is_file():
        data = json.loads(pointer.read_text(encoding="utf-8"))
        generation = data.get("generation")
        if not isinstance(generation, str) or not generation or "/" in generation or "\\" in generation:
            raise ValueError("Active index generation pointer is invalid.")
        target = folder / "generations" / generation
        if not target.is_dir():
            raise ValueError("Active index generation directory is missing.")
        return target, data
    # Read-only compatibility with the pre-generation index layout.
    if (folder / "evidence.faiss").is_file() and (folder / "chunks.json").is_file():
        return folder, None
    raise FileNotFoundError(f"No complete evidence index exists for case {case_id}.")


def _publish_generation(case_id: str, vectors, metadata: list[dict], manifest: dict) -> dict:
    import faiss
    import numpy as np

    if len(vectors) != len(metadata) or not vectors:
        raise ValueError("Cannot publish an empty or mismatched FAISS generation.")
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix is invalid.")
    with _timed_stage("faiss_index_build", vectors=len(vectors), dimension=int(matrix.shape[1])):
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

    folder = _case_dir(case_id)
    generations = folder / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    generation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:10]
    staging = generations / (".staging-" + generation_id)
    final = generations / generation_id
    staging.mkdir()
    try:
        with _timed_stage("faiss_index_write", vectors=int(index.ntotal)):
            faiss.write_index(index, str(staging / "evidence.faiss"))
        with _timed_stage("chunks_metadata_write", chunks=len(metadata)):
            _atomic_write_json(staging / "chunks.json", metadata)
        manifest = {
            **manifest,
            "schema_version": INDEX_SCHEMA_VERSION,
            "generation": generation_id,
            "vector_count": int(index.ntotal),
            "embedding_dimension": int(index.d),
            "published_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_json(staging / "manifest.json", manifest)
        # Directory rename publishes all three immutable files together.
        staging.replace(final)
        pointer = {"generation": generation_id, "published_at": manifest["published_at"]}
        _atomic_write_json(folder / "active.json", pointer)
        return {"generation": generation_id, "manifest": manifest, "directory": str(final)}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _cached_extraction(path: Path, sha256: str, filename: str) -> tuple[str, list[dict]]:
    cache_dir = CACHE_ROOT / "extractions"
    cache_file = cache_dir / f"{sha256}-{EXTRACTION_PIPELINE_VERSION}.json"
    if cache_file.is_file():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            if payload.get("pipeline_version") == EXTRACTION_PIPELINE_VERSION and isinstance(payload.get("text"), str) and isinstance(payload.get("observations"), list):
                print("[INDEX_CACHE] " + json.dumps({"kind": "extraction", "result": "hit", "sha256": sha256[:12]}, sort_keys=True), flush=True)
                return payload["text"], payload["observations"]
        except (OSError, json.JSONDecodeError):
            pass
    print("[INDEX_CACHE] " + json.dumps({"kind": "extraction", "result": "miss", "sha256": sha256[:12]}, sort_keys=True), flush=True)
    with _timed_stage("evidence_extraction", filename=filename):
        text, observations = _extract(path, filename)
    _atomic_write_json(cache_file, {
        "pipeline_version": EXTRACTION_PIPELINE_VERSION,
        "sha256": sha256,
        "text": text,
        "observations": observations,
    })
    return text, observations


def _cached_embeddings(embedder, sha256: str, chunks: list[str]):
    import numpy as np
    cache_dir = CACHE_ROOT / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_digest = hashlib.sha256("\0".join(chunks).encode("utf-8")).hexdigest()
    model_digest = hashlib.sha256(f"{EMBED_MODEL}|{EMBEDDING_PIPELINE_VERSION}".encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{sha256}-{chunk_digest}-{model_digest}.npy"
    if cache_file.is_file():
        try:
            vectors = np.load(cache_file, allow_pickle=False)
            if vectors.ndim == 2 and vectors.shape[0] == len(chunks) and np.isfinite(vectors).all():
                print("[INDEX_CACHE] " + json.dumps({"kind": "embeddings", "result": "hit", "sha256": sha256[:12], "chunks": len(chunks)}, sort_keys=True), flush=True)
                return vectors
        except (OSError, ValueError):
            pass
    print("[INDEX_CACHE] " + json.dumps({"kind": "embeddings", "result": "miss", "sha256": sha256[:12], "chunks": len(chunks)}, sort_keys=True), flush=True)
    with _timed_stage("embedding_encode", chunks=len(chunks), model=EMBED_MODEL):
        vectors = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or not np.isfinite(vectors).all():
        raise RuntimeError("Embedding output is invalid or does not match chunks.")
    temporary = cache_file.with_name(cache_file.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, vectors, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(cache_file)
    finally:
        temporary.unlink(missing_ok=True)
    return vectors


# =========================================================
# 13. CASE EVIDENCE INGESTION
# =========================================================

def ingest_case(case_id: str, records: dict[str, dict]) -> dict:
    """Incrementally build and atomically publish a complete case index.

    Emits per-stage [INDEX_TIMING] records and [INDEX_CACHE] hit/miss records.
    Content-addressed caches and atomic generation publication are preserved.
    """
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("Case ID cannot be empty.")
    if not isinstance(records, dict) or not records:
        raise ValueError("No evidence records were provided.")

    ingest_started = time.perf_counter()
    lock_wait_started = time.perf_counter()
    with _case_lock(case_id):
        print("[INDEX_TIMING] " + json.dumps({"stage": "case_lock_wait", "case_id": case_id, "seconds": round(time.perf_counter() - lock_wait_started, 3), "status": "ok"}, sort_keys=True), flush=True)
        with _timed_stage("embedding_model_ready", case_id=case_id):
            embedder = _embedder()
        all_vectors, metadata, summaries, warnings = [], [], [], []
        evidence_manifest = []

        for evidence_id, record in records.items():
            evidence_started = time.perf_counter()
            if not isinstance(record, dict):
                summaries.append({"evidence_id": evidence_id, "status": "failed", "error": "Invalid evidence record."})
                continue
            try:
                record_case_id = record.get("case_id")
                if record_case_id and record_case_id != case_id:
                    raise ValueError("Evidence record belongs to a different case.")
                storage_path = record.get("storage_path")
                if not storage_path:
                    raise ValueError("Evidence record has no storage_path.")
                path = _safe_path(storage_path)
                if not path.is_file():
                    raise FileNotFoundError("Local evidence file was not found.")
                expected = record.get("sha256")
                if not isinstance(expected, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected):
                    raise ValueError("Evidence record has no valid SHA-256.")
                with _timed_stage("sha256_verification", case_id=case_id, evidence_id=evidence_id):
                    actual = _sha256_file(path)
                if actual.lower() != expected.lower():
                    raise ValueError("SHA-256 mismatch. Evidence was not indexed.")

                filename = record.get("original_filename") or path.name
                extracted_text, observations = _cached_extraction(path, actual, filename)
                with _timed_stage("chunking", case_id=case_id, evidence_id=evidence_id):
                    chunks = _chunks(extracted_text)
                vectors = _cached_embeddings(embedder, actual, chunks)
                modality = record.get("modality", path.suffix.lower().lstrip("."))
                evidence_manifest.append({
                    "evidence_id": str(evidence_id), "sha256": actual.lower(),
                    "storage_path": str(storage_path), "modality": str(modality),
                    "extraction_pipeline": EXTRACTION_PIPELINE_VERSION,
                })

                graph_chunks = []
                for index, chunk in enumerate(chunks):
                    chunk_id = f"{evidence_id}-C{index + 1:04d}"
                    item = {
                        "case_id": case_id, "evidence_id": str(evidence_id),
                        "chunk_id": chunk_id, "filename": filename,
                        "modality": modality, "chunk_index": index,
                        "text": chunk, "sha256": actual.lower(),
                        "source_metadata": observations if index == 0 else [],
                        "frame_metadata": observations if index == 0 else [],
                        "observation_schema_version": 1,
                        "indexed_at": datetime.now(timezone.utc).isoformat(),
                    }
                    metadata.append(item)
                    all_vectors.append(vectors[index].tolist())
                    graph_chunks.append({
                        "chunk_id": chunk_id, "filename": filename,
                        "modality": modality, "text": chunk,
                        "sha256": actual.lower(), "entities": _entities(chunk),
                    })

                graph_enabled = os.getenv("SHERLOCK_ENABLE_NEO4J", "true").strip().lower() in {"1", "true", "yes", "on"}
                if graph_enabled:
                    from backend.services.graph_rag import upsert_evidence_chunks
                    with _timed_stage("neo4j_evidence_write", case_id=case_id, evidence_id=evidence_id, chunks=len(graph_chunks)):
                        graph_written = upsert_evidence_chunks(case_id, str(evidence_id), graph_chunks)
                    # graph_rag swallows Neo4j outages, so the boolean result, not an
                    # exception, is what tells us the graph write really happened.
                    record["graph_status"] = "written" if graph_written else "failed"
                    if not graph_written:
                        warnings.append({"evidence_id": str(evidence_id), "component": "neo4j", "error": "graph_write_failed"})
                else:
                    record["graph_status"] = "disabled"

                record["analysis_status"] = "indexed"
                record.pop("analysis_error", None)
                record["analysis"] = {
                    "extraction_method": EXTRACTION_PIPELINE_VERSION,
                    "extracted_character_count": len(extracted_text),
                    "chunk_count": len(chunks), "sha256_verified": True,
                    "observation_count": len(observations),
                    "observation_schema_version": 1,
                    "camera_clock_offset_seconds": CAMERA_CLOCK_OFFSET_SECONDS,
                    "vision_model": VISION_MODEL_NAME,
                    "embedding_model": EMBED_MODEL,
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                }
                summaries.append({"evidence_id": str(evidence_id), "status": "indexed", "chunks": len(chunks), "modality": modality, "sha256_verified": True, "cache": "content-addressed"})
                print("[INDEX_TIMING] " + json.dumps({"stage": "evidence_total", "case_id": case_id, "evidence_id": str(evidence_id), "filename": filename, "seconds": round(time.perf_counter() - evidence_started, 3), "status": "ok"}, sort_keys=True), flush=True)
            except Exception as exc:
                record["analysis_status"] = "failed"
                record["analysis_error"] = f"{type(exc).__name__}: {exc}"
                summaries.append({"evidence_id": str(evidence_id), "status": "failed", "error": record["analysis_error"]})
                print("[INDEX_TIMING] " + json.dumps({"stage": "evidence_total", "case_id": case_id, "evidence_id": str(evidence_id), "seconds": round(time.perf_counter() - evidence_started, 3), "status": "error", "error_type": type(exc).__name__}, sort_keys=True), flush=True)

        failed = [item for item in summaries if item.get("status") == "failed"]
        if failed or not all_vectors:
            result = {
                "case_id": case_id, "published": False,
                "preserved_previous_generation": (_case_dir(case_id) / "active.json").is_file(),
                "indexed_chunks": len(metadata), "indexed_evidence": len(summaries) - len(failed),
                "failed_evidence": len(failed), "evidence": summaries, "warnings": warnings,
                "message": "Index generation was not published because one or more exhibits failed.",
            }
            print("[INDEX_TIMING] " + json.dumps({"stage": "case_ingestion_total", "case_id": case_id, "seconds": round(time.perf_counter() - ingest_started, 3), "published": False}, sort_keys=True), flush=True)
            return result

        evidence_manifest.sort(key=lambda item: item["evidence_id"])
        manifest = {
            "case_id": case_id,
            "evidence": evidence_manifest,
            "evidence_manifest_sha256": hashlib.sha256(json.dumps(evidence_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "embedding_model": EMBED_MODEL,
            "embedding_pipeline_version": EMBEDDING_PIPELINE_VERSION,
            "extraction_pipeline_version": EXTRACTION_PIPELINE_VERSION,
            "chunking_pipeline_version": CHUNKING_PIPELINE_VERSION,
            "neo4j_enabled": os.getenv("SHERLOCK_ENABLE_NEO4J", "true").strip().lower() in {"1", "true", "yes", "on"},
        }
        with _timed_stage("generation_publication", case_id=case_id, chunks=len(metadata)):
            published = _publish_generation(case_id, all_vectors, metadata, manifest)
        graph_status = "disabled"
        if os.getenv("SHERLOCK_ENABLE_NEO4J", "true").strip().lower() in {"1", "true", "yes", "on"}:
            graph_write_failed = any(item.get("component") == "neo4j" for item in warnings)
            if graph_write_failed:
                graph_status = "degraded"
            else:
                from backend.services.graph_rag import set_case_generation
                with _timed_stage("neo4j_generation_commit", case_id=case_id):
                    generation_committed = set_case_generation(case_id, published["generation"], manifest["evidence_manifest_sha256"])
                if generation_committed:
                    graph_status = "ready"
                else:
                    graph_status = "degraded"
                    warnings.append({"component": "neo4j_generation", "error": "generation_not_persisted"})
        result = {
            "case_id": case_id, "published": True,
            "graph_status": graph_status,
            "generation": published["generation"],
            "indexed_chunks": len(metadata), "indexed_evidence": len(summaries),
            "failed_evidence": 0, "evidence": summaries, "warnings": warnings,
            "index_path": published["directory"],
            "manifest": published["manifest"],
            "vision_model": VISION_MODEL_NAME,
            "reasoning_model": REASONING_MODEL_NAME,
            "embedding_model": EMBED_MODEL,
            "camera_clock_offset_seconds": CAMERA_CLOCK_OFFSET_SECONDS,
        }
        print("[INDEX_TIMING] " + json.dumps({"stage": "case_ingestion_total", "case_id": case_id, "seconds": round(time.perf_counter() - ingest_started, 3), "published": True, "chunks": len(metadata), "evidence": len(summaries)}, sort_keys=True), flush=True)
        return result


# =========================================================
# HELPER FUNCTIONS
# =========================================================


def _build_evidence_baseline(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """Summarize evidence and identify traceable candidate correlations."""

    import re
    from itertools import combinations

    sections = [
        "Evidence-grounded investigation summary",
        "",
        f"Question: {question.strip()}",
        "",
    ]

    grouped = {}
    unavailable_visual_sources = []

    stop_words = {
        "the", "and", "that", "this", "with", "from", "were",
        "was", "are", "for", "has", "have", "had", "not",
        "into", "before", "after", "then", "there", "their",
        "case", "evidence", "item", "record", "source",
        "image", "video", "text", "exhibit",
    }

    # Identifiers such as CASE-2026-071 and ATM-CHN-0457.
    identifier_pattern = re.compile(
        r"\b(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)"
        r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+){1,}\b",
        re.IGNORECASE,
    )

    # Dates and times are candidate anchors, not automatically
    # comparable clock times. Video-relative times need special care.
    date_pattern = re.compile(
        r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}"
        r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"
    )
    time_pattern = re.compile(
        r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\b",
        re.IGNORECASE,
    )

    def normalize(value):
        return re.sub(r"\s+", " ", value).strip().casefold()

    def chunk_number(chunk):
        match = re.search(r"-C(\d+)$", chunk["chunk_id"])
        return int(match.group(1)) if match else 0

    def anchors(text):
        identifiers = {
            normalize(value)
            for value in identifier_pattern.findall(text)
        }
        dates = {
            normalize(value)
            for value in date_pattern.findall(text)
        }
        times = {
            normalize(value)
            for value in time_pattern.findall(text)
        }
        return identifiers, dates, times

    def content_terms(text):
        terms = set(re.findall(r"\b[a-z0-9]{4,}\b", text.casefold()))
        return terms - stop_words

    for item in retrieved_chunks:
        filename = item["filename"]
        modality = item.get("modality", "unknown")
        evidence_text = item["text"].strip()

        if filename not in grouped:
            grouped[filename] = {
                "evidence_id": item["evidence_id"],
                "modality": modality,
                "chunks": [],
            }

        grouped[filename]["chunks"].append({
            "chunk_id": item["chunk_id"],
            "text": evidence_text,
        })

        if modality in {"image", "video"} and re.search(
            r"VISUAL ANALYSIS UNAVAILABLE|ReadTimeout|"
            r"Visual observations:\s*\[Visual analysis unavailable",
            evidence_text,
            re.IGNORECASE,
        ):
            if filename not in unavailable_visual_sources:
                unavailable_visual_sources.append(filename)

    # Keep the evidence excerpts, with source provenance.
    for filename, record in grouped.items():
        sections.extend([
            f"## Source: {filename}",
            f"Evidence ID: {record['evidence_id']}",
            f"Modality: {record['modality']}",
        ])

        seen_lines = set()
        findings = []

        for chunk in sorted(record["chunks"], key=chunk_number):
            for raw_line in chunk["text"].splitlines():
                line = re.sub(r"\s+", " ", raw_line).strip()

                if not line:
                    continue

                if re.search(
                    r"ReadTimeout|HTTPConnectionPool|"
                    r"IMAGE VISUAL ANALYSIS UNAVAILABLE|"
                    r"VIDEO VISUAL ANALYSIS UNAVAILABLE",
                    line,
                    re.IGNORECASE,
                ):
                    continue

                if line.upper() == "IMAGE OCR TEXT:":
                    continue

                if re.match(
                    r"^(?:tripe fallback;|anti-skimming module\)|\d+s\])",
                    line,
                    re.IGNORECASE,
                ):
                    continue

                if len(line) < 5 and not line.isupper():
                    continue

                normalized = normalize(line)
                if normalized in seen_lines:
                    continue

                seen_lines.add(normalized)
                findings.append((chunk["chunk_id"], line))

        if not findings:
            sections.append(
                "- No usable text findings in the retrieved chunks."
            )
        else:
            for chunk_id, line in findings[:12]:
                sections.append(f"- {line} [Chunk: {chunk_id}]")

        sections.append("")

    # Build one record per evidence item, not per chunk.
    evidence_items = []
    for filename, record in grouped.items():
        combined_text = "\n".join(
            chunk["text"] for chunk in record["chunks"]
        )
        ids, dates, times = anchors(combined_text)

        evidence_items.append({
            "filename": filename,
            "evidence_id": record["evidence_id"],
            "modality": record["modality"],
            "text": combined_text,
            "ids": ids,
            "dates": dates,
            "times": times,
            "terms": content_terms(combined_text),
            "chunks": record["chunks"],
        })

    sections.extend([
        "## Candidate cross-evidence correlations",
        "These are leads for review, not verified conclusions.",
        "",
    ])

    candidates = []

    for first, second in combinations(evidence_items, 2):
        shared_ids = sorted(first["ids"] & second["ids"])
        shared_dates = sorted(first["dates"] & second["dates"])
        shared_times = sorted(first["times"] & second["times"])

        # A shared timestamp alone is weak; report it separately
        # rather than treating it as proof of an event match.
        shared_terms = sorted(first["terms"] & second["terms"])

        reasons = []
        strength = None

        if shared_ids:
            reasons.append(
                "Exact identifier match: " + ", ".join(shared_ids)
            )
            strength = "identifier_match"

        if shared_dates:
            reasons.append(
                "Exact date string match: " + ", ".join(shared_dates)
            )
            if strength is None:
                strength = "date_match"

        if shared_times:
            reasons.append(
                "Exact time string match: " + ", ".join(shared_times)
            )
            if strength is None:
                strength = "time_match"

        # Lexical overlap is only a weak lead, and only included
        # when there is no stronger shared anchor.
        if not reasons and len(shared_terms) >= 4:
            reasons.append(
                "Shared content terms: " + ", ".join(shared_terms[:8])
            )
            strength = "lexical_overlap"

        if not reasons:
            continue

        candidates.append({
            "first": first,
            "second": second,
            "reasons": reasons,
            "strength": strength,
        })

    if not candidates:
        sections.append(
            "- No candidate links were detected among the retrieved "
            "evidence items. This does not establish that no links exist."
        )
    else:
        for number, candidate in enumerate(candidates, start=1):
            first = candidate["first"]
            second = candidate["second"]

            sections.extend([
                f"### Candidate link {number}",
                (
                    f"- Source A: {first['filename']} "
                    f"({first['evidence_id']}; {first['modality']})"
                ),
                (
                    f"- Source B: {second['filename']} "
                    f"({second['evidence_id']}; {second['modality']})"
                ),
                f"- Match type: {candidate['strength']}",
            ])

            for reason in candidate["reasons"]:
                sections.append(f"- Basis: {reason}")

            sections.append(
                "- Assessment: Review the original evidence and relevant "
                "chunks before treating this as a meaningful relationship."
            )
            sections.append(
                "- Limitation: Matching text does not prove common origin, "
                "event timing, identity, causation, or guilt."
            )
            sections.append("")

    sections.extend(["## Evidence gaps"])

    if unavailable_visual_sources:
        sections.append(
            "- Visual analysis was reported unavailable for: "
            + ", ".join(unavailable_visual_sources)
            + ". Do not treat OCR text as a verified visual observation."
        )
    else:
        sections.append(
            "- No visual-analysis failure marker was detected in the "
            "retrieved image/video chunks. This does not verify visual "
            "analysis quality."
        )

    sections.extend([
        "- Only retrieved chunks were compared; relevant evidence may "
        "have been missed by retrieval.",
        "- Matching dates or times does not establish synchronized clocks.",
        "- OCR and generated visual descriptions may contain errors.",
        "- No person's identity, intent, or guilt should be inferred "
        "from these records alone.",
    ])

    return "\n".join(sections)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _question_named_evidence(
    question: str,
    metadata: list[dict],
) -> set[str]:
    """Find evidence IDs explicitly mentioned in the question."""

    question_lower = question.casefold()
    matched = set()

    for item in metadata:
        evidence_id = str(item.get("evidence_id", "")).strip()
        filename = str(item.get("filename", "")).strip()
        stem = Path(filename).stem

        aliases = {evidence_id, stem}

        # Recognize IDs such as V1 in filenames like V1_gate_camera.mp4.
        prefix_match = re.match(
            r"^([A-Za-z]{1,4}\d{1,4})(?:\b|[_-])",
            stem,
        )

        if prefix_match:
            aliases.add(prefix_match.group(1))

        for alias in aliases:
            if not alias:
                continue

            pattern = (
                r"(?<![A-Za-z0-9])"
                + re.escape(alias.casefold())
                + r"(?![A-Za-z0-9])"
            )

            if re.search(pattern, question_lower):
                if evidence_id:
                    matched.add(evidence_id)
                break

    # Resolve ordinary-language source references. Filename terms are the
    # strongest clue; generic words such as "photo" are only used when the
    # case contains a single image, so one word cannot accidentally select
    # every photograph in a case.
    video_items = []
    image_items = []
    for item in metadata:
        evidence_id = str(item.get("evidence_id", "")).strip()
        filename_value = str(item.get("filename", "")).strip()
        filename = filename_value.casefold()
        modality = str(item.get("modality", "")).casefold()
        suffix = Path(filename).suffix
        is_video = suffix in {".mp4", ".mov", ".avi", ".mkv"} or any(
            token in modality for token in ("video", "cctv")
        )
        is_image = suffix in {".jpg", ".jpeg", ".png", ".webp"} or any(
            token in modality for token in ("image", "photo", "picture")
        )
        if evidence_id and is_video:
            video_items.append((evidence_id, filename_value))
        if evidence_id and is_image:
            image_items.append((evidence_id, filename_value))

        stem = Path(filename_value).stem.casefold()
        terms = [
            token for token in re.findall(r"[a-z0-9]{3,}", stem)
            if token not in {
                "image", "photo", "photograph", "picture", "video",
                "file", "evidence", "camera", "frame", "cctv",
            }
        ]
        if evidence_id and any(
            re.search(r"\b" + re.escape(token) + r"\b", question_lower)
            for token in terms
        ):
            matched.add(evidence_id)

    if re.search(r"\b(cctv|footage|video|camera recording)\b", question_lower):
        if len(video_items) == 1:
            matched.add(video_items[0][0])
        elif len(video_items) > 1:
            # If the question says CCTV, prefer files whose name includes CCTV;
            # otherwise leave ambiguous sources unresolved rather than guessing.
            named_videos = [
                evidence_id for evidence_id, filename in video_items
                if "cctv" in filename.casefold()
            ]
            if len(named_videos) == 1:
                matched.add(named_videos[0])

    if re.search(r"\b(photo|photograph|picture|image|lock photograph)\b", question_lower):
        if len(image_items) == 1:
            matched.add(image_items[0][0])
        elif len(image_items) > 1:
            named_images = [
                evidence_id for evidence_id, filename in image_items
                if any(
                    token in filename.casefold()
                    for token in re.findall(r"[a-z0-9]{3,}", question_lower)
                    if token not in {"what", "does", "show", "from", "this", "that", "photo", "photograph", "picture", "image", "verify", "verified", "evidence"}
                )
            ]
            matched.update(named_images)
    return matched

# =========================================================
# HELPER FUNCTIONS
# =========================================================



# =========================================================
# 14. CASE-SPECIFIC EVIDENCE RETRIEVAL AND ANSWERING
# =========================================================


def _is_cross_evidence_question(question: str) -> bool:
    """Use one detector for full-index search and balanced exhibit selection."""
    return bool(re.search(
        r"\b(compare|comparison|correlat\w*|cross[- ]evidence|across evidence|"
        r"both sources|consistent with|contradict\w*|match(?:es|ed)?|"
        r"relat\w*|timeline|connect\w*|link\w*|"
        r"same|both|identical|similar|differ\w*)\b",
        str(question or ""),
        re.IGNORECASE,
    ))


def answer_case(
    case_id: str,
    question: str,
    top_k: int = 6,
) -> dict:
    """Answer a question using retrieved, case-specific evidence."""

    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("Case ID cannot be empty.")

    if not isinstance(question, str) or len(question.strip()) < 3:
        raise ValueError("Question must contain at least 3 characters.")

    if len(question) > 2000:
        raise ValueError("Question cannot exceed 2000 characters.")

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be an integer.")

    if not 1 <= top_k <= 12:
        raise ValueError("top_k must be between 1 and 12.")

    case_id = case_id.strip()
    question = question.strip()

    # Resolve only this case's index. Never silently rebuild it.
    folder, active_pointer = _active_generation(case_id)
    index_path = folder / "evidence.faiss"
    metadata_path = folder / "chunks.json"

    if not index_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"No complete evidence index exists for case {case_id}. "
            "Index the case's evidence before asking questions."
        )
    if active_pointer is not None:
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Active index generation has no manifest.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("case_id") != case_id or manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise ValueError("Active index manifest is incompatible or belongs to another case.")
        if (
            manifest.get("extraction_pipeline_version") != EXTRACTION_PIPELINE_VERSION
            or manifest.get("chunking_pipeline_version") != CHUNKING_PIPELINE_VERSION
        ):
            raise ValueError(
                "This case's index was built with an older extraction/chunking "
                "pipeline. Re-index the case's evidence before asking questions so "
                "answers use the current, cleaned extraction."
            )

    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "The evidence index metadata is unreadable. "
            "Check the files before considering re-indexing."
        ) from exc

    if not isinstance(metadata, list):
        raise ValueError(
            "The evidence index metadata must be a list."
        )

    if not metadata:
        raise ValueError(
            "The evidence index contains no metadata chunks."
        )

    if any(not isinstance(item, dict) for item in metadata):
        raise ValueError(
            "The evidence index contains invalid metadata records."
        )

    required_fields = (
        "evidence_id",
        "chunk_id",
        "filename",
        "text",
    )

    for item in metadata:
        if any(
            not isinstance(item.get(field), str)
            or not item[field].strip()
            for field in required_fields
        ):
            raise ValueError(
                "An indexed chunk is missing required source metadata."
            )

        if item.get("case_id") != case_id:
            raise ValueError(
                "Index metadata contains records from another case. "
                "Refusing cross-case retrieval."
            )

    try:
        import faiss
        import numpy as np
        index = faiss.read_index(str(index_path))
    except Exception as exc:
        raise ValueError(
            "The FAISS index could not be loaded. "
            "Check the index file before re-indexing."
        ) from exc

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS vector count does not match chunks.json metadata. "
            "Refusing to return potentially incorrect source references."
        )

    embedder = _embedder()

    if index.ntotal == 0 or index.d != (
        embedder.get_sentence_embedding_dimension()
    ):
        raise ValueError(
            "The FAISS index is empty or its embedding dimension "
            "does not match the configured embedding model."
        )

    # Embed the question using the existing configured embedder.
    query_vector = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_vector = np.asarray(query_vector, dtype="float32")

    if (
        query_vector.shape != (1, index.d)
        or not np.isfinite(query_vector).all()
    ):
        raise ValueError("The question embedding is invalid.")

    def modality_group(value):
        value = str(value or "unknown").strip().lower()

        if any(
            term in value
            for term in ("video", "cctv", ".mp4", ".mov", ".avi", ".mkv")
        ):
            return "video"

        if any(
            term in value
            for term in (
                "image", "photo", "visual", "picture",
                ".jpg", ".jpeg", ".png", ".webp",
            )
        ):
            return "image"

        if any(
            term in value
            for term in ("text", "document", "pdf", "csv")
        ):
            return "text"

        return value or "unknown"

    # Retrieve more candidates than we will ultimately return. For cross-evidence
    # questions, search the whole case index first: otherwise FAISS can exclude
    # image/video descriptions before the balancing step has a chance to select them.
    asks_for_comparison = _is_cross_evidence_question(question)
    candidate_count = index.ntotal if asks_for_comparison else min(
        max(top_k * 8, 24),
        index.ntotal,
    )

    scores, indices = index.search(
        query_vector,
        candidate_count,
    )

    candidates = []
    seen_positions = set()

    for score, position in zip(scores[0], indices[0]):
        position = int(position)
        similarity = float(score)

        if position < 0 or position >= len(metadata):
            continue

        if position in seen_positions:
            continue

        if not np.isfinite(similarity):
            continue

        item = metadata[position]

        if not item["text"].strip():
            continue

        seen_positions.add(position)

        candidates.append({
            "position": position,
            "similarity": similarity,
            "modality_group": modality_group(
                item.get("modality", "unknown")
            ),
        })

    # Identify source IDs explicitly mentioned by the user.
    named_evidence = _question_named_evidence(
        question,
        metadata,
    )

    # If a question names a source, make sure its chunks can be
    # considered even if FAISS did not include them in top results.
    for position, item in enumerate(metadata):
        if item["evidence_id"] not in named_evidence:
            continue

        if position in seen_positions:
            continue

        try:
            vector = np.asarray(
                index.reconstruct(position),
                dtype="float32",
            )

            if (
                vector.shape != (index.d,)
                or not np.isfinite(vector).all()
            ):
                similarity = -1.0
            else:
                similarity = float(
                    np.dot(query_vector[0], vector)
                )
        except Exception:
            # Keep the explicitly requested source available even if
            # this FAISS index type does not support reconstruction.
            similarity = -1.0

        candidates.append({
            "position": position,
            "similarity": similarity,
            "modality_group": modality_group(
                item.get("modality", "unknown")
            ),
        })

        seen_positions.add(position)

    # For explicit multi-evidence questions, include candidates from every
    # named evidence ID, even when semantic search favored one modality.
    # Already done above via exact-index reconstruction.

    # Detect questions that genuinely request a comparison.
    cross_evidence_question = _is_cross_evidence_question(question)

    # Single-source questions should not be answered using unrelated
    # documents. Comparison questions may use multiple sources.
    cross_evidence_question = cross_evidence_question or len(named_evidence) >= 2

    if len(named_evidence) == 1 and not cross_evidence_question:
        candidates = [
            candidate
            for candidate in candidates
            if metadata[candidate["position"]]["evidence_id"]
            in named_evidence
        ]

    def is_visual_question(value):
        return bool(
            re.search(
                r"\b("
                r"visible|visually|see|shown|depict|"
                r"images?|photos?|photographs?|videos?|frames?|"
                r"rider|clothing|motorcycle|vehicle|"
                r"number plate|license plate|"
                r"colour|color|appearance|describe|"
                r"what does .* look like"
                r")\b",
                value,
                re.IGNORECASE,
            )
        )

    visual_question = is_visual_question(question)

    def candidate_priority(candidate):
        item = metadata[candidate["position"]]
        evidence_id = item["evidence_id"]
        modality = str(item.get("modality", "")).lower()

        named_priority = int(
            bool(named_evidence)
            and evidence_id in named_evidence
        )

        visual_priority = int(
            visual_question
            and modality_group(modality) in {"image", "video"}
        )

        # Similarity is the final tie-breaker, not the only criterion.
        return (
            named_priority,
            visual_priority,
            candidate["similarity"],
        )

    candidates.sort(
        key=candidate_priority,
        reverse=True,
    )


    # IMPORTANT: Do not restrict a cross-evidence comparison to only the
    # sources recognized by _question_named_evidence(). That helper can resolve
    # "incident report" and "access log" from filenames while failing to resolve
    # "ATM surveillance images" when the case contains multiple images. Filtering
    # here silently removes the image candidates and makes the answer document-only.
    # Keep all case candidates for comparisons; named sources already receive
    # priority above, and round-robin selection preserves coverage across exhibits.
    if cross_evidence_question:
        # For comparison questions, select chunks round-robin by
        # evidence ID so one exhibit cannot dominate retrieval.
        candidates_by_evidence = {}

        for candidate in candidates:
            item = metadata[candidate["position"]]
            evidence_id = item["evidence_id"]

            candidates_by_evidence.setdefault(
                evidence_id, []
            ).append(candidate)

        selected = []
        selected_positions = set()

        while len(selected) < top_k:
            added_this_round = False

            for group in candidates_by_evidence.values():
                while (
                    group
                    and group[0]["position"] in selected_positions
                ):
                    group.pop(0)

                if group and len(selected) < top_k:
                    candidate = group.pop(0)
                    selected.append(candidate)
                    selected_positions.add(candidate["position"])
                    added_this_round = True

            if not added_this_round:
                break
    else:
        selected = candidates[:top_k]


    sources = []
    retrieved_chunks = []

    for candidate in selected:
        item = metadata[candidate["position"]]

        source = {
            "filename": item["filename"],
            "evidence_id": item["evidence_id"],
            "chunk_id": item["chunk_id"],
            "similarity": candidate["similarity"],
            "modality": item.get("modality", "unknown"),
            "sha256": item.get("sha256"),
        }

        chunk = {
            "evidence_id": item["evidence_id"],
            "filename": item["filename"],
            "chunk_id": item["chunk_id"],
            "modality": item.get("modality", "unknown"),
            "sha256": item.get("sha256"),
            "text": item["text"].strip(),
            "similarity": candidate["similarity"],
            # Per-observation extractor metadata (OCR confidence, camera clock
            # offset) is forwarded so the answer engine can surface the caveats
            # that decide how much weight a matched timestamp/identifier deserves.
            "source_metadata": item.get("source_metadata") or [],
            "frame_metadata": item.get("frame_metadata") or [],
        }

        sources.append(source)
        retrieved_chunks.append(chunk)

    if not retrieved_chunks:
        if named_evidence:
            raise ValueError(
                "The explicitly requested evidence was not found "
                "in the usable index. Check the evidence ID and "
                "index metadata. No unrelated source was substituted."
            )

        raise ValueError(
            "No usable evidence chunks were retrieved for this question."
        )

    # Retrieve graph-linked chunks at answer time. Graph results are supplemental
    # and remain explicitly identified; they never override source provenance.
    graph_context = {"status": "disabled", "relationships": [], "chunks": []}
    if os.getenv("SHERLOCK_ENABLE_NEO4J", "true").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            from backend.services.graph_rag import retrieve_graph_context
            graph_context = retrieve_graph_context(case_id, question, limit=min(12, top_k * 2), expected_generation=(active_pointer or {}).get("generation"))
            indexed_by_chunk = {item["chunk_id"]: item for item in metadata}
            existing = {item["chunk_id"] for item in retrieved_chunks}
            for graph_item in graph_context.get("chunks", []):
                chunk_id = graph_item.get("chunk_id")
                if chunk_id in existing:
                    continue
                # The local FAISS generation is authoritative. A graph chunk is
                # only usable when the same chunk still exists in this case's
                # published index, and its indexed text/metadata is what gets
                # returned so FAISS and Neo4j can never disagree about a source.
                indexed = indexed_by_chunk.get(chunk_id)
                if indexed is None:
                    continue
                retrieved_chunks.append({
                    "evidence_id": indexed["evidence_id"],
                    "filename": indexed["filename"],
                    "chunk_id": indexed["chunk_id"],
                    "modality": indexed.get("modality", "unknown"),
                    "sha256": indexed.get("sha256"),
                    "text": indexed["text"].strip(),
                    "retrieval_source": "neo4j_graph",
                    "source_metadata": indexed.get("source_metadata") or [],
                    "frame_metadata": indexed.get("frame_metadata") or [],
                })
                sources.append({
                    "filename": indexed["filename"],
                    "evidence_id": indexed["evidence_id"],
                    "chunk_id": indexed["chunk_id"],
                    "similarity": None,
                    "retrieval_source": "neo4j_graph",
                    "modality": indexed.get("modality", "unknown"),
                    "sha256": indexed.get("sha256"),
                })
                existing.add(chunk_id)
            # Report only graph relationships whose chunk is still in the index.
            graph_context = {
                **graph_context,
                "chunks": [c for c in graph_context.get("chunks", []) if c.get("chunk_id") in indexed_by_chunk],
                "relationships": [r for r in graph_context.get("relationships", []) if r.get("chunk_id") in indexed_by_chunk],
            }
        except Exception as exc:
            graph_context = {"status": "unavailable", "error": type(exc).__name__, "relationships": [], "chunks": []}

    # Deterministic answer generation consumes FAISS evidence plus graph-linked
    # source chunks; every returned chunk retains its exhibit/chunk provenance.
    # Phase 6: deterministic claim-level synthesis via answer_engine.generate_answer.
    from backend.services.answer_engine import generate_answer
    answer_result = generate_answer(question=question, chunks=retrieved_chunks)
    answer = answer_result["answer"]
    answer_model = "deterministic_evidence_rules"

    evidence_coverage = {}
    for source in sources:
        evidence_id = source["evidence_id"]
        evidence_coverage.setdefault(evidence_id, {
            "evidence_id": evidence_id,
            "filename": source["filename"],
            "modality": source["modality"],
            "chunk_count": 0,
            "best_similarity": source["similarity"],
        })
        evidence_coverage[evidence_id]["chunk_count"] += 1
        score = source.get("similarity")
        current_score = evidence_coverage[evidence_id].get("best_similarity")
        if isinstance(score, (int, float)):
            evidence_coverage[evidence_id]["best_similarity"] = score if current_score is None else max(current_score, score)

    retrieved_evidence_ids = set(evidence_coverage)
    missing_named_evidence = sorted(named_evidence - retrieved_evidence_ids)
    return {
        "case_id": case_id,
        "question": question,
        "answer": answer,
        "answer_status": answer_result.get("answer_status"),
        "claims": answer_result.get("claims", []),
        "all_claims": answer_result.get("all_claims", answer_result.get("claims", [])),
        "correlations": answer_result.get("correlations", []),
        "contradictions": answer_result.get("contradictions", []),
        "identifier_conflicts": answer_result.get("identifier_conflicts", []),
        "caveats": answer_result.get("caveats", []),
        "provenance_validation": answer_result.get("provenance_validation", {}),
        "sources": sources,
        "retrieved_chunks": len(sources),
        "retrieved_evidence_count": len(evidence_coverage),
        "evidence_coverage": list(evidence_coverage.values()),
        "retrieval_mode": "hybrid_faiss_neo4j" if graph_context.get("status") == "ready" and graph_context.get("chunks") else ("cross_evidence_balanced" if cross_evidence_question else "semantic_ranked"),
        "graph_status": graph_context.get("status", "disabled"),
        "graph_relationships": graph_context.get("relationships", []),
        "graph_retrieved_chunks": len(graph_context.get("chunks", [])),
        "cross_evidence_question": cross_evidence_question,
        "missing_named_evidence": missing_named_evidence,
        "reasoning_model": answer_model,
        "answer_generation_method": "deterministic_evidence_rules",
        "embedding_model": EMBED_MODEL,
        "answer_type": (
            "question_answer"
            if answer_model == "deterministic_evidence_rules"
            else "source_excerpt_fallback"
        ),
    }
