"""Safe local evidence-object storage and content validation.

The local backend expects SHERLOCK_EVIDENCE_DIR to point at persistent storage
(not a temporary deployment filesystem). Object paths are relative and are
resolved beneath that root only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import BinaryIO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_ROOT = Path(os.getenv("SHERLOCK_EVIDENCE_DIR", str(PROJECT_ROOT / "evidence_storage"))).resolve()
CHUNK_SIZE = 1024 * 1024

ALLOWED_SIGNATURES = {
    ".pdf": lambda b: b.startswith(b"%PDF-"),
    ".png": lambda b: b.startswith(b"\x89PNG\r\n\x1a\n"),
    ".jpg": lambda b: b.startswith(b"\xff\xd8\xff"),
    ".jpeg": lambda b: b.startswith(b"\xff\xd8\xff"),
    ".webp": lambda b: len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP",
    ".mp4": lambda b: len(b) >= 12 and b[4:8] == b"ftyp",
    ".mov": lambda b: len(b) >= 12 and b[4:8] == b"ftyp",
    ".avi": lambda b: len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"AVI ",
    ".mkv": lambda b: b.startswith(b"\x1aE\xdf\xa3"),
    ".wav": lambda b: len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WAVE",
    ".mp3": lambda b: b.startswith(b"ID3") or (len(b) >= 2 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0),
}
TEXT_EXTENSIONS = {".txt", ".csv", ".json"}


def safe_object_path(relative_path: str, root: Path | None = None) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("Evidence storage path is missing.")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid evidence storage path.")
    storage_root = (root or STORAGE_ROOT).resolve()
    resolved = (storage_root / relative).resolve()
    if not resolved.is_relative_to(storage_root):
        raise ValueError("Evidence path escapes the storage root.")
    return resolved


def validate_content(prefix: bytes, extension: str, text_sample: bytes = b"") -> str:
    ext = extension.lower()
    if ext in ALLOWED_SIGNATURES:
        if not ALLOWED_SIGNATURES[ext](prefix):
            raise ValueError(f"File content does not match the {ext} extension.")
        return {
            ".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".webp": "image/webp", ".mp4": "video/mp4",
            ".mov": "video/quicktime", ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
            ".wav": "audio/wav", ".mp3": "audio/mpeg",
        }[ext]
    if ext not in TEXT_EXTENSIONS:
        raise ValueError("Unsupported evidence extension.")
    try:
        decoded = text_sample.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Text evidence must be valid UTF-8.") from exc
    if ext == ".json":
        try:
            json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise ValueError("JSON evidence is not valid JSON.") from exc
    return {".txt": "text/plain", ".csv": "text/csv", ".json": "application/json"}[ext]


def store_upload(source: BinaryIO, relative_path: str, extension: str, max_bytes: int, root: Path | None = None) -> dict:
    """Stream to a same-filesystem temp file, hash, validate, then atomically publish."""
    destination = safe_object_path(relative_path, root=root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    prefix = b""
    text_sample = bytearray()
    fd, temp_name = tempfile.mkstemp(prefix=".upload-", dir=str(destination.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as output:
            while True:
                block = source.read(CHUNK_SIZE)
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise OverflowError("upload_too_large")
                if len(prefix) < 64:
                    prefix += block[: 64 - len(prefix)]
                if len(text_sample) < 1024 * 1024:
                    text_sample.extend(block[: 1024 * 1024 - len(text_sample)])
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if total == 0:
            raise ValueError("The uploaded file is empty.")
        detected_type = validate_content(prefix, extension, bytes(text_sample))
        sha256 = digest.hexdigest()
        if destination.exists():
            raise FileExistsError("Evidence object already exists.")
        os.replace(temp_path, destination)
        return {"sha256": sha256, "size_bytes": total, "detected_content_type": detected_type}
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def delete_object(relative_path: str, root: Path | None = None) -> bool:
    path = safe_object_path(relative_path, root=root)
    if path.is_file():
        path.unlink()
        return True
    return False
