"""Centralized, validated application configuration.

Only settings required by the selected runtime path are mandatory. Firebase
Storage is optional because this project currently stores evidence locally.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def required_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing environment setting: {name}")
    return value


def _bool_setting(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true or false")


# Firebase Web API key is used by client-side Firebase Auth flows, not Admin SDK.
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY", "").strip()
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL", "").strip()
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", "").strip() or None
_service_account = os.getenv("FIREBASE_SERVICE_ACCOUNT", "").strip()
SERVICE_ACCOUNT_PATH = (
    (ROOT_DIR / _service_account).resolve() if _service_account else None
)

try:
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
except ValueError as exc:
    raise RuntimeError("MAX_UPLOAD_MB must be a whole number") from exc
if not 1 <= MAX_UPLOAD_MB <= 2048:
    raise RuntimeError("MAX_UPLOAD_MB must be between 1 and 2048")

ALLOWED_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
    if origin.strip()
]
REQUIRE_VERIFIED_EMAIL = _bool_setting("REQUIRE_VERIFIED_EMAIL", False)
