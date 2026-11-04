"""Firebase ID-token authentication dependencies."""
import logging

from fastapi import Header, HTTPException
from firebase_admin import auth

from backend.config import REQUIRE_VERIFIED_EMAIL
from backend.firebase import initialize_firebase

logger = logging.getLogger(__name__)


def _invalid_token_exceptions():
    # Resolve exception classes from the installed Firebase Admin SDK.
    names = ("InvalidIdTokenError", "ExpiredIdTokenError", "RevokedIdTokenError", "UserDisabledError")
    return tuple(getattr(auth, name) for name in names if isinstance(getattr(auth, name, None), type))


def get_current_user(authorization: str | None = Header(default=None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Please sign in to continue.")

    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="A valid Bearer token is required.")

    try:
        initialize_firebase()
        decoded_token = auth.verify_id_token(token.strip(), check_revoked=True)
    except _invalid_token_exceptions() + (ValueError,):
        raise HTTPException(status_code=401, detail="Your session is invalid or has expired.") from None
    except Exception as exc:
        # Firebase outages, credential failures and network errors are not bad credentials.
        logger.exception("Firebase token verification service failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Authentication service is temporarily unavailable.") from None

    uid = decoded_token.get("uid")
    if not isinstance(uid, str) or not uid.strip():
        raise HTTPException(status_code=401, detail="Your session is invalid or has expired.")

    email = decoded_token.get("email", "")
    if REQUIRE_VERIFIED_EMAIL and (not email or not decoded_token.get("email_verified", False)):
        raise HTTPException(status_code=403, detail="Please verify your email address before continuing.")

    return {"uid": uid, "email": email}
