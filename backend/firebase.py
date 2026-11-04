"""Firebase Admin SDK lifecycle and database access."""
from threading import RLock

import firebase_admin
from firebase_admin import credentials, db

from backend.config import FIREBASE_DATABASE_URL, SERVICE_ACCOUNT_PATH, FIREBASE_STORAGE_BUCKET

_init_lock = RLock()


def initialize_firebase():
    """Initialize the default Firebase app once, safely across threads."""
    try:
        return firebase_admin.get_app()
    except ValueError:
        pass

    with _init_lock:
        try:
            return firebase_admin.get_app()
        except ValueError:
            pass

        if not FIREBASE_DATABASE_URL:
            raise RuntimeError("Missing environment setting: FIREBASE_DATABASE_URL")
        if SERVICE_ACCOUNT_PATH is None:
            raise RuntimeError("Missing environment setting: FIREBASE_SERVICE_ACCOUNT")
        if not SERVICE_ACCOUNT_PATH.is_file():
            raise RuntimeError(f"Firebase service account not found: {SERVICE_ACCOUNT_PATH}")

        credential = credentials.Certificate(str(SERVICE_ACCOUNT_PATH))
        options = {"databaseURL": FIREBASE_DATABASE_URL}
        if FIREBASE_STORAGE_BUCKET:
            options["storageBucket"] = FIREBASE_STORAGE_BUCKET
        return firebase_admin.initialize_app(credential, options)


def database_ref(path: str):
    """Return a Realtime Database reference after ensuring SDK initialization."""
    initialize_firebase()
    return db.reference(path)
