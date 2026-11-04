"""FastAPI application entry point."""
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import ALLOWED_ORIGINS
from backend.firebase import initialize_firebase
from backend.routes import cases, evidence, analysis

logger = logging.getLogger(__name__)


def _warm_answer_path() -> None:
    """Load the ask-path models and native libraries once, off the request path.

    Loading the sentence embedder (and importing FAISS/NumPy) costs tens of
    seconds on CPU. Without warming, the first question after every boot pays
    that cost while the investigator waits. This runs in a background thread so
    server readiness is not delayed; failures are logged and never fatal because
    the ask path already loads these lazily on demand.
    """
    try:
        import faiss  # noqa: F401
        import numpy  # noqa: F401
        from backend.services.hybrid_rag import _embedder

        _embedder()
        logger.info("Answer-path warmup complete.")
    except Exception as exc:  # pragma: no cover - defensive, best-effort warmup
        logger.warning("Answer-path warmup skipped (%s: %s).", type(exc).__name__, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast on invalid deployment configuration. Liveness/readiness endpoints
    # remain separately defined for orchestration once the app is running.
    initialize_firebase()
    app.state.firebase_ready = True
    threading.Thread(target=_warm_answer_path, name="answer-path-warmup", daemon=True).start()
    try:
        yield
    finally:
        app.state.firebase_ready = False


app = FastAPI(
    title="Mr. Sherlock API",
    description="Multimodal forensic investigation platform",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(cases.router, prefix="/api")
app.include_router(evidence.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")


@app.get("/health/live", tags=["Health"])
def liveness_check():
    return {"status": "alive", "application": "Mr. Sherlock"}


@app.get("/health/ready", tags=["Health"])
def readiness_check():
    try:
        initialize_firebase()
        return {"status": "ready", "dependencies": {"firebase": "available"}}
    except Exception as exc:
        logger.exception("Readiness check failed (%s)", type(exc).__name__)
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Required services are unavailable.") from None


@app.get("/health", tags=["Health"])
def health_check():
    """Backward-compatible liveness endpoint."""
    return liveness_check()
