from fastapi.testclient import TestClient

from backend.main import app
from backend.security import get_current_user


def test_protected_routes_reject_missing_authentication():
    app.dependency_overrides[get_current_user] = (
        lambda: (_ for _ in ()).throw(
            __import__("fastapi").HTTPException(
                status_code=401,
                detail="Please sign in to continue.",
            )
        )
    )

    try:
        client = TestClient(app)
        paths = [
            ("/api/cases", "GET"),
            ("/api/cases", "POST"),
            ("/api/cases/CASE-TEST123", "GET"),
            ("/api/cases/CASE-TEST123/evidence", "GET"),
            ("/api/cases/CASE-TEST123/evidence", "POST"),
            ("/api/cases/CASE-TEST123/analyze", "POST"),
            ("/api/cases/CASE-TEST123/ingest-index", "POST"),
            ("/api/cases/CASE-TEST123/jobs", "GET"),
            ("/api/cases/CASE-TEST123/ask", "POST"),
        ]

        for path, method in paths:
            response = client.request(method, path)
            assert response.status_code == 401, (
                f"{method} {path} returned "
                f"{response.status_code}, expected 401"
            )
    finally:
        app.dependency_overrides.pop(get_current_user, None)
