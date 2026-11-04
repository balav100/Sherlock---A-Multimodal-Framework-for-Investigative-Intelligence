"""Mr. Sherlock frontend.

Start from the project root:

    python -m streamlit run frontend/app.py

Requires: streamlit>=1.42, requests, python-dotenv
"""

from __future__ import annotations

import copy
import html
import json
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
load_dotenv(ROOT_DIR / ".env")

APP_NAME = "Mr. Sherlock"
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY", "").strip()
FIREBASE_AUTH_URL = "https://identitytoolkit.googleapis.com/v1"
API_URL = os.getenv("SHERLOCK_API_URL", "http://127.0.0.1:8001/api").rstrip("/")

STYLES_PATH = APP_DIR / "styles.css"
CACHE_PATH = APP_DIR / ".sherlock_cache.json"

UPLOAD_TYPES = [
    "jpg", "jpeg", "png", "webp", "bmp",
    "mp4", "mov", "avi", "mkv",
    "wav", "mp3", "m4a",
    "pdf", "txt", "csv", "json", "docx",
]
MAX_PASSAGES = 12
DEFAULT_PASSAGES = 6

SESSION_DEFAULTS = {
    "token": None,
    "refresh_token": None,
    "email": None,
    "display_name": None,
    "view": "home",
    "case_id": None,
    "case_name": None,
    "case_description": None,
    "notice": None,
    "flashes": [],
    "upload_generation": 0,
    "passages": DEFAULT_PASSAGES,
}

AUTH_ERRORS = {
    "EMAIL_NOT_FOUND": "No account matches that email address.",
    "INVALID_PASSWORD": "The password is incorrect.",
    "INVALID_LOGIN_CREDENTIALS": "The email address or password is incorrect.",
    "USER_DISABLED": "This account has been disabled. Contact the administrator.",
    "EMAIL_EXISTS": "An account already exists for that email address.",
    "INVALID_EMAIL": "Enter a valid email address.",
    "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts. Wait a few minutes and try again.",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def markup(text: str) -> str:
    """Collapse an HTML block to one line so Markdown does not read it as code."""
    return " ".join(line.strip() for line in text.strip().splitlines() if line.strip())


def show_html(text: str) -> None:
    st.markdown(markup(text), unsafe_allow_html=True)


def inject_styles() -> None:
    css = STYLES_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def init_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(value)


def clear_session(keep_notice: str | None = None) -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()
    if keep_notice:
        st.session_state.notice = keep_notice


def flash(kind: str, text: str) -> None:
    st.session_state.flashes.append((kind, text))


def show_flashes() -> None:
    items = list(st.session_state.flashes)
    st.session_state.flashes = []
    for kind, text in items:
        if kind == "success":
            st.success(text)
        elif kind == "error":
            st.error(text)
        else:
            st.info(text)


def section_head(number: str, title: str, note: str = "") -> None:
    note_html = f'<p class="section-note">{esc(note)}</p>' if note else ""
    show_html(
        f'<div class="section-head"><div class="section-num">{esc(number)}</div>'
        f'<h2 class="section-title">{esc(title)}</h2>{note_html}</div>'
    )


def human_size(size) -> str:
    try:
        value = float(size)
    except (TypeError, ValueError):
        return "Size unknown"
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:,.0f} {unit}" if unit == "B" else f"{value:,.1f} {unit}"
        value /= 1024
    return "Size unknown"


def modality_label(value) -> str:
    return str(value or "file").replace("_", " ").upper()


def display_name() -> str:
    name = st.session_state.get("display_name")
    if name:
        return str(name)
    email = st.session_state.get("email") or ""
    local = email.split("@")[0]
    for separator in (".", "_", "-"):
        local = local.replace(separator, " ")
    words = [word.capitalize() for word in local.split() if word]
    return " ".join(words) or "Investigator"


def initials(name: str) -> str:
    letters = "".join(word[0] for word in name.split()[:2]).upper()
    return letters or "?"


# ---------------------------------------------------------------------------
# Firebase authentication
# ---------------------------------------------------------------------------

class AuthError(Exception):
    def __init__(self, message: str, code: str = ""):
        super().__init__(message)
        self.code = code


def firebase(endpoint: str, payload: dict) -> dict:
    if not FIREBASE_API_KEY:
        raise AuthError("Firebase is not configured. Add FIREBASE_API_KEY to the .env file.")

    try:
        response = requests.post(
            f"{FIREBASE_AUTH_URL}/{endpoint}",
            params={"key": FIREBASE_API_KEY},
            json=payload,
            timeout=20,
        )
    except requests.Timeout as exc:
        raise AuthError("Firebase did not respond in time. Try again.") from exc
    except requests.RequestException as exc:
        raise AuthError("Could not reach Firebase. Check your internet connection.") from exc

    if not response.ok:
        raw = ""
        try:
            raw = str(response.json().get("error", {}).get("message", ""))
        except (ValueError, AttributeError):
            raw = ""
        code = raw.split(" :")[0].strip()
        message = AUTH_ERRORS.get(code, "Firebase could not complete the request. Try again.")
        raise AuthError(message, code)

    try:
        return response.json()
    except ValueError as exc:
        raise AuthError("Firebase returned an unreadable response.") from exc


def sign_in(email: str, password: str) -> None:
    data = firebase(
        "accounts:signInWithPassword",
        {"email": email, "password": password, "returnSecureToken": True},
    )
    lookup = firebase("accounts:lookup", {"idToken": data["idToken"]})
    users = lookup.get("users", [])
    if not users:
        raise AuthError("This account could not be found.")

    account = users[0]
    if not account.get("emailVerified", False):
        raise AuthError(
            "Verify your email address before signing in. Check your inbox for "
            "the verification link."
        )

    st.session_state.token = data["idToken"]
    st.session_state.refresh_token = data.get("refreshToken")
    st.session_state.email = account.get("email") or email
    st.session_state.display_name = account.get("displayName") or None


def create_account(name: str, email: str, password: str) -> None:
    data = firebase(
        "accounts:signUp",
        {"email": email, "password": password, "returnSecureToken": True},
    )
    token = data["idToken"]
    if name:
        firebase(
            "accounts:update",
            {"idToken": token, "displayName": name, "returnSecureToken": False},
        )
    firebase("accounts:sendOobCode", {"requestType": "VERIFY_EMAIL", "idToken": token})


def reset_password(email: str) -> None:
    try:
        firebase("accounts:sendOobCode", {"requestType": "PASSWORD_RESET", "email": email})
    except AuthError as exc:
        # Do not reveal whether an address has an account.
        if exc.code != "EMAIL_NOT_FOUND":
            raise


# ---------------------------------------------------------------------------
# Backend API
# ---------------------------------------------------------------------------

class ApiError(Exception):
    pass


def _api_detail(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return "The backend returned an unreadable response."

    detail = body.get("detail", body.get("message", "")) if isinstance(body, dict) else body
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    if isinstance(detail, list):
        parts = [
            str(item.get("msg", item)) if isinstance(item, dict) else str(item)
            for item in detail
        ]
        return "; ".join(parts) or "The request could not be completed."
    return "The request could not be completed."


def api(method: str, path: str, timeout=(10, 180), **kwargs):
    headers = dict(kwargs.pop("headers", {}) or {})
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.request(
            method,
            f"{API_URL}{path}",
            headers=headers,
            timeout=timeout,
            **kwargs,
        )
    except requests.Timeout as exc:
        raise ApiError(
            "The operation took longer than expected. Local analysis can take "
            "several minutes. Check the backend and try again."
        ) from exc
    except requests.ConnectionError as exc:
        raise ApiError(f"The Sherlock backend is not reachable at {API_URL}.") from exc
    except requests.RequestException as exc:
        raise ApiError("The request could not be completed.") from exc

    if response.status_code == 401:
        clear_session(keep_notice="Your session has expired. Sign in again to continue.")
        st.rerun()

    if not response.ok:
        raise ApiError(_api_detail(response))

    if response.status_code == 204 or not response.content:
        return {}

    try:
        return response.json()
    except ValueError as exc:
        raise ApiError("The backend returned an unreadable response.") from exc


def _records(payload, key: str) -> list[dict]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get(key, [])
    else:
        items = []
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def list_cases() -> list[dict]:
    return _records(api("GET", "/cases"), "cases")


def list_evidence(case_id: str) -> list[dict]:
    return _records(api("GET", f"/cases/{case_id}/evidence"), "evidence")


def get_index_status(case_id: str) -> dict | None:
    """Read the real index status from the backend, not the UI cache."""
    result = api("GET", f"/cases/{case_id}/index-status")
    return result if isinstance(result, dict) else None


def summarize_analysis(result, evidence_count: int) -> dict:
    result = result if isinstance(result, dict) else {}

    chunks = result.get("indexed_chunks", 0)
    if isinstance(chunks, list):
        chunks = len(chunks)
    try:
        chunks = int(chunks)
    except (TypeError, ValueError):
        chunks = 0

    records = result.get("evidence")
    processed = len(records) if isinstance(records, list) and records else evidence_count

    return {"chunks": chunks, "processed": processed}


# ---------------------------------------------------------------------------
# Local analysis summary cache
#
# The backend does not return a stored analysis summary, so the chunk and
# evidence counts are kept in a small local file keyed by user and case.
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(data: dict) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def _cache_key(case_id: str) -> str:
    return f"{st.session_state.get('email', '')}::{case_id}"


def get_analysis(case_id: str) -> dict | None:
    value = _load_cache().get(_cache_key(case_id))
    return value if isinstance(value, dict) else None


def set_analysis(case_id: str, summary: dict) -> None:
    data = _load_cache()
    data[_cache_key(case_id)] = summary
    _save_cache(data)


def clear_analysis(case_id: str) -> None:
    data = _load_cache()
    if data.pop(_cache_key(case_id), None) is not None:
        _save_cache(data)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def open_case(case_id: str, name: str, description: str = "") -> None:
    st.session_state.case_id = case_id
    st.session_state.case_name = name
    st.session_state.case_description = description
    st.session_state.view = "case"
    st.rerun()


def go_home() -> None:
    st.session_state.view = "home"
    st.session_state.case_id = None
    st.rerun()


# ---------------------------------------------------------------------------
# Shared components
# ---------------------------------------------------------------------------

@st.dialog("New case")
def new_case_dialog() -> None:
    with st.form("new_case_form"):
        name = st.text_input("Case name", placeholder="Office access review")
        description = st.text_area(
            "Case description",
            height=150,
            placeholder="Describe the incident, the parties involved and the questions to investigate.",
        )
        submitted = st.form_submit_button("Create case", type="primary", use_container_width=True)

    if not submitted:
        return

    if not name.strip() or not description.strip():
        st.warning("Enter both a case name and a case description.")
        return

    try:
        result = api(
            "POST",
            "/cases",
            json={"case_name": name.strip(), "case_description": description.strip()},
        )
    except ApiError as exc:
        st.error(str(exc))
        return

    case_id = str(result.get("case_id", "")) if isinstance(result, dict) else ""
    if not case_id:
        st.error("The backend did not return a case identifier.")
        return

    flash("success", "Case created. Upload evidence to begin the investigation.")
    open_case(case_id, name.strip(), description.strip())


def profile_popover(cases: list[dict]) -> None:
    name = display_name()
    email = st.session_state.get("email") or ""

    with st.popover(initials(name)):
        show_html(
            f'<div class="profile"><div class="avatar">{esc(initials(name))}</div>'
            f'<div><div class="profile-name">{esc(name)}</div>'
            f'<div class="profile-email">{esc(email)}</div></div></div>'
        )
        show_html(f'<div class="eyebrow">Your cases ({len(cases)})</div>')

        if not cases:
            st.caption("No cases yet. Create one to begin.")

        for index, case in enumerate(cases):
            case_id = str(case.get("case_id", ""))
            case_name = str(case.get("case_name") or "Untitled case")
            if case_id and st.button(
                case_name,
                key=f"profile_case_{index}_{case_id}",
                use_container_width=True,
            ):
                open_case(case_id, case_name, str(case.get("case_description") or ""))

        st.divider()
        if st.button("Sign out", key="profile_sign_out", use_container_width=True):
            clear_session()
            st.rerun()


def top_bar(cases: list[dict]) -> None:
    brand_col, action_col, profile_col = st.columns([6, 2, 1], vertical_alignment="center")
    with brand_col:
        show_html(
            '<div class="brand">Mr. Sherlock</div>'
            '<div class="brand-sub">Case intelligence workspace</div>'
        )
    with action_col:
        if st.button("New case", type="primary", use_container_width=True, key="topbar_new_case"):
            new_case_dialog()
    with profile_col:
        profile_popover(cases)

    show_html('<div class="top-rule"></div>')


def note_card(case: dict, index: int) -> None:
    case_id = str(case.get("case_id", ""))
    name = str(case.get("case_name") or "Untitled case")
    description = str(case.get("case_description") or "No description recorded.")

    with st.container(border=True):
        show_html(
            '<div class="paper-marker note-marker"></div>'
            f'<div class="note-title">{esc(name)}</div>'
            f'<div class="note-id">{esc(case_id or "No identifier")}</div>'
            f'<p class="note-body">{esc(description)}</p>'
        )
        if st.button(
            "Open case",
            key=f"open_case_{index}_{case_id}",
            use_container_width=True,
            disabled=not case_id,
        ):
            open_case(case_id, name, description)


def evidence_row(case_id: str, item: dict) -> None:
    name = str(item.get("original_filename") or "Unnamed file")
    sha = str(item.get("sha256") or "Identifier unavailable")
    modality = modality_label(item.get("modality"))
    size = human_size(item.get("size_bytes"))

    with st.container(border=True):
        show_html(
            f'<div class="ev-head"><div class="ev-name">{esc(name)}</div>'
            f'<div class="chips"><span class="chip">{esc(modality)}</span>'
            f'<span class="chip">{esc(size)}</span></div></div>'
            f'<div class="ev-sha"><span class="ev-label">SHA-256 identifier</span>{esc(sha)}</div>'
        )
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id and st.button("Delete evidence", key=f"delete_evidence_{case_id}_{evidence_id}"):
            try:
                api("DELETE", f"/cases/{case_id}/evidence/{evidence_id}")
                clear_analysis(case_id)
                flash("success", f"{name} was deleted. Rebuild the index before asking questions.")
                st.rerun()
            except ApiError as exc:
                flash("error", f"Could not delete {name}: {exc}")


def source_card(source: dict) -> None:
    filename = str(source.get("filename") or "Evidence file")
    chunk = str(source.get("chunk_id") or "Unavailable")
    modality = modality_label(source.get("modality"))
    similarity = source.get("similarity")
    score = f"{float(similarity):.3f}" if isinstance(similarity, (int, float)) else "n/a"

    show_html(
        f'<div class="source"><div class="source-name">{esc(filename)}</div>'
        f'<div class="source-meta"><span class="chip">{esc(modality)}</span>'
        f'<span class="mono-dim">Chunk {esc(chunk)}</span>'
        f'<span class="mono-dim">Score {esc(score)}</span></div></div>'
    )


def loading_html(items: list[dict]) -> str:
    notes = []
    for index, item in enumerate(items[:12]):
        label = esc(item.get("original_filename") or "Evidence")
        notes.append(
            f'<div class="load-note" style="--d:{index * 0.15:.2f}s">'
            f'<span class="load-pin"></span>{label}</div>'
        )
    return markup(
        '<div class="loader-card">'
        '<div class="loader-head"><span class="ring"></span><div>'
        '<div class="loader-title">Analyzing evidences</div>'
        f'<div class="loader-sub">Verifying, extracting, embedding and indexing '
        f"{len(items)} item(s). The first run can take several minutes.</div>"
        "</div></div>"
        f'<div class="load-notes">{"".join(notes)}</div>'
        '<div class="scan"><span></span></div>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Authentication page
# ---------------------------------------------------------------------------

def auth_page() -> None:
    notice = st.session_state.notice
    st.session_state.notice = None

    _, center, _ = st.columns([1, 1.25, 1])

    with center:
        show_html(
            '<div class="auth-brand"><div class="brand">Mr. Sherlock</div>'
            '<p class="lede">A forensic workspace for multimodal evidence. Every '
            "finding stays tied to the files that support it.</p></div>"
        )

        if notice:
            st.info(notice)

        with st.container(border=True):
            show_html(
                '<div class="paper-marker"></div>'
                '<div class="auth-title">Sign in to your workspace</div>'
            )

            tab_in, tab_new, tab_reset = st.tabs(["Sign in", "Create account", "Reset password"])

            with tab_in:
                with st.form("sign_in_form"):
                    email = st.text_input(
                        "Email address",
                        key="si_email",
                        placeholder="name@organization.com",
                    )
                    password = st.text_input("Password", type="password", key="si_password")
                    submitted = st.form_submit_button(
                        "Sign in", type="primary", use_container_width=True
                    )

                if submitted:
                    if not email.strip() or not password:
                        st.warning("Enter your email address and password.")
                    else:
                        try:
                            with st.spinner("Signing in"):
                                sign_in(email.strip(), password)
                        except AuthError as exc:
                            st.error(str(exc))
                        else:
                            st.session_state.view = "home"
                            st.rerun()

            with tab_new:
                with st.form("create_account_form"):
                    name = st.text_input("Full name", key="cr_name")
                    email = st.text_input(
                        "Email address",
                        key="cr_email",
                        placeholder="name@organization.com",
                    )
                    password = st.text_input(
                        "Password",
                        type="password",
                        key="cr_password",
                        help="At least 8 characters.",
                    )
                    confirm = st.text_input("Confirm password", type="password", key="cr_confirm")
                    submitted = st.form_submit_button(
                        "Create account", type="primary", use_container_width=True
                    )

                if submitted:
                    if not name.strip() or not email.strip():
                        st.warning("Enter your name and email address.")
                    elif len(password) < 8:
                        st.warning("Use at least 8 characters for the password.")
                    elif password != confirm:
                        st.error("The passwords do not match.")
                    else:
                        try:
                            with st.spinner("Creating account"):
                                create_account(name.strip(), email.strip(), password)
                        except AuthError as exc:
                            st.error(str(exc))
                        else:
                            st.success(
                                "Account created. Open the verification link sent to "
                                "your inbox, then sign in."
                            )

            with tab_reset:
                with st.form("reset_password_form"):
                    email = st.text_input(
                        "Account email address",
                        key="rs_email",
                        placeholder="name@organization.com",
                    )
                    submitted = st.form_submit_button(
                        "Send reset link", type="primary", use_container_width=True
                    )

                if submitted:
                    if not email.strip():
                        st.warning("Enter the email address for your account.")
                    else:
                        try:
                            reset_password(email.strip())
                        except AuthError as exc:
                            st.error(str(exc))
                        else:
                            st.success(
                                "If an account exists for this address, a reset link "
                                "has been sent."
                            )

        show_html('<p class="fine-print">Accounts require a verified email address.</p>')


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------

def home_page() -> None:
    load_error = None
    try:
        cases = list_cases()
    except ApiError as exc:
        cases = []
        load_error = str(exc)

    top_bar(cases)
    show_flashes()

    if not cases:
        count_text = "No cases yet"
    elif len(cases) == 1:
        count_text = "1 case"
    else:
        count_text = f"{len(cases)} cases"

    show_html(
        '<div class="page-head"><div class="eyebrow">Workspace</div>'
        '<h1 class="page-title">Your cases</h1>'
        f'<p class="lede">{esc(count_text)}. Open a case to manage its evidence and '
        "questions, or start a new one.</p></div>"
    )

    if load_error:
        st.error(load_error)
        return

    if not cases:
        with st.container(border=True):
            show_html(
                '<div class="empty-title">Start your first investigation</div>'
                '<p class="empty-text">Create a case with a name and description. '
                "Its evidence, analysis and questions will be kept together.</p>"
            )
        return

    for row_start in range(0, len(cases), 3):
        row = cases[row_start:row_start + 3]
        columns = st.columns(3, gap="medium")
        for offset, (column, case) in enumerate(zip(columns, row)):
            with column:
                note_card(case, row_start + offset)


# ---------------------------------------------------------------------------
# Case investigation page
# ---------------------------------------------------------------------------

def upload_files(case_id: str, files: list) -> None:
    stored = 0
    failures: list[str] = []
    progress = st.progress(0.0, text="Uploading evidence")

    for index, item in enumerate(files, start=1):
        try:
            api(
                "POST",
                f"/cases/{case_id}/evidence",
                files={
                    "file": (
                        item.name,
                        item.getvalue(),
                        item.type or "application/octet-stream",
                    )
                },
                timeout=(10, 180),
            )
            stored += 1
        except ApiError as exc:
            failures.append(f"{item.name}: {exc}")
        progress.progress(index / len(files), text=f"Processed {index} of {len(files)}")

    if stored:
        # New evidence makes any earlier analysis out of date.
        clear_analysis(case_id)
        st.session_state.upload_generation += 1
        flash("success", f"{stored} file(s) stored. Each has a SHA-256 identifier in the vault.")

    for failure in failures:
        flash("error", failure)

    st.rerun()


def upload_section(case_id: str) -> None:
    section_head(
        "01",
        "Upload evidence",
        "Add images, recordings, documents and text files. Each stored file receives "
        "a unique SHA-256 identifier.",
    )

    with st.container(border=True):
        files = st.file_uploader(
            "Choose files",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"uploader_{case_id}_{st.session_state.upload_generation}",
        )
        if st.button(
            "Upload evidence",
            type="primary",
            use_container_width=True,
            disabled=not files,
            key=f"upload_button_{case_id}",
        ):
            upload_files(case_id, files)


def vault_section(case_id: str, items: list[dict]) -> None:
    note = f"{len(items)} item(s) stored for this case." if items else ""
    section_head("02", "Evidence vault", note)

    if not items:
        with st.container(border=True):
            show_html(
                '<div class="empty-title">The vault is empty</div>'
                '<p class="empty-text">Uploaded files appear here with their modality, '
                "size and SHA-256 identifier.</p>"
            )
        return

    for item in items:
        evidence_row(case_id, item)


def run_analysis(case_id: str, items: list[dict]) -> None:
    slot = st.empty()
    slot.markdown(loading_html(items), unsafe_allow_html=True)

    try:
        result = api("POST", f"/cases/{case_id}/ingest-index", timeout=(10, 1800))
    except ApiError as exc:
        slot.empty()
        st.error(str(exc))
        return

    slot.empty()
    set_analysis(case_id, summarize_analysis(result, len(items)))
    flash("success", "Analysis complete.")
    st.rerun()


def analysis_section(
    case_id: str,
    items: list[dict],
    index_status: dict | None,
) -> None:
    section_head(
        "03",
        "Analysis",
        "Existing indexes are detected automatically. Only analyze evidence when a valid index is unavailable.",
    )

    if not items:
        with st.container(border=True):
            show_html(
                '<div class="empty-title">Nothing to analyze yet</div>'
                '<p class="empty-text">Upload evidence first. The analysis step opens '
                "once the vault contains files.</p>"
            )
        return

    ready = bool(index_status and index_status.get("ready"))
    if ready:
        chunks = int(index_status.get("total_chunks", 0) or 0)
        processed = int(index_status.get("processed_evidence", 0) or 0)
        total_evidence = int(index_status.get("total_evidence", len(items)) or len(items))
        show_html(
            '<div class="stats">'
            '<div class="stat"><div class="stat-label">Total chunks</div>'
            f'<div class="stat-value">{chunks:,}</div></div>'
            '<div class="stat"><div class="stat-label">Evidences processed</div>'
            f'<div class="stat-value">{processed:,} / {total_evidence:,}</div></div>'
            '</div>'
        )
        show_html(
            '<div class="empty-title" style="margin-top:1.25rem">Existing index verified</div>'
            '<p class="empty-text">FAISS and chunks.json are present and consistent. '
            'Ask Sherlock is enabled; no re-indexing is needed.</p>'
        )
        records = index_status.get("evidence", [])
        if records:
            show_html('<div class="label" style="margin:1.2rem 0 .65rem">Indexed evidence breakdown</div>')
            for record in records:
                if not isinstance(record, dict):
                    continue
                filename = esc(record.get("filename", "Unknown evidence"))
                modality = esc(modality_label(record.get("modality", "unknown")))
                count = int(record.get("chunks", 0) or 0)
                status_label = "Indexed" if count > 0 else "Not indexed"
                show_html(
                    '<div class="evidence-row">'
                    '<div class="evidence-main">'
                    f'<div class="evidence-name">{filename}</div>'
                    f'<div class="evidence-meta">{modality} · {status_label}</div>'
                    '</div>'
                    f'<div class="evidence-size">{count} chunk{"s" if count != 1 else ""}</div>'
                    '</div>'
                )
        if index_status.get("index_path"):
            show_html(
                '<div class="mono-dim" style="margin-top:.8rem">Index: '
                f'{esc(index_status["index_path"])}</div>'
            )
        return

    if index_status and index_status.get("reason"):
        st.caption(f"Existing index not ready: {index_status['reason']}")
    with st.container(border=True):
        show_html(
            '<div class="empty-title">Index not ready</div>'
            '<p class="empty-text">No valid, matching FAISS index and chunks.json were '
            'confirmed for this case. Analyze evidence to create or rebuild the index.</p>'
        )
        if st.button(
            "Analyze evidences",
            type="primary",
            use_container_width=True,
            key=f"analyze_{case_id}",
        ):
            run_analysis(case_id, items)


def render_message(message: dict) -> None:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["text"])
        return

    with st.chat_message("assistant"):
        if message.get("error"):
            st.error(message["error"])
            return

        result = message.get("result") or {}
        answer = result.get("answer", "No answer text was returned.")

        if str(result.get("answer_type", "")).lower() == "source_excerpt_fallback":
            st.caption(
                "No validated conclusion was returned. The passages below are the "
                "closest evidence found."
            )

        if isinstance(answer, (dict, list)):
            st.code(json.dumps(answer, indent=2, ensure_ascii=False), language="json")
        else:
            st.markdown(str(answer))

        # Provenance Validation Pill
        prov = result.get("provenance_validation")
        if isinstance(prov, dict) and prov.get("status"):
            status_color = "green" if prov.get("status") == "passed" else "red"
            st.caption(
                f"🛡️ Provenance Validation: :{status_color}[{prov.get('status', '').upper()}] "
                f"({prov.get('valid_claims', 0)}/{prov.get('claims_checked', 0)} claims verified)"
            )

        # Cross-Evidence Correlations
        correlations = result.get("correlations") or []
        if correlations and isinstance(correlations, list):
            with st.expander(f"🔗 Cross-Evidence Correlations ({len(correlations)})", expanded=True):
                for corr in correlations:
                    match_type = str(corr.get("match_type", "correlation")).replace("_", " ").title()
                    strength = corr.get("lead_strength", 0.0)
                    evidence_ids = ", ".join(corr.get("evidence_ids", []))
                    st.markdown(f"**{match_type}** (Lead strength: `{strength:.2f}` | Exhibits: `{evidence_ids}`)")
                    if anchors := corr.get("shared_anchors"):
                        st.markdown(f"*Shared anchors*: {', '.join(f'`{a}`' for a in anchors)}")
                    if interp := corr.get("interpretation"):
                        st.info(interp)
                    if limits := corr.get("limitations"):
                        for lim in limits:
                            st.caption(f"⚠️ {lim}")
                    st.divider()

        # Contradictions
        contradictions = result.get("contradictions") or []
        if contradictions and isinstance(contradictions, list):
            with st.expander(f"⚠️ Potential Contradictions ({len(contradictions)})", expanded=True):
                for con in contradictions:
                    subject = con.get("subject_anchor", "Unknown")
                    topic = str(con.get("topic", "conflict")).replace("_", " ").title()
                    st.markdown(f"**Conflicting {topic} for `{subject}`**")
                    for stmt in con.get("statements", []):
                        state = stmt.get("state", "State").upper()
                        text = stmt.get("text", "")
                        st.markdown(f"- **[{state}]**: {text}")
                    if note := con.get("note"):
                        st.caption(f"ℹ️ {note}")
                    st.divider()

        # Disputed identifiers: a model visual description naming an ID that OCR does not contain
        disputes = result.get("identifier_conflicts") or []
        if disputes and isinstance(disputes, list):
            with st.expander(f"🚫 Disputed Identifiers ({len(disputes)})", expanded=True):
                for dispute in disputes:
                    evidence_id = dispute.get("evidence_id", "")
                    ocr_value = dispute.get("ocr_value", "")
                    disputed_value = dispute.get("disputed_value", "")
                    st.markdown(
                        f"**`{esc(disputed_value)}`** is not in the OCR of `{esc(evidence_id)}`; "
                        f"OCR reads **`{esc(ocr_value)}`**."
                    )
                    if note := dispute.get("note"):
                        st.caption(f"ℹ️ {note}")
                    st.divider()

        # Atomic Claims
        claims = result.get("claims") or []
        if claims and isinstance(claims, list):
            with st.expander(f"📋 Extracted Claims ({len(claims)})", expanded=False):
                for claim in claims:
                    classification = claim.get("classification", "SOURCE_TEXT")
                    text = claim.get("text", "")
                    disputed = claim.get("disputed_identifiers") or []
                    if disputed:
                        st.markdown(
                            f"- `{classification}`: {text} — :orange[DISPUTED identifier(s) "
                            f"not present in OCR: {esc(', '.join(disputed))}]"
                        )
                    else:
                        st.markdown(f"- `{classification}`: {text}")

        raw_sources = result.get("sources", [])
        if isinstance(raw_sources, list):
            sources = [s for s in raw_sources if isinstance(s, dict)]
        else:
            sources = []

        if sources:
            show_html('<div class="label">Retrieved passages</div>')
            for source in sources:
                source_card(source)
        else:
            st.caption("No passages were returned for this answer.")


def chatbot(case_id: str) -> None:
    history_key = f"chat_{case_id}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []
    history: list[dict] = st.session_state[history_key]

    with st.container(border=True):
        passages = st.slider(
            "Passages to retrieve",
            min_value=1,
            max_value=MAX_PASSAGES,
            value=st.session_state.passages,
            key=f"passages_{case_id}",
        )
        st.session_state.passages = passages
        history_box = st.container()

    prompt = st.chat_input("Ask a question about this case", key=f"chat_input_{case_id}")

    if prompt and prompt.strip():
        question = prompt.strip()
        history.append({"role": "user", "text": question})

        with history_box:
            with st.spinner("Searching the evidence"):
                try:
                    result = api(
                        "POST",
                        f"/cases/{case_id}/ask",
                        json={"question": question, "top_k": int(passages)},
                        timeout=(10, 300),
                    )
                except ApiError as exc:
                    history.append({"role": "assistant", "error": str(exc)})
                else:
                    history.append({"role": "assistant", "result": result})

    with history_box:
        if not history:
            st.caption("Ask a question to begin. Answers cite the evidence passages they draw on.")
        for message in history:
            render_message(message)


def ask_section(case_id: str, summary: dict | None) -> None:
    section_head(
        "04",
        "Ask Sherlock",
        "Questions are answered only from the indexed evidence. The passages used for "
        "each answer are listed beneath it.",
    )

    if summary is None:
        with st.container(border=True):
            show_html(
                '<div class="empty-title">Available after analysis</div>'
                '<p class="empty-text">Run the analysis to build the index. Ask Sherlock '
                "opens when it is complete.</p>"
            )
        return

    chat_flag = f"chat_open_{case_id}"
    if not st.session_state.get(chat_flag):
        if st.button(
            "Ask Sherlock",
            type="primary",
            use_container_width=True,
            key=f"open_chat_{case_id}",
        ):
            st.session_state[chat_flag] = True
            st.rerun()
        return

    chatbot(case_id)


def case_page() -> None:
    case_id = st.session_state.case_id

    try:
        cases = list_cases()
    except ApiError:
        cases = []

    top_bar(cases)
    show_flashes()

    if st.button("Back to cases", key="back_to_cases"):
        go_home()

    try:
        case = api("GET", f"/cases/{case_id}")
    except ApiError as exc:
        st.error(str(exc))
        return

    name = str(case.get("case_name") or st.session_state.case_name or "Investigation")
    description = str(case.get("case_description") or st.session_state.case_description or "")
    st.session_state.case_name = name

    show_html(
        '<div class="page-head"><div class="eyebrow">Case file</div>'
        f'<h1 class="page-title">{esc(name)}</h1>'
        f'<p class="lede">{esc(description)}</p>'
        f'<div class="mono-dim">{esc(case_id)}</div></div>'
    )

    try:
        items = list_evidence(case_id)
    except ApiError as exc:
        st.error(str(exc))
        items = []

    upload_section(case_id)
    vault_section(case_id, items)

    index_status = None
    if items:
        try:
            index_status = get_index_status(case_id)
        except ApiError as exc:
            st.warning(f"Could not verify the saved index: {exc}")

    analysis_section(case_id, items, index_status)
    ask_summary = None
    if index_status and index_status.get("ready"):
        ask_summary = {
            "chunks": index_status.get("total_chunks", 0),
            "processed": index_status.get("processed_evidence", 0),
        }
    ask_section(case_id, ask_summary)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=":material/search:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    init_state()

    if not st.session_state.token:
        auth_page()
        return

    if st.session_state.view == "case" and st.session_state.case_id:
        case_page()
    else:
        home_page()


if __name__ == "__main__":
    main()
