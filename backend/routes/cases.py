"""Case management routes."""
import re
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.firebase import database_ref
from backend.security import get_current_user

router = APIRouter(prefix="/cases", tags=["Cases"])


class CaseCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    case_name: str = Field(min_length=3, max_length=150)
    case_description: str = Field(min_length=10, max_length=5000)

    @field_validator("case_name", "case_description", mode="before")
    @classmethod
    def strip_required_text(cls, value):
        if not isinstance(value, str):
            raise ValueError("A text value is required.")
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be blank.")
        return value


def check_case_access(case_id: str, uid: str):
    # Firebase keys cannot contain these characters; reject malformed IDs early.
    if not isinstance(case_id, str) or not case_id.strip() or re.search(r"[.#$\[\]/]", case_id):
        raise HTTPException(status_code=404, detail="Case not found.")
    case = database_ref(f"cases/{case_id}").get()
    if not isinstance(case, dict) or case.get("owner_id") != uid:
        raise HTTPException(status_code=404, detail="Case not found.")
    return case


@router.post("")
def create_case(payload: CaseCreate, user=Depends(get_current_user)):
    case_id = "CASE-" + uuid4().hex[:12].upper()
    created_at = datetime.now(timezone.utc).isoformat()
    case_data = {
        "case_id": case_id,
        "owner_id": user["uid"],
        "case_name": payload.case_name,
        "case_description": payload.case_description,
        "status": "created",
        "created_at": created_at,
        "evidence_count": 0,
    }
    # One root-level multi-location update prevents a case/index entry split-brain.
    database_ref("/").update({
        f"cases/{case_id}": case_data,
        f"users/{user['uid']}/cases/{case_id}": True,
    })
    return case_data


@router.get("")
def list_cases(user=Depends(get_current_user)):
    memberships = database_ref(f"users/{user['uid']}/cases").get() or {}
    if not isinstance(memberships, dict) or not memberships:
        return {"cases": []}

    # Read only cases explicitly indexed under this user's membership node.
    # This avoids downloading every user's case data into the API process.
    results = []
    for case_id in memberships:
        case = database_ref(f"cases/{case_id}").get()
        if isinstance(case, dict) and case.get("owner_id") == user["uid"]:
            results.append(case)
    results.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return {"cases": results}


@router.get("/{case_id}")
def get_case(case_id: str, user=Depends(get_current_user)):
    return check_case_access(case_id, user["uid"])
