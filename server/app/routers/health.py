from fastapi import APIRouter, Request

from .. import db_hooks

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict:
    svc = request.app.state.service
    return {
        "status": "ok",
        "backend": svc.backend.name,
        "timeout_s": svc.timeout_s,
        "db_module": db_hooks.enabled(),
    }
