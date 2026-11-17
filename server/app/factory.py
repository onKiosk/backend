import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import db_hooks, speech_hooks
from .backends import ModelBackend, create_backend
from .routers import action, health, orders, speech
from .service import ActionService

DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"


def create_app(
    backend: Optional[ModelBackend] = None, timeout_s: Optional[float] = None
) -> FastAPI:
    """환경변수: ONKIOSK_BACKEND(mock|hf|rkllm), ONKIOSK_TIMEOUT_S(기본 3.0), ONKIOSK_CORS_ORIGINS(쉼표 구분)"""
    app = FastAPI(title="OnKiosk Action API", version="0.2.0")

    origins = [o.strip() for o in os.getenv("ONKIOSK_CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    db_hooks.init()
    if os.getenv("ONKIOSK_BACKEND", "mock").lower() == "rkllm":
        speech_hooks.warm_up()
    app.state.service = ActionService(
        backend or create_backend(),
        timeout_s if timeout_s is not None else float(os.getenv("ONKIOSK_TIMEOUT_S", "3.0")),
    )
    app.include_router(health.router)
    app.include_router(action.router)
    app.include_router(orders.router)
    app.include_router(speech.router)
    return app
