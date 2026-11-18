"""MediVision AI — FastAPI application (fully local, no external AI services)."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # make the `ml` package importable when run from backend/
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

from app.api import auth, copilot, misc, reports, studies  # noqa: E402
from app.config import APP_VERSION, DISCLAIMER, get_settings  # noqa: E402
from app.db import init_db  # noqa: E402
from app.inference.registry import registry  # noqa: E402

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)

for noisy in ("httpx", "httpx2", "httpcore", "multipart"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
    init_db()
    from sqlmodel import Session

    from app.db import get_engine
    from app.services.auth import seed_users

    with Session(get_engine()) as s:
        seed_users(s)
    registry.load()
    yield


app = FastAPI(
    title="MediVision AI API",
    version=APP_VERSION,
    description=f"Intelligent medical imaging copilot. **{DISCLAIMER}** All models are self-trained; "
    "inference is fully local.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list + ["http://localhost:3000"],
    allow_origin_regex=r"https://.*\.(vercel\.app|hf\.space)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disclaimer_header(request: Request, call_next):  # type: ignore[no-untyped-def]
    resp = await call_next(request)
    resp.headers["X-MediVision-Disclaimer"] = "research prototype - not a medical device"
    return resp


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = (
        exc.detail
        if isinstance(exc.detail, dict)
        else {"code": "http_error", "message": str(exc.detail)}
    )
    return JSONResponse({"error": detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "code": "validation_error",
                "message": "Invalid request",
                "details": exc.errors(),
            }
        },
        status_code=422,
    )


for r in (studies.router, reports.router, copilot.router, misc.router, auth.router):
    app.include_router(r)

settings.storage_dir.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(settings.storage_dir)), name="files")
if settings.artifacts_dir.exists():
    app.mount("/artifacts", StaticFiles(directory=str(settings.artifacts_dir)), name="artifacts")


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "name": "MediVision AI API",
        "version": APP_VERSION,
        "docs": "/docs",
        "disclaimer": DISCLAIMER,
    }
