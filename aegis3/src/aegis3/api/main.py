from __future__ import annotations

import uvicorn
from fastapi import Depends, FastAPI

from aegis3 import __version__
from aegis3.api.auth import LoopbackOnlyMiddleware, hmac_auth
from aegis3.api.routes import findings, graph, hypotheses, jobs, policy, projects, reports, sources
from aegis3.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title="Aegis3 API",
        version=__version__,
        docs_url="/docs",
        redoc_url=None,
    )

    app.add_middleware(LoopbackOnlyMiddleware)

    @app.get("/v1/system/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    auth = [Depends(hmac_auth)]
    app.include_router(projects.router, prefix="/v1", dependencies=auth)
    app.include_router(sources.router, prefix="/v1", dependencies=auth)
    app.include_router(jobs.router, prefix="/v1", dependencies=auth)
    app.include_router(findings.router, prefix="/v1", dependencies=auth)
    app.include_router(graph.router, prefix="/v1", dependencies=auth)
    app.include_router(hypotheses.router, prefix="/v1", dependencies=auth)
    app.include_router(reports.router, prefix="/v1", dependencies=auth)
    app.include_router(policy.router, prefix="/v1", dependencies=auth)
    return app


app = create_app()


def run() -> None:
    uvicorn.run(
        "aegis3.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )


if __name__ == "__main__":
    run()
