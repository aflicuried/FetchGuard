from fastapi import FastAPI
from .routes import ctg


def create_app() -> FastAPI:
    app = FastAPI(title="FetalGuard CTG Classifier", version="0.1.0")
    app.include_router(ctg.router, prefix="/api/ctg", tags=["ctg"])
    return app


app = create_app()


