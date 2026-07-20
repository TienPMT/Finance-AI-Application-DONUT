from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.database import DEFAULT_DATABASE_PATH
from app.web_queries import DataUnavailableError
from app.web_routes import create_web_router, templates

APP_DIR = Path(__file__).parent


def create_app(database_path: Path = DEFAULT_DATABASE_PATH) -> FastAPI:
    application = FastAPI(title="KPDL Product Recommendation Demo")
    application.state.database_path = database_path
    application.mount(
        "/static",
        StaticFiles(directory=APP_DIR / "static"),
        name="static",
    )
    application.include_router(create_web_router(database_path))

    @application.exception_handler(DataUnavailableError)
    async def data_unavailable_handler(
        request: Request,
        exc: DataUnavailableError,
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"message": str(exc)},
            status_code=503,
        )

    return application


app = create_app()
