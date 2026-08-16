"""Optional REST API (FastAPI).

Start it with ``aidetect serve``. The models load once at startup and are shared
across requests, so per-request latency is inference-only.

Endpoints
---------
``GET  /health``          liveness + loaded component status
``GET  /config``          the effective configuration
``GET  /taxonomy``        the 70 artifacts and their families
``POST /detect``          Stage 1 only (fast verdict)
``POST /analyze``         full pipeline, JSON result
``POST /analyze/visual``  full pipeline, returns the summary panel as PNG
"""

from __future__ import annotations

import io
from typing import Any, Optional

from .config import AppConfig
from .exceptions import BackendUnavailableError
from .logging_utils import get_logger
from .pipeline import AnalysisPipeline

logger = get_logger(__name__)


def _require_fastapi():
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        raise BackendUnavailableError(
            "server",
            "Install the API extras: pip install 'fastapi>=0.110' 'uvicorn[standard]'",
        ) from exc


def create_app(config: Optional[AppConfig] = None, checkpoint: Optional[str] = None) -> Any:
    """Build the FastAPI application."""
    _require_fastapi()

    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.responses import JSONResponse, Response

    from .artifacts import taxonomy
    from .io_utils import ensure_dir
    from .version import __version__

    from contextlib import asynccontextmanager

    cfg = (config or AppConfig()).validate()
    pipeline = AnalysisPipeline(cfg, checkpoint=checkpoint)

    @asynccontextmanager
    async def lifespan(_app):
        # Models load lazily on first request; release them on shutdown.
        yield
        pipeline.close()

    app = FastAPI(
        title="AI Image Detection API",
        version=__version__,
        description="Detect AI-generated images, localise artifacts and explain them.",
        lifespan=lifespan,
    )

    def _save_upload(upload: "UploadFile") -> str:
        import tempfile

        suffix = "." + (upload.filename or "upload.png").rsplit(".", 1)[-1]
        tmp_dir = ensure_dir(cfg.runtime.output_dir + "/uploads")
        handle = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, dir=str(tmp_dir)
        )
        handle.write(upload.file.read())
        handle.close()
        return handle.name

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "device": pipeline.detector.device,
            "checkpoint": str(pipeline.detector.checkpoint_path),
        }

    @app.get("/config")
    def get_config() -> dict:
        return cfg.to_dict()

    @app.get("/taxonomy")
    def get_taxonomy() -> dict:
        return {
            "categories": [
                {"key": c.key, "name": c.name, "description": c.description}
                for c in taxonomy.CATEGORIES
            ],
            "artifacts": taxonomy.taxonomy_table(),
        }

    @app.post("/detect")
    def detect(file: UploadFile = File(...)) -> dict:
        path = _save_upload(file)
        try:
            result = pipeline.detector.predict(path)
            return result.to_dict()
        except Exception as exc:  # noqa: BLE001 - surfaced as HTTP 400
            logger.exception("detect failed")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/analyze")
    def analyze(
        file: UploadFile = File(...),
        save_visuals: bool = Query(False, description="Also write visual artefacts to disk"),
    ) -> JSONResponse:
        path = _save_upload(file)
        try:
            result = pipeline.analyze(path, save_visuals=save_visuals)
            return JSONResponse(result.to_dict(include_config=False))
        except Exception as exc:  # noqa: BLE001
            logger.exception("analyze failed")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/analyze/visual")
    def analyze_visual(file: UploadFile = File(...)) -> Response:
        from PIL import Image

        from .explainability import visualize
        from .io_utils import load_image

        path = _save_upload(file)
        try:
            image = load_image(path)
            detection = pipeline.detector.predict(image, image_path=path)
            cam, regions = pipeline.detector.localize(image)
            crop = visualize.crop_region(image, regions[0], cfg.explain.min_crop_size)
            overlay = visualize.overlay_heatmap(
                image, cam, alpha=cfg.explain.overlay_alpha, colormap=cfg.explain.colormap
            )
            report = None
            if detection.is_fake:
                report = pipeline.classifier.classify(crop)
            panel: Image.Image = visualize.summary_panel(
                image,
                overlay,
                crop,
                verdict=detection.prediction,
                confidence=detection.confidence,
                artifacts=report.descriptors(3) if report else [],
                colormap=cfg.explain.colormap,
            )
            buffer = io.BytesIO()
            panel.save(buffer, format="PNG")
            return Response(content=buffer.getvalue(), media_type="image/png")
        except Exception as exc:  # noqa: BLE001
            logger.exception("analyze/visual failed")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def serve(
    config: Optional[AppConfig] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    checkpoint: Optional[str] = None,
    reload: bool = False,
) -> None:
    """Run the API with uvicorn (blocking)."""
    _require_fastapi()
    import uvicorn

    logger.info("Serving the aidetect API on http://%s:%d", host, port)
    uvicorn.run(create_app(config, checkpoint), host=host, port=port, reload=reload)
