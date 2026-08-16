"""End-to-end orchestration of the three stages.

    Stage 1  detector      -> REAL/FAKE + saliency + ranked regions
    Stage 2  classifier    -> 70-way zero-shot artifact scores on the crop
    Stage 3  generator     -> natural-language explanation

Every stage is lazy: analysing a REAL image with default settings never loads
SigLIP or a VLM at all. Components are shared across images, so a batch pays the
model-loading cost exactly once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .artifacts.classifier import ArtifactClassifier
from .config import AppConfig
from .core.detector import Detector
from .describe.generator import DescriptionGenerator
from .exceptions import AidetectError
from .explainability import visualize
from .io_utils import ensure_dir, load_image, slugify
from .logging_utils import StageTimer, get_logger
from .types import AnalysisResult, ArtifactReport, DetectionResult, Region

logger = get_logger(__name__)


class AnalysisPipeline:
    """Full detect -> localise -> classify -> explain pipeline."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = (config or AppConfig()).validate()
        self.device = device
        self._detector: Optional[Detector] = None
        self._classifier: Optional[ArtifactClassifier] = None
        self._generator: Optional[DescriptionGenerator] = None
        self._checkpoint = checkpoint

    # -- lazily constructed components ------------------------------------- #

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            self._detector = Detector(self.config, self._checkpoint, self.device)
        return self._detector

    @property
    def classifier(self) -> ArtifactClassifier:
        if self._classifier is None:
            self._classifier = ArtifactClassifier(self.config, self.device)
        return self._classifier

    @property
    def generator(self) -> DescriptionGenerator:
        if self._generator is None:
            self._generator = DescriptionGenerator(self.config, device=self.device)
        return self._generator

    def close(self) -> None:
        """Release every loaded model."""
        for component in (self._detector, self._classifier, self._generator):
            if component is not None:
                component.unload()
        self._detector = self._classifier = self._generator = None

    def __enter__(self) -> "AnalysisPipeline":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- main entry point --------------------------------------------------- #

    def analyze(
        self,
        image_path: str | Path,
        save_visuals: bool = True,
        output_dir: Optional[str | Path] = None,
        force_explain: Optional[bool] = None,
    ) -> AnalysisResult:
        """Analyse one image and return a fully populated :class:`AnalysisResult`."""
        path = Path(image_path)
        timer = StageTimer()
        warnings: List[str] = []

        with timer("load"):
            image = load_image(path)

        # ---- Stage 1: detection ------------------------------------------ #
        with timer("detect"):
            detection = self.detector.predict(image, image_path=str(path))
        logger.info(
            "%s -> %s (%.1f%% confidence)", path.name, detection.prediction, detection.confidence
        )

        with timer("saliency"):
            cam, regions = self.detector.localize(image)

        result = AnalysisResult(
            image_path=str(path),
            detection=detection,
            regions=regions,
            config_snapshot=self.config.to_dict(),
        )

        explain_real = (
            self.config.runtime.explain_real_images if force_explain is None else force_explain
        )
        run_stage_23 = detection.is_fake or explain_real

        # ---- crop the primary region ------------------------------------- #
        crop = None
        if regions:
            crop = visualize.crop_region(
                image, regions[0], min_size=self.config.explain.min_crop_size
            )

        # ---- Stage 2: artifacts ------------------------------------------ #
        if run_stage_23:
            try:
                with timer("artifacts"):
                    result.artifacts = self.classifier.classify(crop or image)
            except AidetectError as exc:
                warnings.append(f"Stage 2 skipped: {exc}")
                logger.warning("Stage 2 failed for %s: %s", path.name, exc)

            # ---- Stage 3: description ------------------------------------ #
            try:
                with timer("describe"):
                    result.explanation = self.generator.generate(
                        image=crop or image,
                        report=result.artifacts,
                        detection=detection,
                        regions=regions,
                    )
            except AidetectError as exc:
                warnings.append(f"Stage 3 skipped: {exc}")
                logger.warning("Stage 3 failed for %s: %s", path.name, exc)
        else:
            logger.debug("Image classified REAL — Stages 2/3 skipped")

        # ---- visual artefacts -------------------------------------------- #
        if save_visuals:
            with timer("render"):
                result.saved_files = self._save_visuals(
                    image, cam, regions, detection, result.artifacts, crop, path, output_dir
                )
                if result.saved_files.get("crop") and result.regions:
                    result.regions[0].crop_path = result.saved_files["crop"]

        result.warnings = warnings
        result.timings = timer.as_dict()
        return result

    def analyze_many(
        self,
        image_paths: Sequence[str | Path],
        save_visuals: bool = True,
        output_dir: Optional[str | Path] = None,
    ) -> Iterator[AnalysisResult]:
        """Analyse images one by one, isolating per-image failures."""
        for path in image_paths:
            try:
                yield self.analyze(path, save_visuals=save_visuals, output_dir=output_dir)
            except (AidetectError, OSError) as exc:
                logger.error("Failed on %s: %s", path, exc)
                yield AnalysisResult(
                    image_path=str(path),
                    detection=DetectionResult(
                        image_path=str(path), prediction="ERROR", confidence=0.0
                    ),
                    error=str(exc),
                )

    # -- rendering --------------------------------------------------------- #

    def _save_visuals(
        self,
        image,
        cam,
        regions: Sequence[Region],
        detection: DetectionResult,
        report: Optional[ArtifactReport],
        crop,
        source: Path,
        output_dir: Optional[str | Path],
    ) -> Dict[str, str]:
        cfg = self.config.explain
        wanted = set(cfg.save_outputs)
        if not wanted:
            return {}

        base = ensure_dir(Path(output_dir or self.config.runtime.output_dir) / slugify(source.stem))
        accent = visualize.FAKE_COLOR if detection.is_fake else visualize.REAL_COLOR
        saved: Dict[str, str] = {}

        overlay = None
        if wanted & {"overlay", "panel"}:
            overlay = visualize.overlay_heatmap(
                image,
                cam,
                alpha=cfg.overlay_alpha,
                colormap=cfg.colormap,
                gamma=cfg.gamma,
            )

        if "raw" in wanted:
            heat = visualize.heatmap_image(cam, size=image.size, colormap=cfg.colormap, gamma=cfg.gamma)
            saved["raw"] = str(_save(heat, base / "saliency_raw.png"))

        if "overlay" in wanted and overlay is not None:
            saved["overlay"] = str(_save(overlay, base / "saliency_overlay.png"))

        if "crop" in wanted and crop is not None:
            saved["crop"] = str(_save(crop, base / "region_crop.png"))

        if "annotated" in wanted:
            annotated = visualize.draw_regions(image, regions, color=accent)
            annotated = visualize.add_banner(
                annotated,
                f"{detection.prediction}  ·  {detection.confidence:.1f}%  ·  {source.name}",
                accent=accent,
            )
            saved["annotated"] = str(_save(annotated, base / "annotated.png"))

        if "panel" in wanted and overlay is not None:
            panel = visualize.summary_panel(
                image,
                overlay,
                crop,
                verdict=detection.prediction,
                confidence=detection.confidence,
                artifacts=report.descriptors(3) if report else [],
                colormap=cfg.colormap,
            )
            saved["panel"] = str(_save(panel, base / "summary_panel.png"))

        return saved

    # -- diagnostics -------------------------------------------------------- #

    def describe(self) -> Dict[str, Any]:
        """Component-level status without forcing anything to load."""
        return {
            "detector": {
                "checkpoint": str(self.detector.checkpoint_path),
                "device": self.detector.device,
            },
            "artifacts": self.classifier.describe(),
            "describe": self.generator.describe(),
            "explain_method": self.config.explain.method,
        }


def _save(image, path: Path) -> Path:
    ensure_dir(path.parent)
    image.save(path)
    return path


def analyze_image(
    image_path: str | Path,
    config: Optional[AppConfig] = None,
    **kwargs: Any,
) -> AnalysisResult:
    """One-shot convenience wrapper around :class:`AnalysisPipeline`."""
    with AnalysisPipeline(config) as pipeline:
        return pipeline.analyze(image_path, **kwargs)
