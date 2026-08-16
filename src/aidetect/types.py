"""Typed result objects shared by every stage.

These dataclasses are deliberately dependency-free (no torch, no PIL) so they can
be imported by the CLI, the REST API, the report writer and the test-suite
without pulling in the heavy ML stack.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

BBox = Tuple[int, int, int, int]


def _round(value: float, digits: int = 6) -> float:
    return float(round(float(value), digits))


@dataclass
class Region:
    """A rectangular suspicious area found in the saliency map."""

    bbox: BBox
    score: float
    rank: int = 1
    area_fraction: float = 0.0
    peak_xy: Tuple[int, int] = (0, 0)
    #: Peak position expressed as a percentage of width/height, handy for prose.
    peak_pct: Tuple[float, float] = (0.0, 0.0)
    crop_path: Optional[str] = None

    @property
    def width(self) -> int:
        return int(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return int(self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def position_phrase(self) -> str:
        """Human-readable position such as ``'upper-left'`` or ``'centre'``."""
        x_pct, y_pct = self.peak_pct
        vertical = "upper" if y_pct < 33 else ("lower" if y_pct > 66 else "middle")
        horizontal = "left" if x_pct < 33 else ("right" if x_pct > 66 else "centre")
        if vertical == "middle" and horizontal == "centre":
            return "centre"
        if vertical == "middle":
            return f"{horizontal} side"
        if horizontal == "centre":
            return f"{vertical} centre"
        return f"{vertical}-{horizontal}"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bbox"] = [int(v) for v in self.bbox]
        data["score"] = _round(self.score)
        data["width"] = self.width
        data["height"] = self.height
        data["position"] = self.position_phrase()
        return data


@dataclass
class DetectionResult:
    """Stage 1 output."""

    image_path: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.5
    logits: List[float] = field(default_factory=list)
    model_path: str = ""
    model_type: str = "unknown"
    checkpoint_meta: Dict[str, Any] = field(default_factory=dict)
    tta: bool = False
    latency_ms: float = 0.0

    @property
    def is_fake(self) -> bool:
        return self.prediction.upper() == "FAKE"

    @property
    def fake_prob(self) -> float:
        return float(self.probabilities.get("FAKE", 0.0))

    @property
    def real_prob(self) -> float:
        return float(self.probabilities.get("REAL", 0.0))

    #: How far the FAKE probability sits from the decision threshold.
    @property
    def margin(self) -> float:
        return abs(self.fake_prob - self.threshold)

    def certainty_label(self) -> str:
        margin = self.margin
        if margin >= 0.40:
            return "very high"
        if margin >= 0.25:
            return "high"
        if margin >= 0.10:
            return "moderate"
        return "borderline"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["probabilities"] = {k: _round(v) for k, v in self.probabilities.items()}
        data["confidence"] = _round(self.confidence, 4)
        data["is_fake"] = self.is_fake
        data["certainty"] = self.certainty_label()
        return data


@dataclass
class ArtifactMatch:
    """A single zero-shot descriptor match from Stage 2."""

    descriptor: str
    score: float
    category: str
    rank: int = 0
    raw_score: float = 0.0
    artifact_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "descriptor": self.descriptor,
            "artifact_id": self.artifact_id,
            "category": self.category,
            "score": _round(self.score),
            "raw_score": _round(self.raw_score),
        }


@dataclass
class ArtifactReport:
    """Stage 2 output."""

    matches: List[ArtifactMatch] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    model_name: str = ""
    score_mode: str = "raw"
    latency_ms: float = 0.0

    @property
    def top(self) -> Optional[ArtifactMatch]:
        return self.matches[0] if self.matches else None

    def descriptors(self, limit: Optional[int] = None) -> List[str]:
        return [m.descriptor for m in self.matches[:limit]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "score_mode": self.score_mode,
            "top_artifacts": [m.to_dict() for m in self.matches],
            "detected_categories": self.categories,
            "category_scores": {k: _round(v) for k, v in self.category_scores.items()},
            "latency_ms": self.latency_ms,
        }


@dataclass
class Explanation:
    """Stage 3 output."""

    text: str
    backend: str
    prompt: str = ""
    word_count: int = 0
    truncated: bool = False
    fallback_used: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.text,
            "backend": self.backend,
            "word_count": self.word_count or len(self.text.split()),
            "truncated": self.truncated,
            "fallback_used": self.fallback_used,
            "latency_ms": self.latency_ms,
        }


@dataclass
class AnalysisResult:
    """The complete end-to-end result for one image."""

    image_path: str
    detection: DetectionResult
    artifacts: Optional[ArtifactReport] = None
    explanation: Optional[Explanation] = None
    regions: List[Region] = field(default_factory=list)
    saved_files: Dict[str, str] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    schema_version: str = "2.0"

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def description(self) -> str:
        return self.explanation.text if self.explanation else ""

    def summary_line(self) -> str:
        verdict = self.detection.prediction
        conf = self.detection.confidence
        top = self.artifacts.top.descriptor if self.artifacts and self.artifacts.top else "-"
        return f"{self.image_path}: {verdict} ({conf:.1f}%) | top artifact: {top}"

    def to_dict(self, include_config: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "image": self.image_path,
            "detection": self.detection.to_dict(),
            "regions": [r.to_dict() for r in self.regions],
            "artifacts": self.artifacts.to_dict() if self.artifacts else None,
            "explanation": self.explanation.to_dict() if self.explanation else None,
            "saved_files": self.saved_files,
            "timings": self.timings,
            "warnings": self.warnings,
            "error": self.error,
        }
        if include_config and self.config_snapshot:
            data["config"] = self.config_snapshot
        return data

    def to_json(self, indent: int = 2, include_config: bool = True) -> str:
        return json.dumps(self.to_dict(include_config), indent=indent)

    # -- reconstruction (used by report/evaluate commands) ----------------- #

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalysisResult":
        det = data.get("detection", {}) or {}
        detection = DetectionResult(
            image_path=str(data.get("image", det.get("image_path", ""))),
            prediction=str(det.get("prediction", "UNKNOWN")),
            confidence=float(det.get("confidence", 0.0)),
            probabilities={k: float(v) for k, v in (det.get("probabilities") or {}).items()},
            threshold=float(det.get("threshold", 0.5)),
            logits=[float(v) for v in det.get("logits", [])],
            model_path=str(det.get("model_path", "")),
            model_type=str(det.get("model_type", "unknown")),
            checkpoint_meta=dict(det.get("checkpoint_meta", {})),
            tta=bool(det.get("tta", False)),
            latency_ms=float(det.get("latency_ms", 0.0)),
        )

        artifacts = None
        art_data = data.get("artifacts")
        if art_data:
            artifacts = ArtifactReport(
                matches=[
                    ArtifactMatch(
                        descriptor=str(m.get("descriptor", "")),
                        score=float(m.get("score", 0.0)),
                        category=str(m.get("category", "")),
                        rank=int(m.get("rank", 0)),
                        raw_score=float(m.get("raw_score", 0.0)),
                        artifact_id=str(m.get("artifact_id", "")),
                    )
                    for m in art_data.get("top_artifacts", [])
                ],
                categories=list(art_data.get("detected_categories", [])),
                category_scores={
                    k: float(v) for k, v in (art_data.get("category_scores") or {}).items()
                },
                model_name=str(art_data.get("model_name", "")),
                score_mode=str(art_data.get("score_mode", "raw")),
                latency_ms=float(art_data.get("latency_ms", 0.0)),
            )

        explanation = None
        exp_data = data.get("explanation")
        if exp_data:
            explanation = Explanation(
                text=str(exp_data.get("description", "")),
                backend=str(exp_data.get("backend", "unknown")),
                word_count=int(exp_data.get("word_count", 0)),
                truncated=bool(exp_data.get("truncated", False)),
                fallback_used=bool(exp_data.get("fallback_used", False)),
                latency_ms=float(exp_data.get("latency_ms", 0.0)),
            )

        regions = [
            Region(
                bbox=tuple(int(v) for v in r.get("bbox", (0, 0, 0, 0)))[:4],  # type: ignore[arg-type]
                score=float(r.get("score", 0.0)),
                rank=int(r.get("rank", 1)),
                area_fraction=float(r.get("area_fraction", 0.0)),
                peak_xy=tuple(int(v) for v in r.get("peak_xy", (0, 0)))[:2],  # type: ignore[arg-type]
                peak_pct=tuple(float(v) for v in r.get("peak_pct", (0.0, 0.0)))[:2],  # type: ignore[arg-type]
                crop_path=r.get("crop_path"),
            )
            for r in data.get("regions", [])
        ]

        return cls(
            image_path=str(data.get("image", "")),
            detection=detection,
            artifacts=artifacts,
            explanation=explanation,
            regions=regions,
            saved_files=dict(data.get("saved_files", {})),
            timings=dict(data.get("timings", {})),
            config_snapshot=dict(data.get("config", {})),
            warnings=list(data.get("warnings", [])),
            error=data.get("error"),
            schema_version=str(data.get("schema_version", "2.0")),
        )


@dataclass
class BatchSummary:
    """Aggregate statistics produced by the batch runner."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    fake: int = 0
    real: int = 0
    elapsed_s: float = 0.0
    failures: List[Dict[str, str]] = field(default_factory=list)

    @property
    def fake_rate(self) -> float:
        return self.fake / self.succeeded if self.succeeded else 0.0

    @property
    def throughput(self) -> float:
        return self.total / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["fake_rate"] = _round(self.fake_rate, 4)
        data["images_per_second"] = _round(self.throughput, 3)
        return data


def results_to_records(results: Sequence[AnalysisResult]) -> List[Dict[str, Any]]:
    """Flatten results into rows suitable for CSV export."""
    rows: List[Dict[str, Any]] = []
    for res in results:
        top = res.artifacts.top if res.artifacts else None
        rows.append(
            {
                "image": res.image_path,
                "prediction": res.detection.prediction,
                "confidence": round(res.detection.confidence, 2),
                "fake_prob": round(res.detection.fake_prob, 6),
                "real_prob": round(res.detection.real_prob, 6),
                "top_artifact": top.descriptor if top else "",
                "top_category": top.category if top else "",
                "description": res.description,
                "error": res.error or "",
            }
        )
    return rows
