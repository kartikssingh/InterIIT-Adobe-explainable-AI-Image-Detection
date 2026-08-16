"""Configuration layering, coercion and validation."""

from __future__ import annotations

import json

import pytest

from aidetect.config import AppConfig, load_config_file, set_by_path
from aidetect.exceptions import ConfigError


def test_defaults_match_the_trained_architecture():
    cfg = AppConfig()
    assert cfg.detector.backbone == "vit_base_patch14_dinov2.lvd142m"
    assert cfg.detector.image_size == 518
    assert cfg.detector.head_hidden_dim == 256
    assert cfg.detector.class_names == ["FAKE", "REAL"]
    assert cfg.detector.fake_index == 0


def test_roundtrip_through_dict():
    cfg = AppConfig()
    cfg.detector.decision_threshold = 0.63
    cfg.describe.backend = "qwen2vl"
    restored = AppConfig.from_dict(cfg.to_dict())
    assert restored.detector.decision_threshold == 0.63
    assert restored.describe.backend == "qwen2vl"


def test_json_serialisable():
    payload = json.loads(AppConfig().to_json())
    assert payload["artifacts"]["top_k"] == 5


def test_overrides_are_coerced_to_field_types():
    cfg = AppConfig.load(overrides=["detector.image_size=224", "detector.tta_hflip=true"], use_env=False)
    assert cfg.detector.image_size == 224
    assert cfg.detector.tta_hflip is True


def test_list_override_from_comma_string():
    cfg = AppConfig.load(overrides=["explain.save_outputs=overlay,crop"], use_env=False)
    assert cfg.explain.save_outputs == ["overlay", "crop"]


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("AIDETECT__runtime__device", "cpu")
    monkeypatch.setenv("AIDETECT__artifacts__top_k", "9")
    cfg = AppConfig.load()
    assert cfg.runtime.device == "cpu"
    assert cfg.artifacts.top_k == 9


def test_cli_override_beats_env(monkeypatch):
    monkeypatch.setenv("AIDETECT__artifacts__top_k", "9")
    cfg = AppConfig.load(overrides=["artifacts.top_k=3"])
    assert cfg.artifacts.top_k == 3


def test_yaml_file_layer(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("detector:\n  decision_threshold: 0.7\ndescribe:\n  backend: moondream\n")
    cfg = AppConfig.load(path, use_env=False)
    assert cfg.detector.decision_threshold == 0.7
    assert cfg.describe.backend == "moondream"


def test_json_file_layer(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"artifacts": {"top_k": 7}}))
    assert load_config_file(path) == {"artifacts": {"top_k": 7}}
    assert AppConfig.load(path, use_env=False).artifacts.top_k == 7


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError):
        AppConfig.from_dict({"detector": {"not_a_field": 1}})


def test_malformed_override_is_rejected():
    with pytest.raises(ConfigError):
        AppConfig.load(overrides=["detector.image_size"], use_env=False)


@pytest.mark.parametrize(
    "dotted,value",
    [
        ("detector.decision_threshold", 1.5),
        ("explain.method", "saliency-vibes"),
        ("artifacts.score_mode", "banana"),
        ("describe.backend", "gpt5"),
        ("explain.save_outputs", ["overlay", "hologram"]),
    ],
)
def test_validation_catches_bad_values(dotted, value):
    cfg = AppConfig()
    set_by_path(cfg, dotted, value)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_get_and_copy_with():
    cfg = AppConfig()
    assert cfg.get("detector.image_size") == 518
    clone = cfg.copy_with(**{"describe.backend": "moondream"})
    assert clone.describe.backend == "moondream"
    assert cfg.describe.backend == "rule_based"  # original untouched


def test_missing_config_file_raises():
    with pytest.raises(ConfigError):
        load_config_file("definitely/not/here.yaml")
