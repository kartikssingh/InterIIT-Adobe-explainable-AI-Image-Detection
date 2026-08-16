"""The artifact taxonomy is the contract with the competition — guard it."""

from __future__ import annotations

import pytest

from aidetect.artifacts import taxonomy


def test_exactly_seventy_unique_artifacts():
    assert len(taxonomy.OFFICIAL_ARTIFACTS) == 70
    assert len(set(taxonomy.OFFICIAL_ARTIFACTS)) == 70


def test_validate_taxonomy_passes():
    taxonomy.validate_taxonomy()


def test_every_artifact_has_a_real_category():
    for descriptor in taxonomy.OFFICIAL_ARTIFACTS:
        category = taxonomy.category_of(descriptor)
        assert category in taxonomy.CATEGORY_NAMES
        # Regression: categories used to be the descriptor itself.
        assert category != descriptor


def test_categories_partition_the_artifacts():
    total = sum(len(v) for v in taxonomy.CATEGORY_TO_DESCRIPTORS.values())
    assert total == len(taxonomy.OFFICIAL_ARTIFACTS)
    assert all(taxonomy.CATEGORY_TO_DESCRIPTORS[name] for name in taxonomy.CATEGORY_NAMES)


def test_artifact_ids_are_unique_and_stable():
    ids = [taxonomy.artifact_id(d) for d in taxonomy.OFFICIAL_ARTIFACTS]
    assert len(set(ids)) == len(ids)
    assert taxonomy.artifact_id("Artificial smoothness") == "artificial_smoothness"


@pytest.mark.parametrize(
    "query",
    ["Artificial smoothness", "artificial_smoothness", "ARTIFICIAL SMOOTHNESS"],
)
def test_resolve_descriptor_accepts_name_slug_and_case(query):
    assert taxonomy.resolve_descriptor(query) == "Artificial smoothness"


def test_resolve_descriptor_rejects_unknown():
    assert taxonomy.resolve_descriptor("purple elephant artifacts") is None


def test_prompt_rendering_covers_every_template():
    prompts = taxonomy.render_prompts("Artificial smoothness")
    assert len(prompts) == len(taxonomy.PROMPT_TEMPLATES)
    assert "Artificial smoothness" in prompts
    assert any("artificial smoothness" in p for p in prompts[1:])


def test_evidence_phrase_present_for_every_category():
    for category in taxonomy.CATEGORIES:
        assert taxonomy.evidence_phrase(category.name)


def test_taxonomy_table_shape():
    rows = taxonomy.taxonomy_table()
    assert len(rows) == 70
    assert set(rows[0]) == {"id", "descriptor", "category"}
