"""The 70-artifact taxonomy.

``OFFICIAL_ARTIFACTS`` is the verbatim competition list and must not be reworded
— submissions are keyed on these exact strings.

What *is* new here is the grouping. The previous implementation mapped every
descriptor to itself (``{artifact: artifact}``), so "category scores" were just
the descriptor scores again and the per-category summary carried no information.
Each descriptor is now assigned to one of eight semantic families, which gives
Stage 3 a meaningful signal ("three separate lighting failures" reads very
differently from "three unrelated one-off artifacts").

Every descriptor also gets a stable slug (``artifact_id``) for use as a
dictionary key, filename or CSV column.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Categories
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Category:
    """A semantic family of artifacts."""

    key: str
    name: str
    description: str
    #: Clause used by the rule-based writer when this family dominates.
    evidence_phrase: str


CATEGORIES: Tuple[Category, ...] = (
    Category(
        key="structure",
        name="Structure & Geometry",
        description="Object boundaries, topology and impossible constructions.",
        evidence_phrase="the object geometry does not close consistently",
    ),
    Category(
        key="texture",
        name="Texture & Material",
        description="Surface detail, material response and texture statistics.",
        evidence_phrase="surface texture behaves unlike a real material",
    ),
    Category(
        key="anatomy",
        name="Anatomy & Biology",
        description="Body plans, faces, limbs and other biological structure.",
        evidence_phrase="the anatomy departs from any plausible body plan",
    ),
    Category(
        key="lighting",
        name="Lighting & Shadows",
        description="Shadow direction, reflections, occlusion and light sources.",
        evidence_phrase="the lighting cannot be produced by a single physical scene",
    ),
    Category(
        key="perspective",
        name="Perspective & Depth",
        description="Vanishing points, relative scale and depth ordering.",
        evidence_phrase="perspective and relative scale disagree across the frame",
    ),
    Category(
        key="signal",
        name="Signal & Rendering",
        description="Sampling, sharpening, aliasing and generator fingerprints.",
        evidence_phrase="the pixel statistics carry generator-side rendering marks",
    ),
    Category(
        key="color",
        name="Color & Tone",
        description="Colour coherence, gradients, skin tone and transitions.",
        evidence_phrase="colour transitions are smoother and more uniform than optics allow",
    ),
    Category(
        key="style",
        name="Style & Composition",
        description="Cinematic framing, staged depth of field and over-production.",
        evidence_phrase="the framing is staged in the way generative models over-produce scenes",
    ),
)

CATEGORY_BY_KEY: Dict[str, Category] = {c.key: c for c in CATEGORIES}
CATEGORY_NAMES: Tuple[str, ...] = tuple(c.name for c in CATEGORIES)
CATEGORY_BY_NAME: Dict[str, Category] = {c.name: c for c in CATEGORIES}


# --------------------------------------------------------------------------- #
# The official 70 artifacts (exact wording — do not edit)
# --------------------------------------------------------------------------- #

OFFICIAL_ARTIFACTS: Tuple[str, ...] = (
    "Inconsistent object boundaries",
    "Discontinuous surfaces",
    "Non-manifold geometries in rigid structures",
    "Floating or disconnected components",
    "Asymmetric features in naturally symmetric objects",
    "Misaligned bilateral elements in animal faces",
    "Irregular proportions in mechanical components",
    "Texture bleeding between adjacent regions",
    "Texture repetition patterns",
    "Over-smoothing of natural textures",
    "Artificial noise patterns in uniform surfaces",
    "Unrealistic specular highlights",
    "Inconsistent material properties",
    "Metallic surface artifacts",
    "Dental anomalies in mammals",
    "Anatomically incorrect paw structures",
    "Improper fur direction flows",
    "Unrealistic eye reflections",
    "Misshapen ears or appendages",
    "Impossible mechanical connections",
    "Inconsistent scale of mechanical parts",
    "Physically impossible structural elements",
    "Inconsistent shadow directions",
    "Multiple light source conflicts",
    "Missing ambient occlusion",
    "Incorrect reflection mapping",
    "Incorrect perspective rendering",
    "Scale inconsistencies within single objects",
    "Spatial relationship errors",
    "Depth perception anomalies",
    "Over-sharpening artifacts",
    "Aliasing along high-contrast edges",
    "Blurred boundaries in fine details",
    "Jagged edges in curved structures",
    "Random noise patterns in detailed areas",
    "Loss of fine detail in complex structures",
    "Artificial enhancement artifacts",
    "Incorrect wheel geometry",
    "Implausible aerodynamic structures",
    "Misaligned body panels",
    "Impossible mechanical joints",
    "Distorted window reflections",
    "Anatomically impossible joint configurations",
    "Unnatural pose artifacts",
    "Biological asymmetry errors",
    "Regular grid-like artifacts in textures",
    "Repeated element patterns",
    "Systematic color distribution anomalies",
    "Frequency domain signatures",
    "Color coherence breaks",
    "Unnatural color transitions",
    "Resolution inconsistencies within regions",
    "Unnatural Lighting Gradients",
    "Incorrect Skin Tones",
    "Fake depth of field",
    "Abruptly cut off objects",
    "Glow or light bleed around object boundaries",
    "Ghosting effects: Semi-transparent duplicates of elements",
    "Cinematization Effects",
    "Excessive sharpness in certain image regions",
    "Artificial smoothness",
    "Movie-poster like composition of ordinary scenes",
    "Dramatic lighting that defies natural physics",
    "Artificial depth of field in object presentation",
    "Unnaturally glossy surfaces",
    "Synthetic material appearance",
    "Multiple inconsistent shadow sources",
    "Exaggerated characteristic features",
    "Impossible foreshortening in animal bodies",
    "Scale inconsistencies within the same object class",
)

DESCRIPTORS: Tuple[str, ...] = OFFICIAL_ARTIFACTS


# --------------------------------------------------------------------------- #
# descriptor -> category
# --------------------------------------------------------------------------- #

_CATEGORY_ASSIGNMENT: Dict[str, str] = {
    # Structure & Geometry
    "Inconsistent object boundaries": "structure",
    "Discontinuous surfaces": "structure",
    "Non-manifold geometries in rigid structures": "structure",
    "Floating or disconnected components": "structure",
    "Asymmetric features in naturally symmetric objects": "structure",
    "Irregular proportions in mechanical components": "structure",
    "Impossible mechanical connections": "structure",
    "Inconsistent scale of mechanical parts": "structure",
    "Physically impossible structural elements": "structure",
    "Incorrect wheel geometry": "structure",
    "Implausible aerodynamic structures": "structure",
    "Misaligned body panels": "structure",
    "Impossible mechanical joints": "structure",
    "Abruptly cut off objects": "structure",
    # Texture & Material
    "Texture bleeding between adjacent regions": "texture",
    "Texture repetition patterns": "texture",
    "Over-smoothing of natural textures": "texture",
    "Artificial noise patterns in uniform surfaces": "texture",
    "Inconsistent material properties": "texture",
    "Metallic surface artifacts": "texture",
    "Regular grid-like artifacts in textures": "texture",
    "Repeated element patterns": "texture",
    "Artificial smoothness": "texture",
    "Unnaturally glossy surfaces": "texture",
    "Synthetic material appearance": "texture",
    # Anatomy & Biology
    "Misaligned bilateral elements in animal faces": "anatomy",
    "Dental anomalies in mammals": "anatomy",
    "Anatomically incorrect paw structures": "anatomy",
    "Improper fur direction flows": "anatomy",
    "Unrealistic eye reflections": "anatomy",
    "Misshapen ears or appendages": "anatomy",
    "Anatomically impossible joint configurations": "anatomy",
    "Unnatural pose artifacts": "anatomy",
    "Biological asymmetry errors": "anatomy",
    "Impossible foreshortening in animal bodies": "anatomy",
    "Exaggerated characteristic features": "anatomy",
    # Lighting & Shadows
    "Unrealistic specular highlights": "lighting",
    "Inconsistent shadow directions": "lighting",
    "Multiple light source conflicts": "lighting",
    "Missing ambient occlusion": "lighting",
    "Incorrect reflection mapping": "lighting",
    "Distorted window reflections": "lighting",
    "Unnatural Lighting Gradients": "lighting",
    "Glow or light bleed around object boundaries": "lighting",
    "Multiple inconsistent shadow sources": "lighting",
    "Dramatic lighting that defies natural physics": "lighting",
    # Perspective & Depth
    "Incorrect perspective rendering": "perspective",
    "Scale inconsistencies within single objects": "perspective",
    "Spatial relationship errors": "perspective",
    "Depth perception anomalies": "perspective",
    "Scale inconsistencies within the same object class": "perspective",
    "Fake depth of field": "perspective",
    "Artificial depth of field in object presentation": "perspective",
    # Signal & Rendering
    "Over-sharpening artifacts": "signal",
    "Aliasing along high-contrast edges": "signal",
    "Blurred boundaries in fine details": "signal",
    "Jagged edges in curved structures": "signal",
    "Random noise patterns in detailed areas": "signal",
    "Loss of fine detail in complex structures": "signal",
    "Artificial enhancement artifacts": "signal",
    "Frequency domain signatures": "signal",
    "Resolution inconsistencies within regions": "signal",
    "Excessive sharpness in certain image regions": "signal",
    "Ghosting effects: Semi-transparent duplicates of elements": "signal",
    # Color & Tone
    "Systematic color distribution anomalies": "color",
    "Color coherence breaks": "color",
    "Unnatural color transitions": "color",
    "Incorrect Skin Tones": "color",
    # Style & Composition
    "Cinematization Effects": "style",
    "Movie-poster like composition of ordinary scenes": "style",
}


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return re.sub(r"_{2,}", "_", slug)


ARTIFACT_IDS: Dict[str, str] = {name: _slugify(name) for name in OFFICIAL_ARTIFACTS}
ID_TO_ARTIFACT: Dict[str, str] = {v: k for k, v in ARTIFACT_IDS.items()}

#: descriptor -> human-readable category name
DESCRIPTOR_TO_CATEGORY: Dict[str, str] = {
    name: CATEGORY_BY_KEY[_CATEGORY_ASSIGNMENT[name]].name for name in OFFICIAL_ARTIFACTS
}
#: descriptor -> category key
DESCRIPTOR_TO_CATEGORY_KEY: Dict[str, str] = dict(_CATEGORY_ASSIGNMENT)

#: category name -> its descriptors, in official order
CATEGORY_TO_DESCRIPTORS: Dict[str, List[str]] = {c.name: [] for c in CATEGORIES}
for _descriptor in OFFICIAL_ARTIFACTS:
    CATEGORY_TO_DESCRIPTORS[DESCRIPTOR_TO_CATEGORY[_descriptor]].append(_descriptor)


# --------------------------------------------------------------------------- #
# Prompt templates for zero-shot classification
# --------------------------------------------------------------------------- #

#: Averaging several phrasings per descriptor measurably stabilises CLIP/SigLIP
#: zero-shot scores compared with a single bare label.
PROMPT_TEMPLATES: Tuple[str, ...] = (
    "{descriptor}",
    "a photo showing {descriptor_lower}",
    "an AI-generated image with {descriptor_lower}",
    "a close-up crop exhibiting {descriptor_lower}",
    "visual artifact: {descriptor_lower}",
)

#: Used when prompt ensembling is disabled.
SINGLE_TEMPLATE: Tuple[str, ...] = ("{descriptor}",)


def render_prompts(
    descriptor: str, templates: Sequence[str] = PROMPT_TEMPLATES
) -> List[str]:
    """Expand one descriptor into its prompt variants."""
    lowered = descriptor[0].lower() + descriptor[1:] if descriptor else descriptor
    return [
        template.format(descriptor=descriptor, descriptor_lower=lowered)
        for template in templates
    ]


# --------------------------------------------------------------------------- #
# Lookup helpers
# --------------------------------------------------------------------------- #


def category_of(descriptor: str) -> str:
    """Category name for a descriptor (``'Uncategorised'`` when unknown)."""
    return DESCRIPTOR_TO_CATEGORY.get(descriptor, "Uncategorised")


def category_key_of(descriptor: str) -> str:
    return DESCRIPTOR_TO_CATEGORY_KEY.get(descriptor, "structure")


def artifact_id(descriptor: str) -> str:
    return ARTIFACT_IDS.get(descriptor, _slugify(descriptor))


def resolve_descriptor(text: str) -> Optional[str]:
    """Resolve a descriptor from its exact name, slug or a loose match."""
    if text in ARTIFACT_IDS:
        return text
    if text in ID_TO_ARTIFACT:
        return ID_TO_ARTIFACT[text]
    slug = _slugify(text)
    if slug in ID_TO_ARTIFACT:
        return ID_TO_ARTIFACT[slug]
    lowered = text.strip().lower()
    for descriptor in OFFICIAL_ARTIFACTS:
        if descriptor.lower() == lowered:
            return descriptor
    return None


def evidence_phrase(category_name: str) -> str:
    category = CATEGORY_BY_NAME.get(category_name)
    return category.evidence_phrase if category else "the region shows synthetic rendering traces"


def validate_taxonomy() -> None:
    """Fail loudly if the taxonomy drifts out of sync (called by the tests)."""
    if len(OFFICIAL_ARTIFACTS) != 70:
        raise ValueError(f"Expected 70 artifacts, found {len(OFFICIAL_ARTIFACTS)}")
    if len(set(OFFICIAL_ARTIFACTS)) != len(OFFICIAL_ARTIFACTS):
        duplicates = [a for a in OFFICIAL_ARTIFACTS if list(OFFICIAL_ARTIFACTS).count(a) > 1]
        raise ValueError(f"Duplicate artifacts: {sorted(set(duplicates))}")
    unmapped = [a for a in OFFICIAL_ARTIFACTS if a not in _CATEGORY_ASSIGNMENT]
    if unmapped:
        raise ValueError(f"Artifacts without a category: {unmapped}")
    unknown = {v for v in _CATEGORY_ASSIGNMENT.values()} - set(CATEGORY_BY_KEY)
    if unknown:
        raise ValueError(f"Unknown category keys: {sorted(unknown)}")
    if len(set(ARTIFACT_IDS.values())) != len(ARTIFACT_IDS):
        raise ValueError("Artifact slugs are not unique")


def taxonomy_table() -> List[Dict[str, str]]:
    """Rows of ``{id, descriptor, category}`` for the ``taxonomy`` command."""
    return [
        {
            "id": ARTIFACT_IDS[descriptor],
            "descriptor": descriptor,
            "category": DESCRIPTOR_TO_CATEGORY[descriptor],
        }
        for descriptor in OFFICIAL_ARTIFACTS
    ]
