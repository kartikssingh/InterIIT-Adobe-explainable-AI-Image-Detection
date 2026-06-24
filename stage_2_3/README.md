# Stage 2 and Stage 3 - Artifact Selection and Explanation Generation

## Overview

This folder contains the Stage 2 and Stage 3 pipeline for the Adobe image detection project.

For the current mid-term handoff, Stage 1 localization is not used. The pipeline takes the full image directly and produces Task 2 explanations for fake images.

Pipeline:

1. Stage 2: uses SigLIP zero-shot matching to select artifacts from the official list of 70 artifact names.
2. Stage 3: generates short explanations using either Moondream2, Qwen2-VL, or a rule-based fallback.
3. Output: saves JSON in the expected Task 2 format, where artifact names are keys and explanations are values.

## Folder Structure

```text
Stage_2_3/
|-- README.md
|-- SUBMISSION_NOTES.md
|-- requirements.txt
|-- pipeline.py
|-- run_full_image_batch.py
|-- evaluate.py
|-- integration_example.py
|-- stage2/
|   |-- __init__.py
|   `-- artifact_classifier.py
`-- stage3/
    |-- __init__.py
    `-- description_generator.py
```

Do not submit generated folders/files such as:

```text
venv/
outputs/
__pycache__/
.DS_Store
```

## Important Current Assumption

For this handoff, the full image is used as the Stage 2/3 input.

Stage 1 localization will be integrated later. When Stage 1 is ready, replace the full image path with the localized crop image.

## Setup

Create a virtual environment and install dependencies:

```bash
cd "/path/to/Stage_2_3"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

First model run may download:

- `google/siglip-base-patch16-224` for Stage 2
- `vikhyatk/moondream2` for Stage 3, if using `--backend moondream`

## macOS Setup For Moondream

Moondream requires `pyvips`, and `pyvips` also needs the native `vips` library.

Install native `vips`:

```bash
brew install vips
```

Verify:

```bash
python -c "import pyvips; print(pyvips.version(0), pyvips.version(1), pyvips.version(2))"
```

If Python cannot find `libvips`, run Moondream commands with:

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
```

## Quick Sanity Test

Run this first to verify imports and the basic pipeline:

```bash
python evaluate.py --demo
```

Use `rule_based` first because it is fast and does not need a VLM download.

## Single Image Run

Rule-based:

```bash
python pipeline.py \
  --image "/path/to/image.png" \
  --index 1 \
  --backend rule_based \
  --top-k 3 \
  --output outputs/task2_single.json
```

Moondream:

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH \
python pipeline.py \
  --image "/path/to/image.png" \
  --index 1 \
  --backend moondream \
  --device cpu \
  --top-k 3 \
  --output outputs/task2_single_moondream.json
```

## Batch Run For Mid-Term Full-Image Mode

Rule-based batch:

```bash
python run_full_image_batch.py \
  --input-dir "/path/to/FAKE" \
  --output outputs/task2_rule_based.json \
  --raw-output outputs/raw_rule_based.json \
  --backend rule_based \
  --top-k 3 \
  --limit 10
```

Moondream batch:

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH \
python run_full_image_batch.py \
  --input-dir "/path/to/FAKE" \
  --output outputs/task2_moondream.json \
  --raw-output outputs/raw_moondream.json \
  --backend moondream \
  --device cpu \
  --top-k 3 \
  --limit 10
```

Remove `--limit 10` for the full batch.

## Output Format

The final Task 2 JSON format is:

```json
[
  {
    "index": 1,
    "explanation": {
      "Artificial smoothness": "The image shows artificial smoothness, with visual details that appear inconsistent with a natural photograph."
    }
  }
]
```

Notes:

- `index` should match the test image index.
- Explanation keys must be official artifact names.
- Each explanation is capped to 50 words in `pipeline.py`.
- `--top-k 3` means at most 3 artifact explanations per image.

## Official Artifact List

The official list of 70 artifacts is stored in:

```text
stage2/artifact_classifier.py
```

Specifically:

```python
OFFICIAL_ARTIFACTS = [...]
```

Stage 2 compares the image against these 70 names using SigLIP and returns the top matches.

## Backend Options

| Backend | Use Case | Notes |
| --- | --- | --- |
| `rule_based` | Fast debugging and safe fallback | No VLM required |
| `moondream` | Better image-specific explanations | Recommended when setup works |
| `qwen2vl` | Heavier VLM option | Use only if enough compute/time is available |

Recommended flow:

1. Test with `rule_based`.
2. Run small Moondream batch with `--limit 10`.
3. If output quality is good and time allows, run a larger Moondream batch.

## Evaluation

`evaluate.py` expects raw prediction JSON, not final submission JSON.

Raw prediction format:

```json
[
  {
    "image_id": "1.png",
    "detected_categories": ["Artificial smoothness"],
    "description": "The image appears overly smooth..."
  }
]
```

Ground truth format:

```json
[
  {
    "image_id": "1.png",
    "true_categories": ["Artificial smoothness"],
    "reference_description": "The image appears overly smooth, with natural texture details missing."
  }
]
```

Run:

```bash
python evaluate.py \
  --pred outputs/raw_moondream.json \
  --gt outputs/ground_truth_sample.json \
  --output outputs/metrics_sample.json
```

Metrics include:

- Stage 2 top-1 hit rate
- Stage 2 top-k hit rate
- exact set match
- BLEU
- ROUGE-L
- BERTScore F1

## Common Errors And Fixes

### Missing `einops`

Error:

```text
No module named 'einops'
```

Fix:

```bash
pip install einops
```

Already included in `requirements.txt`.

### Missing `pyvips`

Error:

```text
No module named 'pyvips'
```

Fix:

```bash
pip install pyvips
```

Already included in `requirements.txt`.

### Missing `libvips.42.dylib`

Error:

```text
cannot load library 'libvips.42.dylib'
```

Fix on macOS:

```bash
brew install vips
```

Then run with:

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
```

### Moondream CPU/MPS Error

Error:

```text
Passed CPU tensor to MPS op
```

Fix:

Run Moondream with:

```bash
--device cpu
```

The code in `stage3/description_generator.py` temporarily disables MPS detection during Moondream loading when CPU is requested.

### Moondream Fallback

If Moondream fails, the pipeline automatically falls back to rule-based explanation generation. This prevents batch runs from crashing.



Recommended zip command from inside `Stage_2_3`:

```bash
zip -r Stage_2_3.zip \
  README.md SUBMISSION_NOTES.md requirements.txt \
  pipeline.py run_full_image_batch.py evaluate.py integration_example.py \
  stage2 stage3 \
  -x '*/__pycache__/*' '*.DS_Store'
```

## Current Limitations

- Stage 1 localization is not integrated yet.
- Full image input may dilute artifact localization quality.
- Moondream on macOS should be run on CPU for stability.
- Qwen2-VL has not been prioritized because it is heavier.

