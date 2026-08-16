# Explainable AI-Image Detection

Detect AI-generated images, localise the artifacts that give them away, and explain
them in plain language — one installable package, one CLI, one REST API.

```
image ──► Stage 1: DINOv2 classifier ──► REAL / FAKE + saliency map + ranked regions
                                          │
                                          └─► Stage 2: SigLIP zero-shot ──► 70 artifact scores
                                                                             │
                                                                             └─► Stage 3: VLM ──► explanation
```

---

## Table of contents

- [What this is](#what-this-is)
- [Install](#install)
- [Quick start](#quick-start)
- [The CLI](#the-cli)
- [Python API](#python-api)
- [REST API](#rest-api)
- [Configuration](#configuration)
- [How each stage works](#how-each-stage-works)
- [Output format](#output-format)
- [Competition submissions](#competition-submissions)
- [Evaluation](#evaluation)
- [Training](#training)
- [Project layout](#project-layout)
- [Testing](#testing)
- [What changed in v2](#what-changed-in-v2)
- [Migration from v1](#migration-from-v1)
- [Troubleshooting](#troubleshooting)

---

## What this is

| Stage | Question | Method | Output |
|-------|----------|--------|--------|
| **1. Detect** | Is this image generated? | Adversarially fine-tuned DINOv2 ViT-B/14 | `REAL`/`FAKE` + calibrated probabilities |
| **1b. Localise** | Where is the evidence? | GradCAM / GradCAM++ / attention rollout / occlusion | Heatmap + ranked bounding boxes + crops |
| **2. Classify** | Which artifacts? | SigLIP zero-shot over 70 descriptors | Scored artifacts in 8 semantic families |
| **3. Explain** | Why does it look fake? | Qwen2-VL / Moondream2 / rule-based writer | 2–3 sentence explanation (≤ 50 words) |

**Detector checkpoints** (from the training runs in [`training/`](training/)):

| Checkpoint | Clean accuracy | PGD-10 accuracy | Use for |
|------------|---------------:|----------------:|---------|
| `checkpoints/best_model.pt` (original) | 97.07% | 9.94% | Baseline comparison |
| `checkpoints_adv/adv_best_model.pt` (adversarial) | 97.13% | **66.77%** | **Production — 6.7× more robust** |

The adversarial checkpoint is the default. Both are loaded by the same code path;
`aidetect compare` runs them side by side on the same image.

### Description backends

| Backend | Quality | Speed | GPU | Download | Notes |
|---------|---------|-------|-----|---------:|-------|
| `rule_based` | Solid, deterministic | Instant | No | 0 MB | **Default.** Composes from Stage 1/2 evidence; never fails |
| `moondream` | Good | Fast | Optional | ~1 GB | Small VLM, CPU-viable |
| `qwen2vl` | Best | Slow | Recommended | ~4.2 GB | Most specific descriptions |

Any VLM failure (missing weights, OOM, empty answer) falls back to `rule_based`,
so the pipeline always returns an explanation.

---

## Install

```bash
git clone <repository>
cd InterIIT-Adobe-explainable-AI-Image-Detection

# Everything (detection + artifacts + VLMs + config files)
pip install -e ".[vlm,config]"

# Minimal — Stage 1 detection and saliency only
pip install -e .

# Add-ons
pip install -e ".[qwen]"     # multi-image prompting for Qwen2-VL
pip install -e ".[server]"   # the REST API
pip install -e ".[dev]"      # pytest, ruff, mypy
```

Or without installing the package:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m aidetect --help
```

**Requirements:** Python ≥ 3.9. GPU optional (CPU works; Qwen2-VL wants ~6 GB VRAM).
Put your trained weights at `checkpoints_adv/adv_best_model.pt` — see [Training](#training).

Check the environment at any time:

```bash
aidetect info               # versions, GPU, checkpoints found, backends available
aidetect info --check-models  # actually load the detector and report its shape
```

---

## Quick start

```bash
# Full pipeline on one image
aidetect analyze image.jpg

# Best explanations (downloads Qwen2-VL on first use)
aidetect analyze image.jpg --backend qwen2vl

# Fast verdict only, no saliency, no explanation
aidetect detect data/test/

# A whole folder, resumable, with submissions and an HTML report
aidetect batch data/test -o runs/test --resume --submission --report
```

Everything lands under `outputs/<image-stem>/`:

```
outputs/image/
├── saliency_overlay.png   # heatmap blended over the original
├── region_crop.png        # the strongest suspicious region
├── annotated.png          # boxes + verdict banner
├── saliency_raw.png       # heatmap alone            (opt-in)
├── summary_panel.png      # shareable one-image summary (opt-in)
└── result.json            # the complete structured result
```

---

## The CLI

`aidetect <command>` — every command accepts `--json` for machine-readable output,
`-v`/`-vv` for more logging, `-q` for errors only, and `--set key=value` to override
any configuration key.

| Command | What it does |
|---------|--------------|
| `analyze` | Full pipeline (detect → localise → classify → explain) |
| `detect` | Stage 1 only — fastest path to a verdict, batched |
| `batch` | Full pipeline over a folder; resumable, failure-isolated |
| `artifacts` | Stage 2 only — zero-shot artifact scores for a crop |
| `describe` | Stage 3 — explanation for a crop (`--no-artifacts` skips Stage 2) |
| `compare` | Same image through two checkpoints, side by side |
| `evaluate` | Accuracy / F1 / ROC-AUC / ECE + description quality |
| `report` | Standalone HTML report from finished results |
| `submission` | Write and validate Task 1 / Task 2 submission files |
| `serve` | REST API |
| `taxonomy` | List the 70 artifacts and their families |
| `info` | Environment, checkpoint and backend diagnostics |

### Examples

```bash
# Multiple images, a different saliency method, all visual outputs
aidetect analyze a.jpg b.jpg --explain-method gradcam++ \
    --save-outputs raw,overlay,crop,annotated,panel

# Stricter decision threshold + test-time augmentation
aidetect detect data/val --threshold 0.65 --tta --json > verdicts.json

# Explain even the images classified REAL (useful for error analysis)
aidetect analyze image.jpg --explain-real

# Artifact scores only, all 70, raw SigLIP values
aidetect artifacts crop.png --all --score-mode raw

# Compare checkpoints
aidetect compare image.jpg --checkpoints checkpoints_adv/adv_best_model.pt checkpoints/best_model.pt

# Only the lighting-related artifacts in the taxonomy
aidetect taxonomy --category lighting
```

---

## Python API

```python
from aidetect import AnalysisPipeline, AppConfig

config = AppConfig()
config.describe.backend = "qwen2vl"
config.detector.decision_threshold = 0.6
config.explain.method = "gradcam++"

with AnalysisPipeline(config) as pipeline:
    result = pipeline.analyze("image.jpg")

print(result.detection.prediction)        # 'FAKE'
print(result.detection.fake_prob)         # 0.9312
print(result.detection.certainty_label()) # 'very high'
print(result.regions[0].position_phrase())# 'upper-right'
print(result.artifacts.descriptors(3))    # ['Artificial smoothness', ...]
print(result.description)                 # the explanation
result.to_json()                          # full structured record
```

Use the stages independently:

```python
from aidetect import Detector, ArtifactClassifier, DescriptionGenerator

detector = Detector(config)
detection = detector.predict("image.jpg")
cam, regions = detector.localize("image.jpg")

report = ArtifactClassifier(config).classify("crop.png", top_k=5)
text = DescriptionGenerator(config).generate(report=report, detection=detection)
```

Batch processing:

```python
from aidetect.batch import run_batch
from aidetect.io_utils import iter_image_paths

with AnalysisPipeline(config) as pipeline:
    results, summary = run_batch(
        pipeline, iter_image_paths(["data/test"]), "runs/test", resume=True
    )
print(summary.succeeded, summary.fake, summary.throughput)
```

Importing `aidetect` does **not** import torch — submodules load lazily, so
`aidetect --help`, the taxonomy and the report writer are instant.

---

## REST API

```bash
pip install -e ".[server]"
aidetect serve --port 8000
```

| Endpoint | Method | Returns |
|----------|--------|---------|
| `/health` | GET | Liveness, device, active checkpoint |
| `/config` | GET | The effective configuration |
| `/taxonomy` | GET | 70 artifacts + 8 families |
| `/detect` | POST | Stage 1 verdict for an uploaded image |
| `/analyze` | POST | Full pipeline result as JSON |
| `/analyze/visual` | POST | The summary panel as a PNG |

```bash
curl -F "file=@image.jpg" http://127.0.0.1:8000/analyze | jq .detection
curl -F "file=@image.jpg" http://127.0.0.1:8000/analyze/visual -o panel.png
```

Models load once at startup and are shared across requests.

---

## Configuration

Resolution order — later wins:

```
dataclass defaults  →  YAML/JSON file  →  AIDETECT__* env vars  →  CLI --set
```

```bash
aidetect analyze img.jpg -c configs/default.yaml       # explicit file
AIDETECT__runtime__device=cpu aidetect detect img.jpg  # environment
aidetect analyze img.jpg --set artifacts.top_k=10 --set explain.colormap=inferno
```

`configs/default.yaml` is auto-discovered when you run from the project root and
documents every option. The five sections:

| Section | Key options |
|---------|-------------|
| `detector` | `checkpoint`, `decision_threshold`, `temperature`, `tta_hflip`, `batch_size` |
| `explain` | `method`, `colormap`, `max_regions`, `hot_threshold`, `min_crop_size`, `save_outputs` |
| `artifacts` | `model_name`, `top_k`, `prompt_ensemble`, `score_mode`, `embedding_cache_dir` |
| `describe` | `backend`, `max_words`, `max_sentences`, `fallback_to_rules` |
| `runtime` | `device`, `dtype`, `output_dir`, `log_level`, `explain_real_images` |

Architecture fields (`backbone`, `image_size`, `freeze_blocks`, `head_hidden_dim`,
`head_dropout`) must match the trained checkpoint — changing them makes weights
unloadable. Everything is validated on load, so a typo fails immediately with a
message naming the offending key.

---

## How each stage works

### Stage 1 — detection

DINOv2 ViT-B/14 (`vit_base_patch14_dinov2.lvd142m`, 518×518 input) with a
`Linear(768→256) → GELU → Dropout(0.3) → Linear(256→2)` head. **Identical to the
training definition** — see [`src/aidetect/core/model.py`](src/aidetect/core/model.py).

- Checkpoints load with `weights_only=True` where possible, and both the
  `{"model_state": ...}` and bare-state-dict layouts are accepted, as are
  `module.`/`_orig_mod.` prefixes from DDP or `torch.compile`.
- Probabilities are temperature-scaled and thresholded (`decision_threshold`), not
  argmaxed, so you can trade precision against recall without retraining.
- Optional horizontal-flip TTA; batched inference; automatic device/dtype selection
  (bf16 on Ampere+, fp16 on older GPUs, fp32 on CPU).

### Stage 1b — localisation

Four saliency methods, all returning a `[0, 1]` map:

| Method | Gradients? | Notes |
|--------|-----------|-------|
| `gradcam` | Yes | Default. Fast, class-specific |
| `gradcam++` | Yes | Better when several artifacts co-occur |
| `rollout` | No | Class-agnostic; a good cross-check on GradCAM |
| `occlusion` | No | Slow but assumption-free ground truth |

> **Note on the GradCAM target layer.** The model pools with `global_pool="token"`,
> so only the `[CLS]` token reaches the head — hooking the *output* of the last
> block gives every spatial token exactly zero gradient and a uniformly blank
> heatmap. The hook therefore targets `blocks[-1].norm1`. This is a correctness fix
> over v1, and `tests/test_saliency.py` guards against the regression.

The hot mask is then split into **connected components**, so two artifacts on
opposite sides of an image produce two boxes instead of one giant box spanning
both. Components are labelled on a coarse grid (pure numpy BFS — no scipy, no
OpenCV), ranked by peak activation, de-duplicated by IoU, and grown to
`min_crop_size` so downstream VLMs never reject the crop.

### Stage 2 — artifact classification

SigLIP scores the crop against all 70 official descriptors.

- **Prompt ensembling**: each descriptor is encoded through 5 phrasings and
  averaged — noticeably more stable than a single bare label.
- **Correct SigLIP scoring**: `sigmoid(image·textᵀ · logit_scale + logit_bias)`.
  The `logit_bias` term is part of the trained model; ignoring it shifts every score.
- **Score modes**: `raw` (the sigmoid value), `minmax` (default), `softmax`, `zscore`.
  Raw SigLIP scores cluster in a narrow band and are hard to threshold; the
  normalised view is what gets reported, and `raw_score` is always preserved.
- **8 semantic families**: Structure & Geometry, Texture & Material, Anatomy &
  Biology, Lighting & Shadows, Perspective & Depth, Signal & Rendering, Color &
  Tone, Style & Composition. (In v1 every descriptor mapped to itself, so the
  "category scores" carried no extra information.)
- **Disk-cached text embeddings**, keyed by a hash of model + templates +
  descriptors — a stale cache can never be silently reused.

### Stage 3 — explanation

Prompts are built from the actual evidence (verdict and certainty, ranked
artifacts, their families, and where the saliency peaked), not from a fixed
string. Output is then cleaned: filler openers stripped, sentences de-duplicated,
trimmed to `max_sentences` and `max_words` — preferring the last complete
sentence inside the budget so text never ends mid-clause.

`rule_based` composes 2–3 sentences from the same evidence, varying with the
dominant artifact family, the number of families involved, the region position and
the detector's certainty. It is deterministic, needs no download, and is the
fallback for every VLM failure.

---

## Output format

```jsonc
{
  "schema_version": "2.0",
  "image": "data/test/image_0042.png",
  "detection": {
    "prediction": "FAKE", "confidence": 93.12, "is_fake": true,
    "probabilities": {"FAKE": 0.9312, "REAL": 0.0688},
    "threshold": 0.5, "certainty": "very high",
    "model_type": "adversarial", "model_path": "checkpoints_adv/adv_best_model.pt",
    "latency_ms": 84.2
  },
  "regions": [
    {"rank": 1, "bbox": [120, 64, 310, 250], "score": 0.97,
     "position": "upper-right", "width": 190, "height": 186}
  ],
  "artifacts": {
    "top_artifacts": [
      {"rank": 1, "descriptor": "Artificial smoothness",
       "artifact_id": "artificial_smoothness", "category": "Texture & Material",
       "score": 1.0, "raw_score": 0.0421}
    ],
    "detected_categories": ["Texture & Material", "Lighting & Shadows"],
    "category_scores": {"Texture & Material": 1.0}
  },
  "explanation": {
    "description": "Surface detail in this region behaves unlike a photographed material...",
    "backend": "rule_based", "word_count": 40, "truncated": true, "fallback_used": false
  },
  "saved_files": {"overlay": "outputs/image_0042/saliency_overlay.png"},
  "timings": {"detect": 84.2, "saliency": 61.0, "total_ms": 210.4},
  "warnings": [], "error": null
}
```

A batch run also writes `results.jsonl` (streamed, one record per line),
`summary.csv`, `summary.json`, and optionally `report.html`.

---

## Competition submissions

```bash
aidetect batch data/test -o runs/test --submission
# or, from results you already have:
aidetect submission runs/test/results.jsonl -o runs/test
```

**Task 1** (`task1_submission.json`) — one row per image:

```json
[{"index": 42, "image": "image_0042.png", "prediction": "FAKE",
  "label": 1, "confidence": 93.12, "fake_probability": 0.9312}]
```

**Task 2** (`task2_submission.json`) — artifact → explanation, ≤ 50 words each:

```json
[{"index": 42, "explanation": {
    "Artificial smoothness": "Skin and fabric render as one continuous...",
    "Inconsistent shadow directions": "The cast shadow falls left while..."}}]
```

Indices are read from the filename (`image_0042.png` → `42`). Images classified
REAL are omitted from Task 2 (they have no artifacts to explain); pass
`include_real=True` to `task2_records` if your grader wants a row per image.
Both files are validated on write — duplicate indices, empty explanations and
over-budget text are reported, and the command exits non-zero if any survive.

---

## Evaluation

```bash
aidetect evaluate runs/test/results.jsonl --labels labels.csv
```

`labels.csv` needs `image,label` columns; labels may be `1/0`, `fake/real` or
`true/false`.

**Detection:** accuracy, precision, recall, specificity, F1, balanced accuracy,
MCC, ROC-AUC (tie-corrected), average precision, expected calibration error, and a
threshold sweep that reports the cut-off maximising F1 — the default 0.5 is rarely
optimal on an imbalanced split.

**Descriptions:** word-length distribution, distinct-1/2 (low values mean
templated output), artifact grounding (does the text actually use the detected
artifact vocabulary?), over-budget count, plus ROUGE-L and BLEU when references
are available.

All metrics are implemented in numpy/pure Python — no scikit-learn, no NLTK
downloads, works offline.

---

## Training

**The training code is unchanged** and lives in [`training/`](training/):

| File | Purpose |
|------|---------|
| `Dino.ipynb` | Stage-1 fine-tuning of DINOv2 on the REAL/FAKE dataset |
| `adversarialTraining.py` | PGD adversarial fine-tuning (`dataset/train` + `dataset/test`) |
| `adversarialTraining_stratified.py` | Same, for the `stratified_dataset/` layout with in-/out-of-distribution validation splits |

```bash
cd training
python adversarialTraining.py --probe                    # VRAM probe, then exit
python adversarialTraining.py --baseline_only            # evaluate the input checkpoint
python adversarialTraining.py --epochs 3 --pgd_steps 10  # full run
```

Writes `checkpoints_adv/adv_best_model.pt`, which is what the inference package
loads by default. `aidetect` never modifies these files and imports nothing from
them — the inference-side architecture is a faithful copy in
`src/aidetect/core/model.py`.

---

## Project layout

```
├── src/aidetect/
│   ├── cli.py                  # 12-command CLI
│   ├── config.py               # layered, validated, typed configuration
│   ├── types.py                # result dataclasses (dependency-free)
│   ├── pipeline.py             # stage orchestration
│   ├── batch.py                # resumable batch runner
│   ├── submission.py           # Task 1/2 writers + validator
│   ├── report.py               # standalone HTML reports
│   ├── server.py               # optional FastAPI service
│   ├── core/                   # Stage 1
│   │   ├── model.py            #   architecture (unchanged) + checkpoint loading
│   │   ├── preprocess.py       #   eval transform, TTA views
│   │   ├── detector.py         #   predict / saliency / localise
│   │   └── localization.py     #   connected-component regions (pure numpy)
│   ├── explainability/         # Stage 1b
│   │   ├── gradcam.py          #   GradCAM + GradCAM++
│   │   ├── rollout.py          #   attention rollout
│   │   ├── occlusion.py        #   occlusion sensitivity
│   │   ├── colormaps.py        #   LUT colormaps (no matplotlib)
│   │   └── visualize.py        #   overlays, boxes, panels, contact sheets
│   ├── artifacts/              # Stage 2
│   │   ├── taxonomy.py         #   70 descriptors + 8 families + prompts
│   │   ├── classifier.py       #   SigLIP zero-shot
│   │   └── embedding_cache.py  #   on-disk text-embedding cache
│   ├── describe/               # Stage 3
│   │   ├── base.py             #   backend contract + registry
│   │   ├── prompts.py          #   evidence-grounded prompt builders
│   │   ├── rule_based.py       #   deterministic writer / universal fallback
│   │   ├── moondream.py        #   Moondream2 backend
│   │   ├── qwen2vl.py          #   Qwen2-VL backend
│   │   ├── postprocess.py      #   cleaning + word/sentence budgets
│   │   └── generator.py        #   façade with automatic fallback
│   └── evaluation/             # metrics (numpy / pure Python)
├── training/                   # UNCHANGED training code
├── tests/                      # 173 tests, no weights required
├── configs/default.yaml
├── main.py, predict.py         # v1 compatibility wrappers
└── pyproject.toml
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest                    # 173 tests, ~5s, no downloads, no checkpoint needed
pytest --cov=aidetect
ruff check src tests
mypy src/aidetect
```

The suite covers the taxonomy, configuration layering, result serialisation,
localisation maths, prompt/post-processing, metrics, submissions, CLI parsing,
IO and the HTML report. Two integration files build a randomly-initialised tiny
ViT (`pretrained=False`, nothing downloaded) and run the real detector, all four
saliency methods, rendering, the whole pipeline, the batch runner and the report
writer against it.

---

## What changed in v2

The pipeline behaviour is the same idea; the implementation was rebuilt.

**Correctness fixes**

- **GradCAM produced blank maps.** Hooking the last block's output gives zero
  gradient on every spatial token under `global_pool="token"`; the "hottest
  region" then degenerated to the top-left corner. Now hooks `blocks[-1].norm1`.
- **Artifact categories were meaningless.** `{artifact: artifact}` mapped every
  descriptor to itself. Replaced with 8 real semantic families.
- **SigLIP's `logit_bias` was ignored**, shifting every zero-shot score.
- **One box for many artifacts.** The bounding box of *all* hot pixels merged
  distant regions; connected components now separate them.
- **The patch grid was assumed**, not derived — register-token backbones crashed
  on reshape. The grid now comes from the actual token count.
- **Small crops silently discarded the localisation.** Crops below the VLM's
  minimum size caused a fall back to the whole image; boxes are now grown instead.
- **`sys.path` mutation and `importlib.exec_module`** replaced by real package imports.

**New capabilities**

- 12-command CLI with `--json` output everywhere, replacing two ad-hoc scripts
- Typed, layered, validated configuration (file / env / `--set`)
- GradCAM++, attention rollout and occlusion sensitivity alongside GradCAM
- Multi-region localisation with ranking, IoU de-duplication and position phrases
- Prompt ensembling, score normalisation and a disk cache for Stage 2
- Batch runner: resumable, failure-isolated, ETA, JSONL/CSV/JSON outputs
- Submission writers **with a validator** for both tasks
- Offline evaluation: ROC-AUC, MCC, ECE, threshold sweep, ROUGE-L, BLEU, grounding
- Standalone HTML reports (images inlined, light/dark aware, XSS-safe)
- Optional FastAPI service
- Decision threshold, temperature scaling and hflip TTA
- 173 tests

**Engineering**

- 341 `print()` calls replaced by structured, level-controlled logging
- Duplicated model/config definitions collapsed into one source of truth
- `torch.load(..., weights_only=True)` where the checkpoint allows it
- Lazy imports: `import aidetect` never pulls in torch
- Dependencies cut from 15 to 5 required packages — opencv, matplotlib, scipy,
  scikit-learn, nltk, rouge-score and bert-score are gone, their functionality
  reimplemented in numpy/pure Python

---

## Migration from v1

`main.py` and `predict.py` still work; they translate the old flags onto the new
CLI and print the equivalent modern command.

| v1 | v2 |
|----|----|
| `python predict.py --image x.jpg` | `aidetect analyze x.jpg` |
| `python main.py --image x.jpg --backend qwen2vl` | `aidetect analyze x.jpg --backend qwen2vl` |
| `python main.py --model_type both` | `aidetect compare x.jpg` |
| `python stage_2_3/pipeline.py --image crop.png` | `aidetect artifacts crop.png` |
| `stage_2_3/stage2/artifact_classifier.py` | `aidetect.artifacts.classifier` |
| `stage_2_3/stage3/description_generator.py` | `aidetect.describe` |
| `result_<stem>.json`, `gradcam_*.png` in the CWD | `outputs/<stem>/` |

Result JSON changed shape (it is now versioned via `schema_version`). Old fields
map as: `task1.prediction` → `detection.prediction`, `task1.fake_prob` →
`detection.probabilities.FAKE`, `task2.description` → `explanation.description`,
`task2.top_artifacts` → `artifacts.top_artifacts`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No detector checkpoint found` | Train first, or `--checkpoint path/to/model.pt`. `aidetect info` lists what was tried |
| `Checkpoint does not match the model architecture` | The config's `backbone`/`image_size`/`head_hidden_dim` differ from training — restore the defaults |
| `Backend 'qwen2vl' is unavailable` | `pip install -e ".[vlm,qwen]"`, or use `--backend rule_based` |
| Model weights won't download | Pre-fetch: `huggingface-cli download Qwen/Qwen2-VL-2B-Instruct`. Set `HF_HOME` to a disk with room |
| CUDA out of memory | `--set detector.batch_size=1`, `--backend rule_based`, or `--device cpu` |
| Saliency map looks flat | Try `--explain-method gradcam++` or `occlusion`; lower `explain.hot_threshold` |
| Everything is classified FAKE (or REAL) | Tune the threshold: `aidetect evaluate ... --labels labels.csv` prints the optimal one, then `--threshold <value>` |
| Descriptions are all identical | `rule_based` is intentionally deterministic; switch to `--backend qwen2vl`. Check `distinct_2` in `aidetect evaluate` |
| Explanations are cut off | Raise `--max-words` / `describe.max_sentences`; `truncated: true` in the output marks it |
| Slow first Stage-2 call | Text embeddings are computed once and cached in `artifacts.embedding_cache_dir` |
| Permission errors on the cache | `export HF_HOME=/path/with/space` and set `artifacts.embedding_cache_dir` |

Run any command with `-vv` for debug logging, and `--log-file run.log` to persist it.

---

## License

MIT.
