# 🔍 AI Image Detection Pipeline

## Complete End-to-End System for Detecting and Explaining AI-Generated Images

A production-ready pipeline that detects AI-generated images, identifies specific artifacts, and provides natural language explanations using state-of-the-art Vision Language Models (VLMs).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Pipeline Stages](#pipeline-stages)
- [Models Used](#models-used)
- [Output Format](#output-format)
- [Configuration](#configuration)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Quick Reference](#quick-reference)

---

## Overview

This system provides a **complete solution** for AI image authentication and explanation:

1. **Detects** if an image is AI-generated or real (Stage 1)
2. **Classifies** specific artifacts across 70 categories (Stage 2)
3. **Explains** why in natural language using VLMs (Stage 3)

### Key Capabilities

| Capability | Method | Output |
|------------|--------|--------|
| **Classification** | Adversarially Trained DINOv2 | REAL/FAKE + Confidence |
| **Localization** | GradCAM heatmaps | Suspicious region crop |
| **Artifact Detection** | SigLIP zero-shot | 70 artifact scores |
| **Explanation** | Qwen2-VL / Moondream2 | Natural language description |

### Model Versions

| Model Type | Clean Accuracy | Adversarial Accuracy | Use Case |
|------------|---------------|---------------------|----------|
| **Original** | 97.07% | 9.94% | Baseline detection |
| **Adversarial** | 97.13% | **66.77%** | Production/End-Term |

**The adversarial model is 6.7x more robust against attacks!**

---

## Features

### Core Features
- **Three-Stage Pipeline**: Classification → Artifact Detection → Explanation
- **70 Artifact Categories**: Comprehensive detection of AI artifacts
- **Multiple VLM Backends**: Choose between speed and quality
- **GradCAM Visualization**: Heatmaps showing suspicious regions
- **Automatic Caching**: Models download once, cached forever
- **Batch Processing**: Process multiple images efficiently
- **JSON Output**: Structured results for easy integration
- **Model Comparison**: Compare original vs adversarial models side-by-side

### Visual Outputs
- Raw GradCAM heatmap
- Heatmap overlaid on original image
- Cropped suspicious region
- Annotated image with bounding box + prediction banner

### Flexible Backends

| Backend | Quality | Speed | GPU Required | Size |
|---------|---------|-------|--------------|------|
| **Rule-Based** | Basic | Instant | No | 0 MB |
| **Moondream2** | Good | Fast | No | ~1 GB |
| **Qwen2-VL** | Excellent | Slow | Yes | ~4.2 GB |

---

## Quick Start

### Prerequisites

- **Hardware**: GPU 6GB+ VRAM (Qwen2-VL), RAM 8GB+, Storage 10GB+
- **Software**: Python 3.8+, CUDA 11.8+, Ubuntu 20.04+ (recommended)

### One-Line Setup

```bash
# Clone and setup
git clone <repository>
cd summer-camp

# Install dependencies
pip install -r requirements.txt

# Run the pipeline with adversarial model (default)
python main.py --image image.jpg --backend qwen2vl

# Or compare models
python main.py --image image.jpg --model_type both