# Transient Space Event Detector (TSED)

> An autonomous, ultra-lightweight CNN system for detecting transient space events on the RP2350 microcontroller using F Prime.

## Overview

This project implements an end-to-end **Autonomous Transient Space Event Detector** designed for the **Raspberry Pi RP2350** (Pico 2) using the **F Prime (F')** flight software framework. It features an ultra-lightweight Convolutional Neural Network trained on synthetic astronomical data, quantized to INT8 for efficient edge inference on a Cortex-M33 MCU.

## Key Results

| Metric | Value |
|--------|-------|
| **Float32 Accuracy** | 99.35% |
| **INT8 Accuracy** | 98.80% |
| **Quantization Drop** | 0.55% |
| **Model Size (INT8)** | 12 KB |
| **Parameters** | 2,756 |
| **Simulation Accuracy** | 95.0% |
| **False Positive Rate** | 0.0% |

## Purpose

Space-based telescopes and satellites generate massive amounts of image data. This system provides a low-power, "on-the-edge" detection that autonomously identifies transient events directly on the spacecraft. The CNN classifies 64×64 grayscale camera frames into four categories:

| Class | Description | Action |
|-------|-------------|--------|
| `transient` | Asteroid, debris, comet streaks | **TRIGGER EVENT** |
| `starfield` | Normal star background | Ignore |
| `bright_source` | Sun, Moon, bright planets | Ignore |
| `earth_limb` | Earth horizon / atmospheric glow | Ignore |

## Project Structure

```text
.
├── ml_pipeline/                    # Machine Learning Pipeline (Python)
│   ├── generate_dataset.py         # Synthetic 4-class dataset generator
│   ├── train.py                    # CNN training with cosine LR + early stopping
│   ├── quantize.py                 # Post-training INT8 quantization
│   ├── evaluate.py                 # Metrics, confusion matrix, edge cases
│   ├── simulate.py                 # 6-scenario simulation with results
│   ├── export_to_c.py              # TFLite → C-header converter
│   ├── simulate_visual.html        # Interactive results dashboard
│   └── requirements.txt            # Python dependencies
├── fprime_workspace/               # F Prime Deployment
│   └── Components/
│       └── TransientDetector/      # F' Component for TFLite Micro inference
├── DEPLOY_GUIDE.md                 # RP2350 deployment instructions
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
cd ml_pipeline
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Dataset & Train

```bash
python3 generate_dataset.py        # 20K train + 4K test synthetic images
python3 train.py                   # Train CNN (reaches ~99.3% accuracy)
```

### 3. Quantize & Export

```bash
python3 quantize.py --model output/transient_cnn_fp32.keras
python3 export_to_c.py             # Generates model_data.h for F Prime
```

### 4. Evaluate & Simulate

```bash
python3 evaluate.py                # Confusion matrix + edge case testing
python3 simulate.py                # Run 6 visual test scenarios
```

### 5. View Results

Open `simulate_visual.html` in a browser to see the interactive dashboard, or view the generated images in `output/simulation/`.

## Architecture

```
Input (64×64×1 INT8)
  ↓
Conv2D(16, 3×3, stride=2) → BN → ReLU          → 32×32×16
  ↓
DepthwiseConv2D(3×3, stride=2) → BN → ReLU      → 16×16×16
Conv2D(32, 1×1) → BN → ReLU                     → 16×16×32
  ↓
DepthwiseConv2D(3×3, stride=2) → BN → ReLU      → 8×8×32
Conv2D(32, 1×1) → BN → ReLU                     → 8×8×32
  ↓
GlobalAveragePooling2D                           → 32
  ↓
Dense(4, softmax)                                → 4 classes
```

**Total: 2,756 parameters (10.77 KB float32, 12 KB INT8 quantized)**

## Hardware Requirements

- **Microcontroller**: Raspberry Pi RP2350 (Pico 2)
- **SRAM**: 520 KB (model uses ~120 KB tensor arena)
- **Flash**: 16 MB (model uses ~12 KB)
- **Camera**: Any grayscale camera producing 64×64 frames (e.g., HM01B0, ArduCAM)

## Deployment

See [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md) for complete instructions on:
- Building with Pico SDK + CMSIS-NN
- Generating and flashing UF2 firmware
- Camera wiring and interface
- Memory budget breakdown

## License

MIT
