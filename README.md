# Transient Space Event Detector (TSED)

> [!WARNING]
> **Project Status: Incomplete**
> This project is currently in a prototype phase and is not a fully functional, flight-ready system. 
> **Why it is incomplete:** While the machine learning pipeline and the F Prime component structure have been established, the CNN model has only been trained on proxy data (Fashion MNIST) rather than actual astronomical transient datasets (due to the large size and specific preprocessing requirements of astronomical data). Furthermore, the complete hardware-in-the-loop testing on the actual RP2350 with a real camera sensor interface has not yet been finalized.

## Overview
This project implements an end-to-end **Autonomous Transient Space Event Detector** designed for the **Raspberry Pi RP2350** microcontroller using the **F Prime (F')** flight software framework. It features an ultra-lightweight Convolutional Neural Network (CNN) trained on astronomical data (currently modeled via a proxy dataset), quantized to INT8 for efficient edge inference.

## Purpose
Space-based telescopes and satellites generate massive amounts of image data. The goal of this project is to provide a low-power, "on-the-edge" detection system that can autonomously identify transient events (like supernovae, blazars, or fast radio bursts) locally on the spacecraft. This enables real-time alerting and selective downlinking of high-interest data, conserving valuable bandwidth and energy.

## Extrapolation & Future Applications
While currently scoped for transient astronomical events, this AI pipeline and F Prime deployment strategy can be extrapolated to various edge-AI applications in aerospace and robotics:
- **Autonomous Navigation**: Processing star-tracker or terrain camera data on the edge for pose estimation.
- **Earth Observation**: Detecting specific terrestrial phenomena (e.g., wildfires, algal blooms) directly from CubeSats.
- **Debris Tracking**: Identifying space debris moving against the star background in real-time.
- **Predictive Maintenance**: Analyzing acoustic or vibration data from internal spacecraft components using 1D adaptations of the CNN before transmitting health status.

## Project Structure
```text
.
├── ml_pipeline/                # Machine Learning Pipeline (Python)
│   ├── train.py                # CNN Training script (Fashion MNIST proxy)
│   ├── quantize.py             # Post-Training Quantization to INT8
│   ├── export_to_c.py          # TF Lite to C-header converter
│   └── setup_ml.sh             # Environment setup script
└── fprime_workspace/           # F Prime Deployment
    └── Components/
        └── TransientDetector/  # F' Component for TFLite Micro inference
```

## Libraries & Dependencies
The machine learning pipeline requires the following Python libraries:
- `tensorflow >= 2.15.0` (for model building, training, and INT8 quantization)
- `numpy` (for data manipulation)
- `binascii` (standard Python library, for C-header export)

The deployment environment requires:
- **F Prime (F')** Flight Software Framework
- **TensorFlow Lite Micro** (integrated within the F Prime build system)
- **RP2350 BSP** (Board Support Package) for F Prime compilation
- **CMake** & **GCC ARM Embedded Toolchain** (for compiling targeting the MCU)

## Hardware Requirements
- **Microcontroller**: Raspberry Pi RP2350 (e.g., Pico 2)
- **Memory**: 520KB SRAM (RP2350 standard)
- **Input**: 64x64 Grayscale image buffer (via F Prime `imageIn` port)

## Technical Specifications & RAM Requirements
- **Model Architecture**: Lightweight CNN with Depthwise Separable Convolutions.
- **Quantization**: Full INT8 (input/output/activations).
- **RAM Usage (Inference)**:
    - **Tensor Arena**: ~120 KB (defined in `TransientDetector.hpp`).
    - **Total SRAM Impact**: Fits comfortably within the RP2350's 520KB limit, leaving ~400KB for other F Prime services.
- **Latency**: Sub-50ms inference times (estimated on RP2350).

## Quick Start Instructions

### 1. Training & Exporting the Model
1. Navigate to the `ml_pipeline/` directory.
2. Run `./setup_ml.sh` to initialize the environment.
3. Activate the environment: `source venv/bin/activate`
4. Train the model: `python3 train.py`
5. Quantize and export to C header:
   ```bash
   python3 quantize.py
   python3 export_to_c.py
   ```

### 2. Building F Prime
1. Ensure your F Prime environment is set up and targeting `rp2350-pico`.
2. Build the project:
   ```bash
   fprime-util build rp2350-pico
   ```

## Key Notes
- **Working with Real Data**: The training script currently uses `Fashion MNIST` as a proxy dataset to demonstrate the pipeline. For your domain-specific use case, you should replace the data loading logic in `train.py` with actual astronomical FITS or JPEG datasets.
- **Normalization**: The F Prime component assumes the input image is `uint8` and internally normalizes it to the `int8` range `[-128, 127]` before feeding it into the interpreter.

## License
MIT
