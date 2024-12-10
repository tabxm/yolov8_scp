# Real-Time Hand Gesture Recognition in Pakistani Sign Language Using YOLOv8-CSP

This repository contains the implementation of a real-time hand gesture recognition system for Pakistani Sign Language (PSL). The system leverages the YOLOv8-CSP object detection model to identify and classify PSL gestures from live video feeds or pre-recorded videos.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Setup Instructions](#setup-instructions)
5. [Dataset Preparation](#dataset-preparation)

---

## Introduction

Sign language recognition systems play a vital role in bridging communication gaps for the hearing and speech-impaired community. This project focuses on recognizing PSL gestures using a state-of-the-art YOLOv8-CSP model, optimized for real-time performance and high accuracy.

---

## Features

- Real-time hand gesture detection and classification.
- Support for live video streams via webcam or external camera.
- Configurable detection confidence thresholds.
- Model training pipeline for custom PSL datasets.

---

## Requirements

### Hardware Requirements
- GPU-enabled system with CUDA support (NVIDIA GPU recommended).
- At least 8GB RAM.

### Software Requirements
- Python 3.8 or later.
- Libraries:
  - `ultralytics`
  - `opencv-python`
  - `numpy`
  - `torch` (with GPU support)
  - `matplotlib` (for visualizing results)

Install dependencies via:
```bash
pip install ultralytics opencv-python numpy torch matplotlib
```
## Setup Instructions
### Clone the repository:

```bash
git clone https://github.com/yourusername/psl-hand-gesture-recognition.git
cd psl-hand-gesture-recognition
```
Place the trained YOLOv8-CSP model file (best.pt) in the root directory.

Run the detection script for real-time inference:
```
python real_time_detection.py
```
## Dataset Preparation
1. Data Collection
Collect a diverse dataset of PSL hand gestures, ensuring various lighting conditions and backgrounds.
Use tools like LabelImg to annotate images in YOLO format.
2. Data Structure
```bash
dataset/
├── images/
│   ├── train/
│   ├── val/
│   ├── test/
├── labels/
    ├── train/
    ├── val/
    ├── test/
```
3. Training Configuration
Modify the data.yaml file:
```yaml
train: dataset/images/train
val: dataset/images/val
nc: <number of classes>
names: ['gesture_1', 'gesture_2', ..., 'gesture_n']
```
