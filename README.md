# Automated Coin Sorting and Counting Using YOLO-Based Image Processing

An intelligent, IoT-enabled automation pipeline that leverages the **YOLO (You Only Look Once)** deep learning framework to detect, classify, count, and physically sort Indian coin denominations (₹1, ₹2, ₹5, ₹10) in real time. 

This repository features a **Hardware Abstraction Layer (HAL)** that seamlessly simulates Raspberry Pi GPIO signals and servo motor operations, allowing the entire application to be run, tested, and evaluated on a standard PC/Laptop without any connected physical components.

---

## 🚀 Key Features
- **Real-Time Object Detection:** Built on top of YOLOv8 for single-pass localization and classification.
- **Adaptive Preprocessing:** Uses Contrast Limited Adaptive Histogram Equalization (CLAHE) via OpenCV to mitigate ambient shadows and reflection variations.
- **Centroid Jitter Tracking:** Integrates a localized Euclidean tracking engine to prevent multi-frame double counting.
- **Hardware Simulation Driver:** Fully functional mock system prints servo motor PWM gating actions natively to the command window.

---

## 🛠️ System Architecture

```text
[ Input Video / Camera Feed ] 
             │
             ▼
┌────────────────────────────────────────┐
│ Preprocessing (Resize & CLAHE Filter)  │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ YOLOv8 Inference Engine (best.pt Model)│
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ Centroid Proximity Tracking Engine    │
└────────────┬───────────────────────────┘
             │
             ▼
   [ Is Coin New to Frame? ]
        /         \
    (Yes)         (No)
      /             \
     ▼               ▼
┌───────────────────────────┐    ┌─────────────────────┐
│ Increment Global Counter  │    │ Update Coordinates │
└────────────┬──────────────┘    └─────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ Trigger Actuator (PWM Simulation Log) │
└────────────────────────────────────────┘
