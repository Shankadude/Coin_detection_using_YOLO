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









📦 Local Setup Instructions
1. Prerequisites
Ensure you have Python 3.8+ active on your device. It is highly recommended to run this within a virtual environment.

2. Clone the Repository
Bash
git clone [https://github.com/Shankadude/Coin_detection_using_YOLO.git](https://github.com/Shankadude/Coin_detection_using_YOLO.git)
cd Coin_detection_using_YOLO
3. Install System Dependencies
Install the required computer vision and deep learning matrix libraries using the preconfigured requirements manifest:

Bash
pip install -r requirements.txt
Note: If you have an NVIDIA GPU and want to leverage hardware acceleration, make sure to install a CUDA-enabled version of PyTorch.

4. Fetch the Trained Model Weights
Because deep learning weights (.pt) are heavy binaries, they are omitted from source control tracking.

Download your trained weights (best.pt).

Drop the best.pt file directly into the root directory of this project workspace.

💻 Running the Application
Open app.py and configure your video execution source near the bottom call boundary block:

Python
if __name__ == "__main__":
    # To run via an offline test sample video:
    main_pipeline(source_input="test_coins.mp4", weights_path="best.pt")
    
    # Or to run live using your computer's built-in webcam:
    # main_pipeline(source_input=0, weights_path="best.pt")
Execute the primary script in your terminal window:

Bash
python app.py
Expected Output Logs
When a coin traverses across the image plane interface, the console matrix displays automated logic routing matching your physical mechanical layout specs:

Plaintext
[System Info] Hardware Initialization successful (MOCK MODE active).
[System Info] Processing pipeline successfully engaged. Press 'q' to stop.

⚡ [ACTUATION] Triggering Servo PWM. Routing 5Rupee coin to physical Slot 3 (Gate Open 135°)!
⚡ [ACTUATION] Triggering Servo PWM. Routing 10Rupee coin to physical Slot 4 (Gate Open 180°)!

---

### How to push this to GitHub now:

Save the `README.md` file, open your terminal, and run these three standard commands to update your repository:

```bash
git add README.md
git commit -m "Docs: Added comprehensive README covering setup and architecture"
git push origin main
