import cv2
import numpy as np
from ultralytics import YOLO
import time

# ==========================================
# HARDWARE ABSTRACTION LAYER (MOCKING DRIVER)
# ==========================================
class MockHardwareController:
    """Simulates physical components (Raspberry Pi GPIO, Servo Motors) when hardware is absent."""
    def __init__(self):
        print("[System Info] Hardware Initialization successful (MOCK MODE active).")
        # Map classes to physical sorting slots as specified in your report flowchart
        self.slot_mapping = {
            "1Rupee": "Slot 1 (Gate Open 45°)",
            "2Rupee": "Slot 2 (Gate Open 90°)",
            "5Rupee": "Slot 3 (Gate Open 135°)",
            "10Rupee": "Slot 4 (Gate Open 180°)"
        }
        
    def actuate_servo(self, denomination):
        """Simulates PWM signal transmission to move a servo motor for physical sorting."""
        slot = self.slot_mapping.get(denomination, "Unknown Slot (Reject Bin)")
        print(f"\n⚡ [ACTUATION] Triggering Servo PWM. Routing {denomination} coin to physical {slot}!")
        time.sleep(0.2) # Simulate mechanical gate travel lag time

# ==========================================
# CENTROID TRACKING / COUNTING ENGINE
# ==========================================
class SimpleCoinTracker:
    """Prevents multi-counting jitter across contiguous video frame arrays."""
    def __init__(self, proximity_threshold=40):
        self.tracked_coins = {}  # Format: {id: (x, y, timestamp, counted_flag)}
        self.next_id = 0
        self.proximity_threshold = proximity_threshold

    def update_and_check_if_new(self, centroid):
        cx, cy = centroid
        for coin_id, data in self.tracked_coins.items():
            tx, ty, _, counted_flag = data
            # Calculate Euclidean distance between points
            distance = np.sqrt((cx - tx)**2 + (cy - ty)**2)
            if distance < self.proximity_threshold:
                # Update coordinates of matching existing physical coin
                self.tracked_coins[coin_id] = (cx, cy, time.time(), counted_flag)
                return False, coin_id
                
        # Register completely distinct tracking instance if threshold wasn't hit
        new_id = self.next_id
        self.tracked_coins[new_id] = (cx, cy, time.time(), True)
        self.next_id += 1
        return True, new_id

# ==========================================
# MAIN COMPUTER VISION ENGINE
# ==========================================
def main_pipeline(source_input="test_coins.mp4", weights_path="yolov8n.pt"):
    # Initialize Core Classes
    hardware = MockHardwareController()
    tracker = SimpleCoinTracker()
    
    # Load Weights (Falls back to default weights if custom training isn't done yet)
    print(f"[System Info] Compiling network architectures using weights: {weights_path}")
    model = YOLO(weights_path)
    
    # State Engine Counters
    session_counts = {"1Rupee": 0, "2Rupee": 0, "5Rupee": 0, "10Rupee": 0}
    total_monetary_value = 0.0

    # Initialize Video Capture (Can accept local video filepath or integer 0 for webcam)
    cap = cv2.VideoCapture(source_input)
    if not cap.isOpened():
        print(f"[Error] Source target stream '{source_input}' could not be initialized.")
        return

    print("[System Info] Processing pipeline successfully engaged. Press 'q' to stop.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Stream ended

        # 1. PREPROCESSING LAYER (Resize & Contrast Adjustments)
        # Resize frame cleanly to optimize computational latency bounds
        display_frame = cv2.resize(frame, (640, 640))
        
        # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to help with shadows
        lab = cv2.cvtColor(display_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        preprocessed_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # 2. INFERENCE LAYER (YOLO Deep Learning Execution)
        results = model(preprocessed_frame, verbose=False)[0]

        # 3. ANALYSIS & TRACKING LAYER
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.5: # Ignore uncertain predictions
                continue
                
            # Extract standard bounding box edge coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            
            # Map index values back to your designated target classes
            # (Ensure these mirror the explicit names defined in your data.yaml configuration)
            class_map = {0: "1Rupee", 1: "2Rupee", 2: "5Rupee", 3: "10Rupee"}
            denomination_label = class_map.get(class_id, "Unknown")
            
            if denomination_label == "Unknown":
                continue

            # Calculate center point coordinates (Centroid Tracking)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Validate if coin is unique or a duplicate tracking frame
            is_new, coin_id = tracker.update_and_check_if_new((cx, cy))
            
            if is_new:
                # Update Memory Matrix Registers
                session_counts[denomination_label] += 1
                val_map = {"1Rupee": 1.0, "2Rupee": 2.0, "5Rupee": 5.0, "10Rupee": 10.0}
                total_monetary_value += val_map[denomination_label]
                
                # Execute Physical Automation Routing Call
                hardware.actuate_servo(denomination_label)

            # Draw visual boxes over preview frames for debugging
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(display_frame, f"ID:{coin_id} {denomination_label} ({conf:.2f})", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 4. DATA PRESENTATION SCREEN LAYER (Simulated UI Display Dashboard)
        # Overlay metrics directly onto output matrix windows
        y_offset = 40
        cv2.putText(display_frame, "=== COUNTER METRICS ===", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        for denom, count in session_counts.items():
            y_offset += 25
            cv2.putText(display_frame, f"{denom}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.putText(display_frame, f"TOTAL VALUE: Rs. {total_monetary_value}", (10, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Render complete interactive application preview matrices
        cv2.imshow("Automated Coin Sorter Framework - Test Environment", display_frame)
        
        # Kill command capture hook
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Garbage collector cleanup loops
    cap.release()
    cv2.destroyAllWindows()
    print("\n[System Info] Session safely closed down. Final valuation computed successfully.")

if __name__ == "__main__":
    # For testing right now, pass any sample video file or set to 0 to boot up your webcam!
    main_pipeline(source_input="0", weights_path="best.pt")