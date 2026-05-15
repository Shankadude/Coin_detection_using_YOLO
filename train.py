from ultralytics import YOLO

def train_coin_model():
    # Load a lightweight pre-trained model variant (YOLOv8 Nano) perfect for embedded systems like Raspberry Pi
    model = YOLO("yolov8n.pt") 
    
    # Train the model
    results = model.train(
        data="C:/Users/dattu/OneDrive/Desktop/project code/coin detection by yolo/Indian-Coin-1/data.yaml",        
        epochs=50,                  # Start with 50 epochs; tune based on your mAP scores
        imgsz=640,                  # Image size configured in report preprocessing
        batch=16,                   # Adjust depending on your GPU VRAM
        device=0,                   # Use 0 for CUDA GPU, 'cpu' if no dedicated GPU available
        workers=4
    )
    print("Training Complete! The best weights are saved under 'runs/detect/train/weights/best.pt'")

if __name__ == "__main__":
    train_coin_model()