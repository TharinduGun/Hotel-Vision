import os
from ultralytics import YOLO

def train():
    """
    Trains YOLOv8 on the new 'Hands in transaction' dataset.
    Uses early stopping to prevent overfitting.
    """
    # 1. Define paths
    # Resolve absolute path to the dataset's data.yaml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    dataset_yaml = os.path.join(
        project_root, 
        "resources", 
        "datasets", 
        "Hands.in.transaction.v4i.yolov8", # URL encoding removed for real filesystem path
        "data.yaml"
    )
    
    # We discovered the folder name has spaces on the filesystem but might be encoded online
    # Let's ensure the path matches the actual folder we found
    dataset_yaml = os.path.join(
        project_root, 
        "resources", 
        "datasets", 
        "Hands in transaction.v4i.yolov8",
        "data.yaml"
    )
    
    save_dir = os.path.join(project_root, "pycode", "models", "cash_detector_v2")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Dataset YAML path: {dataset_yaml}")
    print(f"Exists: {os.path.exists(dataset_yaml)}")
    if not os.path.exists(dataset_yaml):
        print("ERROR: data.yaml not found. Please review the path.")
        return

    # 2. Initialize YOLO model (starting from pre-trained weights)
    print("Loading base YOLOv8 model...")
    model = YOLO('yolov8m.pt')  # using medium model for balance of speed/accuracy

    # 3. Train with Early Stopping
    print("Starting training with Early Stopping...")
    model.train(
        data=dataset_yaml,
        epochs=100,           # Max epochs
        patience=15,          # Early stopping: stop if no improvement for 15 epochs
        batch=16,             # Adjust if running out of GPU VRAM
        imgsz=640,            # Image size
        device='cuda',        # Run on GPU
        project=save_dir,     # Where to save the runs
        name="train_v2",      # specific run name
        workers=0,            # Fixes Windows multiprocessing pagefile error
        exist_ok=True         # Allow overwriting this folder if run again
    )
    
    print(f"\nTraining Complete! Best model weights saved to:")
    print(os.path.join(save_dir, "train_v2", "weights", "best.pt"))
    print("\nTo use this new model, update CASH_MODEL_PATH in your configuration to point to this new best.pt file.")

if __name__ == "__main__":
    train()
