import os
import cv2
from ultralytics import YOLO

def main():
    video_path = "resources/videos/Indoor_Original_VideoStream.mp4"
    model_path = "pycode/models/gun_detector/weights/best.pt"
    
    out_dir = "resources/datasets/hard_negatives"
    images_dir = os.path.join(out_dir, "images", "train")
    labels_dir = os.path.join(out_dir, "labels", "train")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
        
    frame_count = 0
    saved_count = 0
    
    print("Extracting hard negatives. This might take a while...")
    print(f"Checking frames for false positives at 0.30 confidence threshold...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process every 5th frame to avoid storing identical images
        if frame_count % 5 != 0:
            continue
            
        # Run detection at low confidence to catch potential false positives
        results = model(frame, conf=0.30, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Add 20% padding to provide some context
                    w = x2 - x1
                    h = y2 - y1
                    px = int(w * 0.2)
                    py = int(h * 0.2)
                    
                    x1 = max(0, x1 - px)
                    y1 = max(0, y1 - py)
                    x2 = min(frame.shape[1], x2 + px)
                    y2 = min(frame.shape[0], y2 + py)
                    
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size == 0:
                        continue
                        
                    # Save cropped image
                    img_name = f"hard_neg_frame{frame_count}_det{idx}.jpg"
                    img_path = os.path.join(images_dir, img_name)
                    cv2.imwrite(img_path, crop)
                    
                    # Save empty label file (YOLO format for background)
                    txt_name = f"hard_neg_frame{frame_count}_det{idx}.txt"
                    txt_path = os.path.join(labels_dir, txt_name)
                    with open(txt_path, "w") as f:
                        pass # Empty file tells YOLO "no weapons in this image"
                        
                    saved_count += 1

    cap.release()
    print(f"\nDone! Extracted {saved_count} hard negative crops.")
    print(f"Saved images to: {images_dir}")
    print(f"Saved empty labels to: {labels_dir}")

if __name__ == "__main__":
    main()
