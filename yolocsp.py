import cv2
import numpy as np
from ultralytics import YOLO

def load_yolo_model(model_path):
    """
    Load the YOLOv8-CSP model.
    Args:
        model_path (str): Path to the trained YOLO model.
    Returns:
        model: Loaded YOLO model.
    """
    model = YOLO(model_path)
    return model

def process_frame(frame, model, confidence_threshold=0.5):
    """
    Process a single frame for gesture recognition.
    Args:
        frame (numpy.ndarray): The input video frame.
        model: YOLO model for inference.
        confidence_threshold (float): Confidence threshold for detection.
    Returns:
        numpy.ndarray: Processed frame with detections.
    """
    # Convert the frame to RGB for YOLO processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform inference
    results = model.predict(rgb_frame)
    
    # Draw detections on the frame
    for result in results[0].boxes:
        conf = result.confidence
        if conf > confidence_threshold:
            x1, y1, x2, y2 = map(int, result.xyxy)
            label = f"{result.label} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main():
    """
    Main function for real-time PSL gesture recognition.
    """
    # Path to your trained YOLO model
    model_path = "best.pt"  # Replace with your YOLOv8-CSP model path
    
    # Load the YOLO model
    model = load_yolo_model(model_path)
    
    # Open a video capture stream
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to exit.")
    
    while True:
        # Capture a frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Process the frame
        processed_frame = process_frame(frame, model)
        
        # Display the processed frame
        cv2.imshow("PSL Gesture Recognition", processed_frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
