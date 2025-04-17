import cv2
from ultralytics import YOLO

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy

# Open the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame)

    # Draw bounding boxes and labels
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            label = result.names[int(box.cls[0])]  # Object class name

            # Only show specific objects
            allowed_objects = ["laptop", "cell phone", "book"]  # Customize as needed
            if label in allowed_objects:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
