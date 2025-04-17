import cv2
import numpy as np
import os
import imutils


# Set paths to model files
prototxt_path = r"D:\cu 1sem projects\AI detect\AI detect\deploy.prototxt"
model_path = r"D:\cu 1sem projects\AI detect\AI detect\mobilenet_iter_73000.caffemodel"




# Check if files exist
if not os.path.exists(prototxt_path):
    print("[ERROR] deploy.prototxt is missing! Please check the file path.")
    exit()
if not os.path.exists(model_path):
    print("[ERROR] mobilenet_iter_73000.caffemodel is missing! Please check the file path.")
    exit()

print("[INFO] Model files found! Loading MobileNet SSD...")

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Define class labels MobileNet SSD can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "pen",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "notebook", "motorbike", "person", "pottedplant",
           "jar", "sofa", "train", "tvmonitor","smartphone"]

# Open laptop camera
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)  # Resize for better processing
    (h, w) = frame.shape[:2]  # Get frame dimensions

    # Convert frame to a blob (preprocessing for the deep learning model)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop through detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Get confidence score

        if confidence > 0.5:  # Only consider detections above 50% confidence
            idx = int(detections[0, 0, i, 1])  # Get class index
            label = CLASSES[idx]  # Get class name
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = f"{label}: {confidence*100:.2f}%"
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Object Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# Release camera and close window
cap.release()
cv2.destroyAllWindows()
