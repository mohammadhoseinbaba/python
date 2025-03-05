from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Run detection
    frame = results[0].plot()  # Draw bounding boxes

    cv2.imshow("YOLO Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
