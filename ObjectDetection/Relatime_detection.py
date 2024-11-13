from ultralytics import YOLO
import cv2 

# Train the YOLOv8 model
model = YOLO("yolo11x-pose.pt", task="pose")

# Start the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 

    # Run YOLOv8 model on the frame
    results = model(frame)

    # Plot detections on the frame
    annotated_frame = results[0].plot()  # Plot annotations from results

    # Display the frame with detections
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
