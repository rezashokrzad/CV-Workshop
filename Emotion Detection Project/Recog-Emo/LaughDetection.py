from fer import FER
import cv2


cap = cv2.VideoCapture(0)
detector = FER(mtcnn=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    result = detector.detect_emotions(frame)
    if result:
        emotion = max(result[0]['emotions'], key=result[0]['emotions'].get)
        cv2.rectangle(frame, (0, 0), (300, 60), (0, 0, 0), -1)  # Black rectangle
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # نمایش تصویر
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
