import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize Face Mesh
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        res, frame = cap.read()
        if not res:
            print("Failed to read frame.")
            break

        # Convert the color space from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check if any face landmarks are detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
                )

                h, w, _ = image.shape

                # Get coordinates of key facial landmarks
                def get_coordinates(landmark):
                    return int(landmark.x * w), int(landmark.y * h)

                lip_top = get_coordinates(face_landmarks.landmark[13])
                lip_bottom = get_coordinates(face_landmarks.landmark[14])
                lip_left = get_coordinates(face_landmarks.landmark[61])
                lip_right = get_coordinates(face_landmarks.landmark[291])
                brow_left = get_coordinates(face_landmarks.landmark[55])
                brow_right = get_coordinates(face_landmarks.landmark[285])
                nose_tip = get_coordinates(face_landmarks.landmark[1])

                # Calculate distances
                mouth_open_distance = np.linalg.norm(np.array(lip_top) - np.array(lip_bottom))
                mouth_width = np.linalg.norm(np.array(lip_left) - np.array(lip_right))
                brow_distance = np.linalg.norm(np.array(brow_left) - np.array(brow_right))
                nose_to_brow = np.linalg.norm(np.array(nose_tip) - np.array(brow_left))

                # Determine facial expression
                emotion = "Neutral"
                if mouth_open_distance > 0.06 * h and mouth_width > 0.25 * w:
                    emotion = "Smile"
                elif brow_distance < 0.15 * w and mouth_open_distance < 0.04 * h:
                    emotion = "Frown"
                elif mouth_open_distance > 0.09 * h:
                    emotion = "Surprised"

                # Draw a rectangle and display the emotion
                cv2.rectangle(image, (10, 10), (280, 60), (0, 0, 0), -1)
                cv2.putText(image, f"Emotion: {emotion}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the image
        cv2.imshow("MediaPipe Face Mesh - Emotion Detection", image)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
