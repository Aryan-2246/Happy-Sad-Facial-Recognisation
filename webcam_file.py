from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the trained emotion model (change to your model path if necessary)
model = load_model('C:/Users/Asus/Desktop/CPP/happy_sad_model.keras')

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels (adjust if you have different labels)
emotion_labels = ['Happy', 'Sad']

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face region
        face = gray[y:y+h, x:x+w]
        
        # Resize the face to the input size for the model
        face_resized = cv2.resize(face, (48, 48))
        
        # Normalize the face pixel values and reshape for the model input
        face_normalized = face_resized / 255.0
        face_input = face_normalized.reshape(1, 48, 48, 1)  # Shape: (1, 48, 48, 1)

        # Predict the emotion (Happy or Sad)
        prediction = model.predict(face_input)
        emotion = emotion_labels[np.argmax(prediction)]  # 'Happy' or 'Sad'

        # Draw bounding box and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with the bounding boxes and emotion labels
    cv2.imshow('Emotion Detection (Happy/Sad)', frame)

    # Break the loop when 'Q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

