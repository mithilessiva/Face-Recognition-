import face_recognition
import cv2

# Load a sample picture and learn how to recognize it.
known_image = face_recognition.load_image_file("known.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]
known_face_names = ["Person 1"]

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]  # BGR to RGB

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces([known_encoding], face_encoding)

        name = "Unknown"
        if True in matches:
            name = known_face_names[0]

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Label
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
