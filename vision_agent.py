import cv2
import time
import urllib.request
import os

# Automatically download the Haar Cascade face detector file if not present locally
xml_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(xml_path):
    url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    urllib.request.urlretrieve(url, xml_path)

face_cascade = cv2.CascadeClassifier(xml_path)

# Setup webcam (0 is the default built-in laptop camera)
cap = cv2.VideoCapture(0)

# Track attendance timers
attendance_log = {}
PRESENCE_THRESHOLD_SECONDS = 5  # Time required to mark attendance

print("Starting Vision Module... Press 'q' on your keyboard to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break

    # Convert frame to grayscale for fast face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    current_time = time.time()

    # Draw rectangles around detected faces and check presence duration
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Student {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        person_key = f"Student_{i + 1}"
        if person_key not in attendance_log:
            attendance_log[person_key] = {"first_seen": current_time, "marked": False}
        
        elapsed_time = current_time - attendance_log[person_key]["first_seen"]
        
        # Check threshold to mark attendance automatically
        if elapsed_time > PRESENCE_THRESHOLD_SECONDS and not attendance_log[person_key]["marked"]:
            attendance_log[person_key]["marked"] = True
            print(f"[ATTENDANCE MARKED] {person_key} is present!")

    # Display the live video feed window
    cv2.imshow('Multimodal Attendance - Vision Module', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()