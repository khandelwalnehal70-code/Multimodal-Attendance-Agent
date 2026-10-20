import streamlit as st
import cv2
import numpy as np
import time
import os
import urllib.request

# Page configuration
st.set_page_config(page_title="Multimodal Attendance & Engagement Agent", layout="wide")

st.title("🎓 Multimodal Attendance & Engagement Agent")
st.markdown("Real-time classroom attendance tracking and engagement analytics dashboard[span_3](start_span)[span_3](end_span)[span_4](start_span)[span_4](end_span).")

# Ensure cascade file exists
xml_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(xml_path):
    url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    urllib.request.urlretrieve(url, xml_path)

face_cascade = cv2.CascadeClassifier(xml_path)

# Sidebar controls
st.sidebar.header("Agent Controls")
run_camera = st.sidebar.checkbox("Start Live Session / Webcam", value=False)
presence_threshold = st.sidebar.slider("Attendance Threshold (seconds)", 2, 15, 5)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Classroom Feed & Vision Tracking")
    video_placeholder = st.empty()

with col2:
    st.subheader("Real-time Analytics & Suggestions")
    attendance_placeholder = st.empty()
    suggestion_placeholder = st.empty()

if run_camera:
    cap = cv2.VideoCapture(0)
    attendance_log = {}
    
    while run_camera and cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.warning("Failed to access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        current_time = time.time()

        present_students = []
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Student {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            person_key = f"Student_{i + 1}"
            if person_key not in attendance_log:
                attendance_log[person_key] = {"first_seen": current_time, "marked": False}
            
            elapsed = current_time - attendance_log[person_key]["first_seen"]
            if elapsed > presence_threshold:
                attendance_log[person_key]["marked"] = True

            if attendance_log[person_key]["marked"]:
                present_students.append(person_key)

        # Convert frame color for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update Live Attendance UI
        with attendance_placeholder.container():
            st.write(f"Total Faces Detected: {len(faces)}")
            st.write("Marked Present:", present_students if present_students else "Tracking presence...")

        # AI Teaching Suggestions based on activity
        with suggestion_placeholder.container():
            if len(faces) == 0:
                st.error("⚠️ No students detected in frame! Check camera angle.")
            elif len(present_students) > 0:
                st.success("💡 AI Suggestion: Engagement looks stable. Continue with current lesson plan.")
            else:
                st.info("ℹ️ AI Suggestion: Waiting for students to meet presence duration threshold...")

        # Small delay to control frame rate
        time.sleep(0.03)

    cap.release()
else:
    st.info("Check 'Start Live Session / Webcam' in the sidebar to launch the system.")