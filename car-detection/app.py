import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import tempfile
import os

# ================= SETTINGS =================
YOLO_WEIGHTS = "yolo/yolov8s.pt"             # YOLO model
COLOR_MODEL_PATH = "model/blue_car_classifier.keras"  # Car color classifier
COLOR_CLASSES = ["blue", "not_blue"]         # only 2 classes
BLUE_CONF_THRESHOLD = 0.8                    # Confidence threshold for blue detection
MIN_BOX_SIZE = 50                             # Ignore tiny detections
# ============================================

st.title("🚗 People & Car Color Detection (Blue / Not Blue)")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
process_button = st.button("Process Video")

if uploaded_file and process_button:
    with st.spinner("Loading models..."):
        # Load color model
        color_model = load_model(COLOR_MODEL_PATH)
        model_input_shape = tuple(color_model.input_shape[1:3])

        # Load YOLO model
        yolo_model = YOLO(YOLO_WEIGHTS)

    # Save uploaded video to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    input_path = tfile.name

    # Prepare output temp file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    # OpenCV video setup
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Error: Could not open uploaded video.")
    else:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_people = 0
        total_cars = 0

        progress_bar = st.progress(0)
        frame_placeholder = st.empty()  # For live preview
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_no = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame, verbose=False)[0]
            frame_people = 0
            frame_cars = 0

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                # ---------- People: count only ----------
                if cls_id == 0:  # Person
                    frame_people += 1

                # ---------- Cars ----------
                elif cls_id == 2 and w > MIN_BOX_SIZE and h > MIN_BOX_SIZE:  
                    frame_cars += 1
                    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                        continue

                    car_crop = frame[y1:y2, x1:x2]
                    if car_crop.size > 0:
                        # Convert BGR → RGB
                        car_img = cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB)
                        car_img = cv2.resize(car_img, model_input_shape)
                        car_img = car_img.astype("float32") / 255.0
                        car_img = np.expand_dims(car_img, axis=0)

                        try:
                            pred = color_model.predict(car_img, verbose=0)
                            # Binary classifier (sigmoid)
                            if pred.shape[-1] == 1:
                                color_label = "blue" if pred[0][0] < BLUE_CONF_THRESHOLD else "not_blue"
                            else:  # Softmax
                                idx = np.argmax(pred)
                                prob = pred[0][idx]
                                color_label = "blue" if idx == 0 and prob > BLUE_CONF_THRESHOLD else "not_blue"
                        except Exception as e:
                            print(f"[WARNING] Prediction failed: {e}")
                            continue

                        # Box color: red for blue cars, blue for not_blue
                        box_color = (0, 0, 255) if color_label == "blue" else (255, 0, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(frame, f"{color_label} {conf:.2f}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            total_people += frame_people
            total_cars += frame_cars

            # Display counters only
            cv2.putText(frame, f"People: {frame_people} (Total: {total_people})", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Cars: {frame_cars} (Total: {total_cars})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            out.write(frame)

            # Convert frame for live display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            frame_no += 1
            progress_bar.progress(frame_no / frame_count)

        cap.release()
        out.release()

        st.success("Processing complete!")
        st.video(output_path)
        st.download_button("Download Processed Video", data=open(output_path, "rb").read(),
                           file_name="processed_video.mp4", mime="video/mp4")
