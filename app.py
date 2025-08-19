import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from sklearn.cluster import KMeans

# -----------------------------
# Load YOLO models (cached)
# -----------------------------
@st.cache_resource
def load_models():
    plate_model = YOLO("runs/detect/license_plate_detector/weights/best.pt")
    vehicle_model = YOLO("yolov8n.pt")
    reader = easyocr.Reader(['en'])
    return plate_model, vehicle_model, reader

plate_model, vehicle_model, reader = load_models()

# -----------------------------
# Color Prediction
# -----------------------------
def predict_color_kmeans(crop_img, k=3):
    img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    img_flat = img.reshape((-1, 3))
    img_flat = img_flat[np.mean(img_flat, axis=1) > 40]  # remove dark pixels
    if len(img_flat) == 0:
        return "unknown"

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(img_flat)
    counts = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]

    color_map = {
        'white': [245,245,245], 'black': [0,0,0], 'red': [200,20,20],
        'blue': [20,100,220], 'gray': [128,128,128], 'silver': [192,192,192],
        'yellow': [240,240,50], 'green': [50,160,50]
    }
    min_dist = float('inf')
    color_pred = 'unknown'
    for color, rgb in color_map.items():
        dist = np.linalg.norm(dominant_color - np.array(rgb))
        if dist < min_dist:
            min_dist = dist
            color_pred = color
    return color_pred

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Vehicle & License Plate Detection", layout="wide")
st.title("🚗 Vehicle & License Plate Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    # Convert uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    annotated_img = image.copy()

    vehicle_info_list = []

    # --- Detect vehicles ---
    vehicle_results = vehicle_model(image)
    for vbox in vehicle_results[0].boxes:
        cls_id = int(vbox.cls[0])
        cls_name = vehicle_model.names[cls_id]

        if cls_name in ['car','bus','truck','motorbike']:
            x1, y1, x2, y2 = map(int, vbox.xyxy[0])
            vehicle_crop = image[y1:y2, x1:x2]
            vehicle_color = predict_color_kmeans(vehicle_crop)

            cv2.rectangle(annotated_img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(annotated_img, f"{cls_name}, {vehicle_color}", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            vehicle_info_list.append({
                "type": cls_name,
                "color": vehicle_color,
                "plate": "Unknown"
            })

    # --- Detect plates + OCR ---
    plate_results = plate_model(image)
    for i, pbox in enumerate(plate_results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, pbox)
        plate_crop = image[y1:y2, x1:x2]
        result = reader.readtext(plate_crop)
        plate_number = result[0][1] if result else "Unknown"

        cv2.rectangle(annotated_img, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(annotated_img, plate_number, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        if i < len(vehicle_info_list):
            vehicle_info_list[i]["plate"] = plate_number

    # --- Display ---
    st.subheader("📸 Processed Image")
    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("📋 Detected Vehicles")
    for v in vehicle_info_list:
        st.write(f"- **Type:** {v['type']} | **Color:** {v['color']} | **Plate:** {v['plate']}")
