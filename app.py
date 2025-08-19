import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from sklearn.cluster import KMeans
import base64
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# -----------------------------
# Streamlit page config (must be first)
# -----------------------------
st.set_page_config(page_title="Vehicle Recognition", layout="wide")

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():
    plate_model = YOLO("runs/detect/license_plate_detector2/weights/best.pt")
    vehicle_model = YOLO("yolov8n.pt")
    reader = easyocr.Reader(['en'], gpu=False)
    return plate_model, vehicle_model, reader

plate_model, vehicle_model, reader = load_models()

# -----------------------------
# Vehicle color prediction
# -----------------------------
def predict_color_kmeans(crop_img, k=5):
    if crop_img is None or crop_img.size == 0:
        return "unknown"
    h, w, _ = crop_img.shape
    crop = crop_img[h//5: 4*h//5, w//5: 4*w//5]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    img_flat = hsv.reshape((-1, 3))
    mask = (img_flat[:,1] > 40) & (img_flat[:,2] > 50) & (img_flat[:,2] < 240)
    img_flat = img_flat[mask]
    if len(img_flat) == 0:
        return "unknown"
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(img_flat)
    counts = np.bincount(kmeans.labels_)
    dominant_hsv = kmeans.cluster_centers_[np.argmax(counts)]
    color_map_hsv = {
        'white':[0,0,255],'black':[0,0,0],'gray':[0,0,128],'silver':[0,0,192],
        'red':[0,255,200],'orange':[15,255,200],'yellow':[30,255,200],
        'green':[60,255,200],'blue':[120,255,200],'brown':[20,150,100]
    }
    min_dist, color_pred = float("inf"), "unknown"
    for color, ref in color_map_hsv.items():
        dist = np.linalg.norm(dominant_hsv - np.array(ref))
        if dist < min_dist:
            min_dist, color_pred = dist, color
    return color_pred

# -----------------------------
# OCR helper
# -----------------------------
def ocr_plate(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    result = reader.readtext(thr, detail=0)
    if result:
        return max(result, key=len).upper().replace(" ", "")
    return "Unknown"

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚗 Vehicle & License Plate Recognition")
# Add horizontal line for separation
st.markdown("---")

uploaded_file = st.file_uploader("📂Upload an image", type=["jpg","jpeg","png"])
show_bbox = True

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    orig_img = image.copy()
    annotated_img = image.copy()
    h, w = image.shape[:2]

    vehicles = []

    # -----------------------------
    # Vehicle detection
    # -----------------------------
    vehicle_results = vehicle_model(image)
    for vbox in vehicle_results[0].boxes:
        if vbox.conf[0] < 0.5:
            continue
        cls_id = int(vbox.cls[0])
        cls_name = vehicle_model.names[cls_id]
        if cls_name not in ["car","bus","truck","motorbike"]:
            continue
        x1, y1, x2, y2 = map(int, vbox.xyxy[0].cpu().numpy())
        crop = image[y1:y2, x1:x2]
        color = predict_color_kmeans(crop)

        if show_bbox:
            cv2.rectangle(annotated_img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(annotated_img, f"{cls_name}, {color}",
                        (x1, max(20,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)

        vehicles.append({"type":cls_name,"color":color,"plate":"Unknown","bbox":(x1,y1,x2,y2)})

    # -----------------------------
    # Plate detection + OCR
    # -----------------------------
    plate_results = plate_model(image)
    for pbox in plate_results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, pbox)
        crop = image[y1:y2, x1:x2]
        plate_text = ocr_plate(crop)

        if show_bbox:
            cv2.rectangle(annotated_img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(annotated_img, plate_text,(x1,max(20,y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)

        # Assign plate to nearest vehicle (center distance fallback)
        pcx, pcy = (x1+x2)//2, (y1+y2)//2
        if vehicles:
            nearest_v = min(vehicles, key=lambda v: ((v["bbox"][0]+v["bbox"][2])//2 - pcx)**2 + ((v["bbox"][1]+v["bbox"][3])//2 - pcy)**2)
            nearest_v["plate"] = plate_text

    # -----------------------------
    # Summary metrics
    # -----------------------------
    # Expandable vehicle details
    # -----------------------------
    for i, v in enumerate(vehicles, 1):
        with st.expander(f"Vehicle {i} Details"):
            st.write(f"**Type:** {v['type']}")
            st.write(f"**Color:** {v['color']}")
            st.write(f"**Plate:** {v['plate']}")
            vx1, vy1, vx2, vy2 = v["bbox"]
            st.image(cv2.cvtColor(image[vy1:vy2, vx1:vx2], cv2.COLOR_BGR2RGB), use_container_width=True)
            if v['plate'] == "Unknown":
                st.warning("OCR failed to detect plate for this vehicle.")

    # -----------------------------
    # Side by side images
    # -----------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Annotated Image", use_container_width=True)

    # -----------------------------
    # Download button for annotated image
    # -----------------------------
    _, buffer = cv2.imencode(".jpg", annotated_img)
    b64 = base64.b64encode(buffer).decode()
    href_img = f'<a href="data:file/jpg;base64,{b64}" download="annotated_image.jpg">📥 Download Annotated Image</a>'
    st.markdown(href_img, unsafe_allow_html=True)

    # Add horizontal line for separation
    st.markdown("---")

    # -----------------------------

    # Summary metrics
    # -----------------------------
    st.subheader("📋 Detected Vehicles Summary")

    if len(vehicles) > 0:
        df = pd.DataFrame(vehicles)

        # Rename columns for better display
        df_display = df[['plate','type','color']].rename(columns={
            'plate': 'Number Plate',
            'type': 'Vehicle Type',
            'color': 'Colour of Vehicle'
        })

        # Styled DataFrame with colored header row & centered
        df_styled = df_display.style.set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#9b59b6'),
                                        ('color', 'white'),
                                        ('font-weight', 'bold'),
                                        ('text-align', 'center')]}
        ]).set_properties(**{'text-align': 'center'})  # center all cells

        # Convert to HTML and center the whole table
        st.markdown(
            f"<div style='display: flex; justify-content: center'>{df_styled.to_html(index=False, escape=False)}</div>",
            unsafe_allow_html=True
        )
        # Total vehicles
        # Instead of st.metric, use HTML with red value
        st.markdown(
            f"<h4>🔢Total Vehicles Detected: <span style='color:red'>{len(vehicles)} vehicles </span></h4>",
            unsafe_allow_html=True
        )

        # Vehicle counts
        type_counts = df['type'].value_counts()
        color_counts = df['color'].value_counts()

        col_type, col_color = st.columns(2)

        with col_type:
            st.markdown("<h4>Vehicle Counts by Type:</h4>",unsafe_allow_html=True)
            fig, ax = plt.subplots()
            type_counts.plot(kind="bar", color="green", ax=ax)
            ax.set_xlabel("Vehicle Type")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            st.markdown("**Top 3 Types:** " + ", ".join(type_counts.index[:3]))

        with col_color:
            st.markdown("<h4>Vehicle Counts by Color:</h4>",unsafe_allow_html=True)
            fig, ax = plt.subplots()
            color_counts.plot(kind="bar", color="blue", ax=ax)
            ax.set_xlabel("Vehicle Color")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            st.markdown("**Top 3 Colors:** " + ", ".join(color_counts.index[:3]))

    else:
        st.warning("⚠️ No vehicles detected in the uploaded image.")

    # Download CSV of vehicle info
    # -----------------------------
    csv = df.to_csv(index=False)
    b64_csv = base64.b64encode(csv.encode()).decode()
    href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="vehicle_data.csv">📥 Download Vehicle Data CSV</a>'
    st.markdown(href_csv, unsafe_allow_html=True)
    
    # Add horizontal line for separation
    st.markdown("---")

        # -----------------------------
    # About the models used
    # -----------------------------
    st.subheader("🎯Models Used:")
    st.markdown("""
    - **YOLOv8n (Ultralytics)** → Used for detecting vehicles such as *cars, buses, trucks, and motorbikes*.  
    - **Custom YOLO model** (license_plate_detector2) → Trained specifically to detect license plates.  
    - **EasyOCR** → Performs Optical Character Recognition (OCR) to read the license plate text.  
    - **KMeans (from scikit-learn)** → Clusters colors in the detected vehicle region to estimate the dominant vehicle color.  

    This system combines **object detection**, **OCR** and **color analysis** to provide a complete vehicle recognition pipeline.
    """)

