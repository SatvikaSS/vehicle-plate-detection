# 🚗 Vehicle and License Plate Detection App

A **Streamlit web app** that detects vehicles, identifies license plates, and extracts vehicle details such as **Number Plate, Vehicle Type, and Colour of Vehicle** using **YOLOv8**, **OpenCV**, and **EasyOCR**.

---

## ✨ Features
- Upload an image of a vehicle 🚘
- Detects and extracts:
  - ✅ Vehicle type (car, bike, truck, etc.)
  - ✅ Vehicle color
  - ✅ License plate number (OCR)
- Displays results in a clean, styled table
- Works directly in the browser (Streamlit)

---

## ⚙️ Tech Stack
- [Streamlit](https://streamlit.io/) – Frontend & UI
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) – Object detection
- [OpenCV](https://opencv.org/) – Image processing
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) – Optical Character Recognition
- [Pandas](https://pandas.pydata.org/) – Tabular display

---

## 🚀 Deployment
This project is deployed on **Streamlit Cloud**.  
👉 [[Click here to try the app]](https://vehicle-plate-detection-y4ecqun865sbzdcabdpjvh.streamlit.app/)(#)  <!-- Replace with your Streamlit Cloud URL -->

---

## 🖥️ Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/SatvikaSS/vehicle-plate-detection.git
cd vehicle-plate-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 📂 Project Structure
```
vehicle-plate-detection/
├── app.py
├── requirements.txt
├── README.md
├── models/                         
│   ├── yolov8n.pt
│   └── best.pt ##license_plate_detector.pt
```
---

## 🎯Models Used:
-YOLOv8n (Ultralytics) → Used for detecting vehicles such as cars, buses, trucks, and motorbikes.
-Custom YOLO model (license_plate_detector2) → Trained specifically to detect license plates.
-EasyOCR → Performs Optical Character Recognition (OCR) to read the license plate text.
-KMeans (from scikit-learn) → Clusters colors in the detected vehicle region to estimate the dominant vehicle color.
-This system combines object detection, OCR and color analysis to provide a complete vehicle recognition pipeline.

---

## 📝 Notes
- Uses **opencv-python-headless** for Streamlit Cloud compatibility.
- You can replace the YOLOv8 model with your own trained weights (place them in `models/`).
- Works with most common vehicle images.

---

## 🙌 Acknowledgements
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)  
- [Streamlit](https://streamlit.io/)  

---
