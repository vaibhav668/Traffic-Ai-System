# 🚨 Intelliguard-AI

**Intelliguard-AI** is a real-time intelligent surveillance system that leverages Computer Vision and Deep Learning to detect critical events such as **accidents, fights, and fire**, along with **multi-object tracking and vehicle speed estimation**.

The system is built with a **FastAPI backend** for high-performance inference and a **Streamlit frontend** for an interactive dashboard.

---

##  Key Features

###  1. Accident Detection

* Based on **vehicle tracking and motion analysis**
* Uses:

  * Speed estimation from frame-to-frame displacement
  * Sudden speed drop detection
  * Bounding box overlap (IoU-based collision logic)
* Buffered logic reduces false positives

---

###  2. Fight Detection

* Custom-trained **ResNet18 model (Transfer Learning)**
* Dataset: Two classes

  * `Violence`
  * `Non-Violence`
* Prediction based on confidence threshold and temporal smoothing

---

###  3. Fire & Smoke Detection

* Fine-tuned **YOLOv8n model**
* Custom dataset:

  * `Fire`
  * `Smoke`
* Real-time detection with confidence filtering and stability buffering

---

###  4. Multi-Object Tracking + Speed Estimation

* Uses **YOLOv8 tracking**
* Tracks multiple vehicles simultaneously
* Displays:

  * Unique ID
  * Estimated speed (px/sec)
* Enables accident detection logic

---

## 🖥️ System Overview

* **Frontend (Streamlit Dashboard)**

  * Webcam or video upload support
  * Live video stream display
  * Alert panel:

    * 🔥 Fire
    * 🥊 Fight
    * 🚗 Accident
    * 📊 Vehicle count

* **Backend (FastAPI)**

  * Handles:

    * Frame processing
    * Model inference
    * Alert generation
  * REST API endpoints for real-time communication

---

## ⚙️ Tech Stack

* **Backend:** FastAPI
* **Frontend:** Streamlit
* **Deep Learning:** PyTorch
* **Computer Vision:** OpenCV
* **Models:**

  * YOLOv8 (Detection + Tracking)
  * Custom YOLOv8 (Fire/Smoke)
  * ResNet18 (Fight Detection)

---

##  Project Structure

```
Traffic-Ai-System/
│
├── dashboard/              # Streamlit frontend
│   └── app.py
│
├── src/
│   ├── api/                # FastAPI backend
│   ├── events/             # Detection modules
│   └── main_pipeline.py    # Core processing logic
│
├── models/                 # Trained models (.pth, .pt)
├── data/                   # Training datasets
├── requirements.txt
├── docker-compose.yml
└── Dockerfile
```

---

##  How It Works

1. User selects:

   * Webcam OR
   * Upload video

2. Frames are sent to backend via API

3. Backend processes each frame:

   * Object detection + tracking
   * Fire detection
   * Fight classification
   * Accident logic

4. Processed frames + alerts returned to frontend

5. Streamlit dashboard displays:

   * Live annotated video
   * Alert indicators
   * Stats

---

## 🐳 Docker Deployment

This project is containerized using Docker:

```bash
docker-compose up --build
```

* Backend runs on: `http://localhost:8000`
* Frontend runs on: `http://localhost:8501`

---

## ⚠️ Known Limitations

* Streamlit uses **WebRTC**, which may:

  * Reduce FPS
  * Cause latency in real-time streaming
* CPU-based inference leads to:

  * Slower performance
  * Reduced detection consistency
* Dockerized version may not fully utilize webcam hardware

---

## 🎥 Demo

Demo videos showcasing detection capabilities are included in the repository:

* Accident detection
* Fight detection
* Fire detection

---

##  DockerHub

A Docker image has been pushed to DockerHub for deployment.
However, due to Streamlit + WebRTC limitations, performance may not be optimal.

---

##  Core Logic

The main pipeline includes:

* YOLO-based tracking
* Speed estimation using positional changes
* IoU-based collision detection
* Buffered event validation for stability



---

##  Future Improvements

* GPU acceleration (CUDA support)
* Replace Streamlit with React + WebSockets for better FPS
* Model optimization (quantization / TensorRT)
* Edge deployment (Jetson / Raspberry Pi)
* Improved datasets for robustness

---

## 🤝 Contribution

Contributions are welcome!
Feel free to open issues or submit pull requests.

---

##  Contact

**Vaibhav Pokhriyal**
vpokhriyal35@gmail.com
---

## ⭐ If you like this project, consider giving it a star!
