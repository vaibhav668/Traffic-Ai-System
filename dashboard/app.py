import streamlit as st
import requests
import time
BACKEND_URL = "http://127.0.0.1:8000"
def alert_card(title, status):
    color = "#ff4b4b" if status else "#2ecc71"
    text = "DETECTED" if status else "NORMAL"

    st.markdown(f"""
    <div style="
        background-color: {color};
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    ">
        {title}<br>{text}
    </div>
    """, unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.title("🕵️‍♂️INTELLIGUARD SURVEILLANCE ")

# ================= CONTROLS =================
st.sidebar.header("Controls")

if st.sidebar.button("Start Webcam"):
    requests.post(f"{BACKEND_URL}/set_webcam")

uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi"])

if uploaded_file is not None:
    files = {"file": uploaded_file}
    requests.post(f"{BACKEND_URL}/upload_video", files=files)

if st.sidebar.button("Stop"):
    requests.post(f"{BACKEND_URL}/stop")

# ================= LAYOUT =================
col1, col2 = st.columns([3, 1])

video_placeholder = col1.empty()
alerts_placeholder = col2.empty()

# ================= MAIN LOOP =================
import cv2
import numpy as np
import time

# persistent placeholders
video_placeholder = col1.empty()
alerts_placeholder = col2.empty()

while True:
    try:
        # -------- FRAME --------
        response = requests.get(f"{BACKEND_URL}/frame", timeout=1)
        img_bytes = response.content

        frame = cv2.imdecode(
            np.frombuffer(img_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        video_placeholder.image(
        frame,
        channels="BGR",
        width="stretch"
        )

        # -------- ALERTS --------
        alerts = requests.get(f"{BACKEND_URL}/alerts", timeout=1).json()

        with alerts_placeholder.container():
            st.subheader("🚨 Alerts")

            alert_card("🔥 Fire", alerts["fire"])
            alert_card("🚗 Accident", alerts["accident"])
            alert_card("🥊 Fight", alerts["fight"])

            st.subheader("📊 Stats")
            st.metric("Vehicle Count", alerts["vehicle_count"])

        time.sleep(0.03)

    except Exception as e:
        st.warning("Waiting for backend...")
        time.sleep(1)