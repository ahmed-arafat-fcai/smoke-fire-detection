# app.py — Combined: friend's UI + your detection functions (image, video, webcam + realtime)
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
from datetime import datetime
#cd "C:\Users\SAMSUNG\PycharmProjects\PythonProject"
#C:\Users\SAMSUNG\AppData\Local\Programs\Python\Python313\python.exe -m streamlit run app.py

# Real-time imports (webrtc)
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    import av
except Exception as e:
    # we will show a friendly error later if real-time is attempted without dependencies
    av = None
    VideoTransformerBase = None
    webrtc_streamer = None

# --- Page config ---
st.set_page_config(page_title="Smoke & Fire AI", layout="wide", page_icon="🔥")

# --- Friend's CSS (design & UI) ---
st.markdown("""
<style>
    :root {
        --primary-gradient: linear-gradient(45deg, #00BFFF, #8A2BE2);
        --accent-orange: #FF5722;
        --text-color: #f0f0f0;
        --bg-color: #0d0d21;
        --card-bg: #1a1e4a;
    }
    .stApp {
        background: var(--bg-color);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 3.0rem;
        font-weight: 800;
        text-align: center;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 10px rgba(138,43,226,0.4);
        margin-bottom: 6px;
    }
    .subheader {
        text-align: center;
        color: #B0C4DE;
        margin-bottom: 18px;
    }
    .stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        padding: 12px 26px;
        border-radius: 50px;
        font-size: 1rem;
        font-weight: bold;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(138,43,226,0.4);
    }
    .accent-btn > button {
        background: var(--accent-orange);
        color: white;
        box-shadow: 0 5px 15px rgba(255,87,34,0.4);
    }
    .accent-btn > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(255,87,34,0.6);
    }
    .feature-card {
        background-color: var(--card-bg);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }
    .result-box {
        background-color: var(--card-bg);
        border-radius: 10px;
        padding: 16px;
    }
    .smoke-text { color: #808080; }
    .fire-text { color: #FF4500; }
    .mix-text { color: #9370DB; }
    .safe-text { color: #32CD32; }
    #MainMenu, footer {
        visibility: hidden;
    }
    /* small responsive tweaks */
    @media (max-width: 640px) {
        .main-header { font-size: 2rem; }
    }
</style>
""", unsafe_allow_html=True)

# --- Model loader (your code) ---
@st.cache_resource
def load_model(weights_path: str = "best.pt"):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    return YOLO(weights_path)

# Load model with error handling
try:
    model = load_model("best.pt")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Class names
class_names = {0: "fire", 1: "smoke"}

# --- Utility helpers (your improved draw function) ---
CLASS_COLORS_RGB = {0: (255, 90, 31), 1: (14, 165, 164)}
DEFAULT_COLOR = (120, 120, 120)

def rgb_to_bgr(rgb):
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))

def draw_boxes_custom(img_bgr, boxes, names, conf_thres=0.25, padding=12, border_thickness=4, alpha=0.18):
    out = img_bgr.copy()
    if boxes is None or len(boxes) == 0:
        return out, [], [], []
    xyxy = boxes.xyxy.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int).flatten()
    confs_arr = boxes.conf.cpu().numpy().flatten()
    boxes_np, confs, classes = [], [], []
    h, w = out.shape[:2]
    for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, cls_ids, confs_arr):
        if conf < conf_thres:
            continue
        x1_p = max(0, int(x1) - padding)
        y1_p = max(0, int(y1) - padding)
        x2_p = min(w - 1, int(x2) + padding)
        y2_p = min(h - 1, int(y2) + padding)

        color_rgb = CLASS_COLORS_RGB.get(cls_id, DEFAULT_COLOR)
        color_bgr = rgb_to_bgr(color_rgb)

        overlay = out.copy()
        cv2.rectangle(overlay, (x1_p, y1_p), (x2_p, y2_p), color_bgr, -1)
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

        cv2.rectangle(out, (x1_p, y1_p), (x2_p, y2_p), color_bgr, border_thickness, lineType=cv2.LINE_AA)

        label = f"{names.get(int(cls_id), str(cls_id))} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        lx1 = x1_p
        ly1 = max(0, y1_p - th - 8)
        lx2 = x1_p + tw + 12
        ly2 = y1_p
        cv2.rectangle(out, (lx1, ly1), (lx2, ly2), color_bgr, -1)
        cv2.putText(out, label, (lx1 + 6, ly2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        boxes_np.append([x1_p, y1_p, x2_p, y2_p])
        confs.append(float(conf))
        classes.append(int(cls_id))
    return out, boxes_np, confs, classes

# Helper to convert BGR np.array to PIL image for Streamlit (RGB)
def bgr_to_pil(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

# --- UI: header / hero from friend ---
st.markdown("<div class='main-header'>Detect Smoke & Fire in Real Time</div>", unsafe_allow_html=True)
st.markdown("<h3 class='subheader'>Upload images, stream video, or connect your camera — get instant alerts.</h3>", unsafe_allow_html=True)

# CTA buttons
col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns([1, 1, 0.2, 1, 1])
with col_t2:
    st.button("Try Now", key="btn_try", help="Scroll to demo section", use_container_width=True)
with col_t4:
    st.button("Watch Demo", key="btn_watch", help="Watch demo video", use_container_width=True)

st.write("---")

# Feature cards
st.header("Key Features")
feature_cols = st.columns(4)
features = [
    ("Instant Detection", "🔍", "AI-powered analysis for faster-than-human speed."),
    ("Real-Time Alerts", "🔔", "Get instant notifications via email or SMS."),
    ("Image, Video & Camera Support", "🎥", "Compatible with various media formats and live feeds."),
    ("Responsive & Easy", "📱", "A clean interface that works great on all devices.")
]
for i, (title, icon, desc) in enumerate(features):
    with feature_cols[i]:
        st.markdown(f"""
        <div class='feature-card'>
            <div style='font-size: 2.2rem;'>{icon}</div>
            <h4 style="margin-bottom:6px;">{title}</h4>
            <p style="color:#cbd5e1;margin-top:6px;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.write("---")

# --- Demo / Input Section (uses your logic for files, camera, video) ---
st.header("Try the Model")
demo_options = st.radio(
    "Choose your input:",
    ('Upload Image', 'Upload Video', 'Live Camera Feed'),
    horizontal=True
)

# Ensure session history
if 'history' not in st.session_state:
    st.session_state.history = []

# Image branch
if demo_options == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img_bytes = uploaded_file.getvalue()
        nparr = np.frombuffer(img_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not read uploaded image.")
        else:
            with st.spinner("Running model..."):
                results = model(bgr, conf=0.25)
                r = results[0]
                out_bgr, boxes_np, confs, classes = draw_boxes_custom(bgr, r.boxes, class_names, conf_thres=0.25)
                pil_img = bgr_to_pil(out_bgr)

            result_col1, result_col2 = st.columns([2, 1])
            with result_col1:
                st.image(pil_img, use_container_width=True, caption="Results with bounding boxes")
            with result_col2:
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.subheader("Detection Results")
                if classes:
                    has_fire = 0 in classes
                    has_smoke = 1 in classes
                    if has_fire and has_smoke:
                        st.markdown(f"<p class='mix-text'><b>Result:</b> Smoke & Fire detected</p>", unsafe_allow_html=True)
                    elif has_fire:
                        st.markdown(f"<p class='fire-text'><b>Result:</b> Fire detected</p>", unsafe_allow_html=True)
                    elif has_smoke:
                        st.markdown(f"<p class='smoke-text'><b>Result:</b> Smoke detected</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p class='safe-text'><b>Result:</b> No Hazard Detected</p>", unsafe_allow_html=True)

                    st.write(f"Confidence(s): {', '.join(f'{c:.2f}' for c in confs)}")
                    st.write("**Detections:**")
                    for i, (cid, confv) in enumerate(zip(classes, confs)):
                        st.caption(f"{i+1}. {class_names.get(cid, str(cid))} — {confv:.2f}")
                else:
                    st.markdown(f"<p class='safe-text'><b>Result:</b> No Hazard Detected</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            thumb = cv2.resize(out_bgr, (240, 160))
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            st.session_state.history.append({'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'img_rgb': thumb_rgb})
            if len(st.session_state.history) > 12:
                st.session_state.history.pop(0)

# Video branch
elif demo_options == 'Upload Video':
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi", "mkv"])
    conf_slider = st.slider("Confidence threshold for video frames", 0.01, 1.0, 0.25, 0.01)
    imgsz_select = st.selectbox("Image size (imgsz)", [320, 416, 640, 800], index=2)
    if uploaded_video:
        st.warning("Video processing may take a while depending on length.")
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        pb = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = 0
        with st.spinner("Processing video frames..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, conf=conf_slider, imgsz=imgsz_select)
                ann = results[0].plot()
                # results[0].plot usually returns an RGB image (h, w, 3) in numpy
                if ann is not None:
                    # ensure we display RGB correctly
                    stframe.image(ann, use_container_width=True)
                idx += 1
                if total_frames:
                    pb.progress(min(1.0, idx/total_frames))
            cap.release()
        pb.empty()
        st.success("Video processing complete!")

# Live Camera branch (webcam, realtime with streamlit-webrtc)
elif demo_options == 'Live Camera Feed':
    st.write("Choose camera mode:")

    cam_mode = st.radio(
        "Camera Mode",
        ["📸 Snapshot Camera", "🎥 Real-time Stream"],
        horizontal=True,
        key="cam_mode"
    )

    # --- Snapshot Camera (old behavior) ---
    if cam_mode == "📸 Snapshot Camera":
        st.write("Take a snapshot from your webcam.")
        cam_file = st.camera_input("Take a photo")

        if cam_file is not None:
            file_bytes = np.asarray(bytearray(cam_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)

            st.write("Snapshot uploaded. Adjust parameters:")

            col1, col2 = st.columns(2)
            with col1:
                conf_snap = st.slider("Confidence", 0.01, 1.0, 0.35, 0.01, key="conf_snap")
            with col2:
                imgsz_snap = st.selectbox("Img size", [320, 416, 640], index=1, key="imgsz_snap")

            if st.button("Run detection", type="primary"):
                results = model(img_bgr, conf=conf_snap, imgsz=imgsz_snap)
                r = results[0]

                out_bgr, boxes, confs, classes = draw_boxes_custom(img_bgr, r.boxes, class_names, conf_thres=conf_snap)
                out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

                st.image(out_rgb, caption="Detection Result", use_container_width=True)

                # Save thumbnail in history
                thumb = cv2.resize(out_bgr, (160, 120))
                thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                st.session_state.history.append({
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'img_rgb': thumb_rgb
                })

    # --- Real-time WebRTC Stream ---
    else:
        st.write("Live camera stream — real-time detection.")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            live_conf = st.slider("Confidence", 0.01, 1.0, 0.35, 0.01, key="live_conf")
        with col2:
            live_imgsz = st.selectbox("Img size", [320, 416, 640], index=1, key="live_imgsz")
        with col3:
            process_every = st.number_input("Process every N frames", min_value=1, max_value=10, value=1, step=1)

        st.markdown("**Press Start / Stop to control the live stream.**")

        from streamlit_webrtc import VideoProcessorBase


        class VideoProcessor(VideoProcessorBase):
            def __init__(self, conf=0.35, imgsz=416, every_n=1):
                self.conf = conf
                self.imgsz = imgsz
                self.every_n = every_n
                self.frame_count = 0
                self.model = model

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img_bgr = frame.to_ndarray(format="bgr24")
                self.frame_count += 1

                if (self.frame_count % self.every_n) != 0:
                    return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

                try:
                    results = self.model(img_bgr, conf=self.conf, imgsz=self.imgsz)
                    r = results[0]
                    out_bgr, boxes, confs, classes = draw_boxes_custom(
                        img_bgr, r.boxes, class_names, conf_thres=self.conf
                    )
                    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
                    return av.VideoFrame.from_ndarray(out_rgb, format="rgb24")
                except Exception as e:
                    print("Error in processor.recv:", e)
                    return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


        webrtc_streamer(
            key="realtime-yolo",
            video_processor_factory=lambda: VideoProcessor(
                conf=live_conf, imgsz=live_imgsz, every_n=process_every
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

# --- Recent Runs gallery (bottom) ---
if st.session_state.history:
    st.write("---")
    st.subheader("Recent Runs")
    cols = st.columns(min(4, len(st.session_state.history)))
    for c, item in zip(cols, reversed(st.session_state.history)):
        with c:
            st.image(item['img_rgb'], use_container_width=False, width=180, caption=item['time'])

# --- Footer ---
st.write("---")
footer_col1, footer_col2 = st.columns([1, 1])
with footer_col1:
    st.markdown("© 2025 FireSmoke AI. All rights reserved.")
with footer_col2:
    st.markdown("Privacy Policy | Terms of Service | Contact Us")
