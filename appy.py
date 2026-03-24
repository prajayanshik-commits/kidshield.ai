import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import time

st.set_page_config(page_title="TinyWatch: Smart Scan", layout="wide")

# ── Session state init ──────────────────────────────────────────────
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "locked" not in st.session_state:
    st.session_state.locked = False

# ── 30-Minute Lock Check (runs BEFORE anything else) ────────────────
elapsed_minutes = (time.time() - st.session_state.start_time) / 60
if elapsed_minutes >= 30:
    st.session_state.locked = True

if st.session_state.locked:
    st.markdown("""
        <div style="background:#1a1a1a;padding:60px 20px;border-radius:12px;text-align:center;margin-top:40px;">
            <h1 style="color:red;font-size:3rem;">🔒 SCREEN LOCKED</h1>
            <h3 style="color:#ccc;">30-Minute Limit Reached</h3>
            <p style="color:#888;">Please take a break and come back later.</p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Header ──────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;'>🛡️ TinyWatch: Smart Scan</h1>", unsafe_allow_html=True)

# Timer display
mins_left = max(0, 30 - int(elapsed_minutes))
st.markdown(
    f"<p style='text-align:center;color:gray;font-size:14px;'>⏱️ Screen time remaining: <b>{mins_left} min</b></p>",
    unsafe_allow_html=True
)

# ── Speaker Function via Web Speech API (works in all browsers) ─────
def speak(text: str):
    """Trigger browser TTS — this is the ONLY reliable way to play audio from Streamlit."""
    safe = text.replace("'", "\\'")
    components.html(f"""
        <script>
            (function() {{
                window.speechSynthesis.cancel();
                var msg = new SpeechSynthesisUtterance('{safe}');
                msg.lang = 'en-US';
                msg.rate = 0.9;
                msg.pitch = 1.0;
                msg.volume = 1.0;
                window.speechSynthesis.speak(msg);
            }})();
        </script>
    """, height=0)

# ── Camera Input — works on mobile and desktop ──────────────────────
st.markdown("#### 📷 Take or upload a photo to scan")

# st.camera_input works on mobile browsers (Chrome/Safari) as of Streamlit 1.18+
# It opens the front camera on mobile automatically
img_file = st.camera_input(" ", label_visibility="collapsed")

if img_file is None:
    # Fallback: file uploader for devices where camera_input fails
    img_file = st.file_uploader(
        "Or upload a photo from your gallery",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )

# ── Face Detection & Logic ───────────────────────────────────────────
if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not read image. Please try again.")
        st.stop()

    height, width = img.shape[:2]

    # Load Haar cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # scaleFactor=1.1, minNeighbors=5 gives fewer false positives
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        st.warning("🔍 No face detected. Please look directly at the camera in good lighting.")
    else:
        # Use the largest detected face
        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest

        face_area_ratio = (w * h) / (width * height)

        # Draw bounding box on image for feedback
        color_bgr = (0, 200, 80) if face_area_ratio > 0.18 else (0, 120, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color_bgr, 3)

        # Show annotated image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, channels="RGB", use_column_width=True)

        st.markdown("---")

        if face_area_ratio > 0.18:
            # ── CHILD / TOO CLOSE ────────────────────────────────────
            st.markdown(
                "<h2 style='color:#28a745;text-align:center;'>✅ CHILD DETECTED</h2>",
                unsafe_allow_html=True
            )
            st.markdown("""
                <div style="background:#ffcccc;padding:20px;border-radius:10px;border:3px solid red;margin:10px 0;">
                    <h3 style='color:red;text-align:center;margin:0;'>
                        🔊 Please sit back — you are too close to the screen!
                    </h3>
                </div>
            """, unsafe_allow_html=True)

            # ✅ FIX: Browser TTS — this actually works unlike hardware speaker calls
            speak("Please sit back. You are too close to the screen!")

            st.markdown("---")
            st.subheader("📺 Safe Kids Content")
            col1, col2 = st.columns(2)
            with col1:
                st.video("https://www.youtube.com/watch?v=hq3yfQnllfQ")
            with col2:
                st.video("https://www.youtube.com/watch?v=71h8MZshGSs")

        else:
            # ── ADULT / SAFE DISTANCE ────────────────────────────────
            st.markdown(
                "<h2 style='color:#007bff;text-align:center;'>👤 ADULT DETECTED</h2>",
                unsafe_allow_html=True
            )
            st.success("✅ Adult access granted. Safe viewing distance confirmed.")
            st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

else:
    st.info("👆 Use the camera button above to scan, or upload a photo from your gallery.")

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:12px;'>TinyWatch MVP — protecting young eyes, one scan at a time 🛡️</p>",
    unsafe_allow_html=True
)
