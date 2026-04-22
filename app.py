import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Cattle Disease Detection", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #2c3e50;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<p class="title">🐄 Cattle Disease Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered diagnosis of cattle diseases</p>', unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    path = os.path.join(os.getcwd(), "model.onnx")
    return ort.InferenceSession(path)

session = load_model()
input_name = session.get_inputs()[0].name

classes = ['foot-and-mouth', 'healthy', 'lumpy']

# ---------- LAYOUT ----------
col1, col2 = st.columns([1,1])

# ---------- LEFT SIDE (UPLOAD) ----------
with col1:
    st.markdown("### 📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, caption="Uploaded Image", use_column_width=True)

# ---------- RIGHT SIDE (RESULTS) ----------
with col2:
    if uploaded_file is not None:

        st.markdown("### 🔍 Prediction Result")

        # Preprocess
        img_resized = cv2.resize(img, (224, 224))
        img_resized = img_resized / 255.0
        img_resized = img_resized.astype(np.float32)
        img_resized = np.expand_dims(img_resized, axis=0)

        # Prediction with loader
        with st.spinner("Analyzing image..."):
            pred = session.run(None, {input_name: img_resized})[0]

        pred_class = classes[np.argmax(pred)]
        confidence = float(np.max(pred))

        # ---------- RESULT BOX ----------
        if pred_class == "healthy":
            st.success(f"✅ Healthy\n\nConfidence: {confidence:.2f}")
        else:
            st.error(f"⚠️ {pred_class.upper()} detected\n\nConfidence: {confidence:.2f}")

        # ---------- PROGRESS BARS ----------
        st.markdown("### 📊 Confidence Scores")

        for i in range(len(classes)):
            st.write(classes[i])
            st.progress(float(pred[0][i]))

        # ---------- INFO ----------
        st.markdown("### 🧠 Model Insight")
        st.info(
            "This model analyzes texture, lesions, and color patterns in cattle images "
            "to classify diseases using deep learning."
        )

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<center>Made with ❤️ using Streamlit | AI Project</center>",
    unsafe_allow_html=True
)