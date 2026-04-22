import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort

# Page config
st.set_page_config(page_title="Cattle Disease Detection", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .title {text-align: center; font-size: 40px; font-weight: bold; color: #2c3e50;}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">🐄 Cattle Disease Detection System</p>', unsafe_allow_html=True)

# Load ONNX model
@st.cache_resource
def load_model():
    session = ort.InferenceSession("model.onnx")
    return session

session = load_model()

# Get input name
input_name = session.get_inputs()[0].name

# Classes (correct order)
classes = ['foot-and-mouth', 'healthy', 'lumpy']

# Layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, caption="Uploaded Image", use_column_width=True)

with col2:
    if uploaded_file is not None:
        st.markdown("### 🔍 Prediction Results")

        # Preprocess (same as training)
        img_resized = cv2.resize(img, (224, 224))
        img_resized = img_resized / 255.0
        img_resized = img_resized.astype(np.float32)
        img_resized = np.expand_dims(img_resized, axis=0)

        # ONNX Prediction
        pred = session.run(None, {input_name: img_resized})[0]

        pred_class = classes[np.argmax(pred)]
        confidence = float(np.max(pred))

        # Status box
        if pred_class == "healthy":
            st.success(f"✅ Healthy ({confidence:.2f})")
        else:
            st.error(f"⚠️ {pred_class.upper()} detected ({confidence:.2f})")

        # Progress bars
        st.markdown("### 📊 Confidence Scores")
        for i in range(len(classes)):
            st.write(classes[i])
            st.progress(float(pred[0][i]))

        # Extra info
        st.markdown("### 🧠 Model Insight")
        st.write(
            "The model analyzes visual patterns such as lesions, texture, and color variations to classify diseases."
        )