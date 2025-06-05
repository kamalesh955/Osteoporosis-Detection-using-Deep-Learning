import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="OsteoVision - AI Bone Health Analyzer",
    layout="centered",
    page_icon="🧬"
)

# Constants
IMAGE_SIZE = 256
CLASS_NAMES = ['normal', 'osteoporosis']

# Load model
@st.cache_resource
def load_trained_model():
    return load_model('osteoporosis_model.h5')

model = load_trained_model()

# App header
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .title {
        font-size: 2.5em;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
    }
    .subtitle {
        font-size: 1.2em;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-card {
        padding: 20px;
        background-color: #eaf2f8;
        border-radius: 12px;
        border-left: 5px solid #2E86C1;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🧬 OsteoVision</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Radiograph Analyzer for Osteoporosis Detection</div>', unsafe_allow_html=True)

# Image preprocessing
def preprocess_image(uploaded_file):
    try:
        img = load_img(uploaded_file, target_size=(IMAGE_SIZE, IMAGE_SIZE), color_mode='rgb')
        img_array = img_to_array(img)  # No normalization
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"❌ Error loading image: {e}")
        st.stop()

# Upload interface
uploaded_file = st.file_uploader("📤 Upload your radiographic image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='📸 Uploaded Image', use_container_width=True)

    processed_image = preprocess_image(uploaded_file)
    predictions = model.predict(processed_image)

    pred_index = np.argmax(predictions[0])
    confidence = predictions[0][pred_index]
    predicted_label = CLASS_NAMES[pred_index].capitalize()

    # Display result card
    st.markdown(f"""
    <div class="result-card">
        <h4>🧠 Prediction Results</h4>
        <p><strong>Condition:</strong> {predicted_label}</p>
        <p><strong>Confidence:</strong> {confidence:.2%}</p>
    </div>
    """, unsafe_allow_html=True)

    # Probability chart
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, predictions[0], color=["#3498DB", "#E74C3C"])
    ax.set_ylabel("Probability")
    ax.set_title("Class Probability Distribution")
    st.pyplot(fig)

st.markdown("---")
st.caption("⚠️ Note: This provides automated analysis and intended to support — not replace — professional medical evaluation.")
