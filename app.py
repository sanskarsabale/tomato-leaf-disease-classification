import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(page_title="Tomato Leaf Disease Classifier", page_icon="🍅")

CONF_THRESHOLD = 0.70

DISEASE_INFO = {
    "Bacterial_spot": "Small dark, water-soaked spots on leaves, often spreading in warm, wet weather.",
    "Early_blight": "Brown spots with concentric rings, usually starting on older leaves.",
    "Late_blight": "Large dark, water-soaked patches; spreads fast in cool, humid conditions.",
    "Leaf_Mold": "Pale yellow patches on top of leaves with mold growth underneath.",
    "Septoria_leaf_spot": "Many small circular spots with grey centres and dark edges.",
    "Spider_mites Two-spotted_spider_mite": "Tiny yellow speckles and fine webbing caused by mites.",
    "Target_Spot": "Brown spots with ring patterns, sometimes confused with early blight.",
    "Tomato_Yellow_Leaf_Curl_Virus": "Yellowing and upward curling of leaves, spread by whiteflies.",
    "Tomato_mosaic_virus": "Mottled light and dark green patterns and distorted leaves.",
    "healthy": "No disease pattern detected by the model.",
}

@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("models/tomato_resnet50_final.keras")
    with open("models/class_names.json") as f:
        classes = json.load(f)
    return model, classes

model, class_names = load_model_and_classes()

st.title("🍅 Tomato Leaf Disease Classifier")
st.write("Upload a clear photo of a single tomato leaf. Works best with a plain background.")

file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    arr = np.array(img.resize((224, 224)), dtype="float32")
    arr = preprocess_input(np.expand_dims(arr, axis=0))   # same as training
    probs = model.predict(arr, verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]

    best = class_names[top[0]].replace("Tomato___", "")
    conf = probs[top[0]]

    st.subheader(f"Prediction: {best.replace('_', ' ')}")
    st.metric("Confidence", f"{conf*100:.1f}%")
    st.info(DISEASE_INFO.get(best, "No description available."))

    if conf < CONF_THRESHOLD:
        st.warning("Low confidence. The image may not be a clear tomato leaf, "
                   "or the condition may be outside what the model was trained on.")

    st.write("**Top 3 predictions**")
    for i in top:
        st.write(f"{class_names[i].replace('Tomato___', '').replace('_', ' ')}: {probs[i]*100:.1f}%")