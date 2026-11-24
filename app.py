"""Run: streamlit run app.py -- --model outputs/mobilenet/best_model.keras"""
import json, sys, pathlib
import numpy as np, streamlit as st, tensorflow as tf
from PIL import Image

model_path = pathlib.Path(sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv
                          else "outputs/mobilenet/best_model.keras")
THRESHOLD = 0.60

@st.cache_resource
def load():
    return tf.keras.models.load_model(model_path), json.load(open(model_path.parent / "classes.json"))

st.title("♻️ Waste Classifier")
st.caption("Decision-support prototype. Check local recycling rules before sorting.")
model, classes = load()
f = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])
if f:
    img = Image.open(f).convert("RGB"); st.image(img, use_container_width=True)
    x = np.array(img.resize((224, 224)), dtype="float32")[None]
    p = model.predict(x, verbose=0)[0]; i = int(p.argmax())
    st.subheader(f"Prediction: {classes[i]} ({p[i]:.1%})")
    if p[i] < THRESHOLD:
        st.warning("Low confidence — please check this item manually.")
    st.bar_chart(dict(zip(classes, p.tolist())))
