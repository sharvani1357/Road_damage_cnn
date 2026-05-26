import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import json
import plotly.express as px

from PIL import Image

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🚧",
    layout="wide"
)

# ==========================================
# LOAD CSS
# ==========================================

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(
    "road_damage_cnn_model.h5"
)

# ==========================================
# LOAD LABELS
# ==========================================

with open("label_mapping.json") as file:
    class_names = json.load(file)

# ==========================================
# HEADER SECTION
# ==========================================

st.markdown("""
# 🚧 AI-Based Road Damage Detection System
### Smart City Infrastructure Monitoring using CNN
""")

st.markdown("---")

# ==========================================
# ABOUT PROJECT
# ==========================================

st.markdown("""
<div class="about-card">

## 📘 About the Project

Road damage monitoring is essential for:
- reducing accidents
- improving transportation safety
- enabling faster infrastructure maintenance

This system uses:
- Convolutional Neural Networks (CNN)
- Computer Vision
- Deep Learning

to automatically analyze road images and detect:
- potholes
- cracks
- damaged surfaces

### 🌍 Industry Applications
- Smart Cities
- Highway Monitoring
- Municipal Infrastructure
- Autonomous Vehicles
- Road Safety Systems

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🚧 AI Monitoring Panel")

st.sidebar.success("System Status: ACTIVE")

st.sidebar.info("CNN Infrastructure Monitoring")

st.sidebar.markdown("---")

st.sidebar.write("TensorFlow + Streamlit")

# ==========================================
# MAIN LAYOUT
# ==========================================

left_col, right_col = st.columns([1,1])

# ==========================================
# IMAGE UPLOAD
# ==========================================

with left_col:

    st.subheader("📤 Upload Road Image")

    uploaded_file = st.file_uploader(
        "Upload JPG / PNG Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Road Image",
            use_column_width=True
        )

# ==========================================
# PREDICTION AREA
# ==========================================

with right_col:

    st.subheader("🧠 Prediction Dashboard")

    if uploaded_file is not None:

        # ----------------------------------
        # IMAGE PROCESSING
        # ----------------------------------

        img = np.array(image)

        img = cv2.resize(
            img,
            (128,128)
        )

        img = img / 255.0

        img = np.expand_dims(
            img,
            axis=0
        )

        # ----------------------------------
        # PREDICTION
        # ----------------------------------

        prediction = model.predict(img)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction)

        predicted_label = class_names[
            str(predicted_class)
        ]

        # ----------------------------------
        # SEVERITY LEVEL
        # ----------------------------------

        if confidence > 0.80:
            severity = "High"

        elif confidence > 0.50:
            severity = "Medium"

        else:
            severity = "Low"

        # ----------------------------------
        # DYNAMIC ROAD HEALTH
        # ----------------------------------

        road_health = int((1 - confidence) * 100)

        # ----------------------------------
        # RESULT CARD
        # ----------------------------------

        st.markdown(f"""
        <div class="prediction-card">

        <h2>🚧 Prediction: {predicted_label}</h2>

        <h3>🎯 Confidence: {confidence*100:.2f}%</h3>

        <h3>⚠️ Severity Level: {severity}</h3>

        <h3>🛣️ Road Health Score: {road_health}%</h3>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================================
        # PROBABILITY CHART
        # ==================================

        probs = prediction[0]

        df = pd.DataFrame({

            "Class": list(class_names.values()),

            "Probability": probs

        })

        fig = px.bar(

            df,

            x="Class",

            y="Probability",

            color="Probability",

            title="Class Confidence Graph"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==========================================
# RECOMMENDATIONS
# ==========================================

if uploaded_file is not None:

    st.markdown("---")

    st.markdown("""
    <div class="recommend-card">

    ## 🛠️ Recommendations

    ### ⚠️ Safety Warning
    Damaged road conditions may increase accident risk.

    ### 🚧 Repair Recommendation
    Immediate maintenance recommended for high severity damages.

    ### 🏙️ Smart City Insight
    AI-based infrastructure monitoring helps prioritize road repair efficiently.

    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER METRICS
# ==========================================

st.markdown("---")

st.subheader("📊 AI System Analytics")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Model Accuracy",
        "22%"
    )

with m2:
    st.metric(
        "Precision",
        "11%"
    )

with m3:
    st.metric(
        "Recall",
        "22%"
    )

with m4:
    st.metric(
        "F1 Score",
        "12%"
    )

st.markdown("---")

st.success("✅ Smart City AI Monitoring System Running")