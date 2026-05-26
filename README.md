# 🚧 AI-Based Road Damage Detection System

Smart City Infrastructure Monitoring using CNN

---

## 📌 Project Overview

This project is a Deep Learning based Road Damage Detection System developed using Convolutional Neural Networks (CNN) and deployed with Streamlit Cloud.

The application helps automate road monitoring by analyzing uploaded road images and detecting road damages such as:
- potholes
- cracks
- damaged surfaces

This system can assist:
- smart city infrastructure teams
- municipal corporations
- highway monitoring authorities
- road safety departments

in prioritizing maintenance and improving transportation safety.

---

## 🎯 Problem Statement

Manual road inspections are:
- time-consuming
- expensive
- inefficient
- prone to delays

Delayed identification of potholes and cracks can increase:
- road accidents
- vehicle damage
- traffic risks

This project provides an AI-powered automated solution using Computer Vision and CNN.

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- CNN (Convolutional Neural Networks)
- OpenCV
- NumPy
- Pandas
- Plotly
- Streamlit

---

## 📂 Dataset

Dataset used:

Road Damage Dataset — Potholes, Cracks and Manholes

Source:
https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes

---

## ⚙️ Features

✅ Upload road images  
✅ AI-based damage prediction  
✅ Confidence score display  
✅ Severity level detection  
✅ Probability visualization graph  
✅ Smart city monitoring dashboard UI  
✅ CNN-based image classification  
✅ Real-time prediction system  
✅ Professional Streamlit deployment  

---

## 🧪 CNN Workflow

### 1. Dataset Understanding
- image analysis
- class identification
- sample visualization

### 2. Data Preprocessing
- image resizing
- normalization
- label encoding
- train/test split

### 3. Data Augmentation
- rotation
- zoom
- horizontal flip
- brightness adjustment

### 4. CNN Model Design
- Convolution Layers
- MaxPooling Layers
- Dropout Layers
- Dense Layers

### 5. Model Training
- TensorFlow/Keras training
- validation split
- multiple epochs

### 6. Model Evaluation
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### 7. Real-Time Prediction
- uploaded image prediction
- unseen road image testing

### 8. Deployment
- Streamlit Cloud deployment
- GitHub integration

---

## 📊 Model Evaluation Metrics

| Metric | Value |
|---|---|
| Accuracy | 22% |
| Precision | 11% |
| Recall | 22% |
| F1 Score | 12% |

> Note: Since the dataset was originally designed for object detection, classification accuracy is limited after conversion into a CNN classification problem.

---

## 🖼️ Application UI

The Streamlit dashboard includes:
- futuristic AI monitoring interface
- image upload section
- prediction dashboard
- confidence visualization
- smart recommendations
- severity analysis
- analytics cards

---

## 🌍 Industry Applications

- Smart City Infrastructure
- Road Safety Systems
- Highway Monitoring
- Municipal Maintenance
- Autonomous Vehicle Assistance
- Traffic Risk Detection

---

## 🚀 Live Deployment

🔗 Streamlit Application:

https://roaddamagecnn-j2jauln6zwtvbsnbtktugi.streamlit.app/

---

## 📁 Project Structure

```bash
Road_Damage_CNN_Project/
│
├── app.py
├── style.css
├── requirements.txt
├── runtime.txt
├── road_damage_cnn_model.h5
├── label_mapping.json
├── .gitignore
```

---

## ▶️ Run Locally

### 1. Clone Repository

```bash
git clone <repository-url>
```

### 2. Create Virtual Environment

```bash
py -3.11 -m venv venv
```

### 3. Activate Environment

```bash
venv\Scripts\activate
```

### 4. Install Requirements

```bash
pip install -r requirements.txt
```

### 5. Run Streamlit App

```bash
streamlit run app.py
```

---

## 📌 Future Improvements

- Higher accuracy CNN architecture
- Object Detection using YOLO
- Real-time video monitoring
- GPS-based damage mapping
- Severity heatmaps
- Mobile deployment
- Cloud-based monitoring system

---

## 👩‍💻 Developed By

Sharvani  
AI / Data Science Project

---

## 📜 License

This project is developed for educational and academic purposes.