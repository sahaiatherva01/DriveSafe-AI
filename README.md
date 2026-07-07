# 🚗 DriveSafe AI


### **AI-Powered Driver Monitoring System for Real-Time Road Safety**

*Enhancing road safety through Computer Vision, MediaPipe, and Machine Learning.*


## Overview

**DriveSafe AI** is an intelligent **Driver Monitoring System (DMS)** designed to improve road safety by continuously monitoring a driver's facial behavior and estimating the risk of unsafe driving conditions.

Using **Computer Vision**, **MediaPipe Face Landmarks**, and **Machine Learning**, the system detects signs of:

* Driver Drowsiness
* Distraction
* Fatigue
* Looking Away
* Cognitive Risk Indicators
* Unsafe Driving Behaviour

The application processes video frames in real time and provides a dynamic dashboard that visualizes driver status, behavioural analytics, and an overall safety risk score.


# Features

### Real-Time Face Tracking

* Face Detection
* Facial Landmark Tracking
* Head Pose Estimation
* Eye Aspect Ratio (EAR)
* Mouth Aspect Ratio (MAR)


### Fatigue Detection

* Eye Closure Detection
* Blink Rate Analysis
* Yawning Detection
* Drowsiness Monitoring
* Driver Alert Generation

---

### AI Risk Analysis

The hybrid AI engine evaluates multiple behavioural signals to estimate the driver's safety level.

It considers:

* Eye Closure
* Head Orientation
* Blink Frequency
* Mouth Opening
* Facial Movement
* Driver Attention

The model predicts an overall **Risk Score** that updates continuously during monitoring.


### Interactive Dashboard

The dashboard provides:

* Live Camera Feed
* Driver Status
* Risk Score
* Fatigue Indicator
* Attention Level
* Behaviour Analytics
* Real-Time Alerts


### Analytics

The application records useful behavioural insights such as:

* Average Risk
* Driver Attention
* Blink Statistics
* Fatigue Trend
* Session Summary
* Driving Behaviour Overview


# System Architecture

```
Camera Input
      │
      ▼
OpenCV Video Capture
      │
      ▼
MediaPipe Face Detection
      │
      ▼
468 Facial Landmarks
      │
      ▼
Feature Extraction
      │
      ▼
Machine Learning Model
      │
      ▼
Risk Prediction
      │
      ▼
Dashboard & Alerts
```


#  Tech Stack

## Backend

* Python
* Flask

## Computer Vision

* OpenCV
* MediaPipe

## Machine Learning

* PyTorch
* NumPy
* Scikit-Learn
* SciPy

## Frontend

* HTML5
* CSS3
* JavaScript


# 📂 Project Structure

```
DriveSafe-AI
│
├── backend/
│   ├── app.py
│   ├── detector.py
│   ├── ai_model.py
│   ├── features.py
│   ├── utils.py
│   ├── requirements.txt
│   ├── risk_mlp.pth
│   └── face_landmarker.task
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── README.md
```


#  Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/sahaiatherva01/DriveSafe-AI.git
cd DriveSafe-AI
```


## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```


## 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```


## 4. Run the Backend

```bash
cd backend
python app.py
```


## 5. Open the Frontend

Simply open:

```
frontend/index.html
```

or serve it using any local HTTP server.


#  Workflow

```
Start Application
        │
        ▼
Initialize Camera
        │
        ▼
Detect Driver Face
        │
        ▼
Extract Face Landmarks
        │
        ▼
Compute Behaviour Features
        │
        ▼
Predict Driver Risk
        │
        ▼
Display Dashboard
        │
        ▼
Generate Alerts
```


# Future Enhancements

* Driver Identity Recognition
* Emotion Detection
* Mobile Application
* Cloud Analytics Dashboard
* Trip History
* Driver Performance Reports
* Multi-Camera Support
* Voice Alerts
* GPS Integration
* Night Driving Optimization
* Weather-aware Risk Prediction
* Driver Behaviour Scoring

# Applications

* Smart Vehicles
* Fleet Management
* Public Transportation
* Logistics
* Commercial Trucks
* Driver Safety Research
* Insurance Analytics
* Driver Training Systems


# Learning Outcomes

This project demonstrates practical implementation of:

* Computer Vision
* Human Behaviour Analysis
* Face Landmark Detection
* Machine Learning
* Real-Time Analytics
* Flask Backend Development
* Frontend Integration
* AI-Based Risk Assessment


> **DriveSafe AI** demonstrates how Artificial Intelligence and Computer Vision can be combined to build practical, real-time driver safety systems that contribute to safer roads and smarter transportation.
