
---

# 🚦 Traffic Prediction Using Machine Learning

**(Time Series + Weather Data | Flask Deployment)**

---

## 📌 Project Overview

Traffic congestion is a major urban challenge that leads to increased travel time, fuel consumption, and pollution.
This project predicts **future traffic volume** using **historical time-series traffic data combined with weather conditions** and deploys the trained machine learning model as a **Flask web application**.

---

## 🎯 Objectives

* Predict traffic volume using machine learning
* Apply **time-series feature engineering**
* Analyze impact of **weather conditions** on traffic
* Deploy ML model using **Flask**
* Visualize predictions using **Matplotlib**
* Build a responsive UI using **Bootstrap**

---

## 🧠 Machine Learning Approach

* **Problem Type:** Time-Series Regression
* **Model Used:** Random Forest Regressor
* **Key Techniques:**

  * Lag features (previous hour, previous day traffic)
  * Temporal features (hour, day, month)
  * Weather features (temperature, rain, snow, clouds)

---

## 🏗️ System Architecture

```
Traffic + Weather Data
        ↓
Data Cleaning & Feature Engineering
        ↓
ML Model Training (Random Forest)
        ↓
Model Serialization (.pkl)
        ↓
Flask Web Application
        ↓
Prediction Output + Visualization
```

---

## 📁 Project Folder Structure

```
Traffic_Prediction_Project/
│
├── data/
│   └── traffic_weather.csv
│
├── model/
│   ├── train_model.py
│   └── traffic_model.pkl
│
├── app/
│   ├── app.py
│   ├── static/
│   └── templates/
│       └── index.html
│
├── requirements.txt
└── README.md
```

---

## 🌐 Web Application Features

* User-friendly web form for traffic and weather inputs
* Real-time traffic volume prediction
* **Dynamic Matplotlib graph generation per prediction**
* Responsive UI with **Bootstrap Navbar & Footer**
* Clean separation of backend, model, and frontend

---

## 🛠️ Tech Stack

* **Language:** Python
* **Backend:** Flask
* **Machine Learning:** Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib
* **Frontend:** HTML, Bootstrap

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Traffic_Prediction_Project.git
cd Traffic_Prediction_Project
```

### 2️⃣ Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 3️⃣ Train the Model

```bash
cd model
python train_model.py
cd ..
```

### 4️⃣ Run Flask Application

```bash
cd app
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

## 📊 Sample Input Values

```
Temperature        : 290
Rain               : 0
Snow               : 0
Clouds (%)         : 40
Hour               : 8
Day (0=Mon)        : 2
Month              : 6
Traffic 1 hr ago   : 8500
Traffic 24 hrs ago : 8200
```

---

## 📈 Output

* Predicted traffic volume
* Bar graph comparing:

  * Previous hour traffic
  * Previous day traffic
  * Predicted traffic

---

## 🧠 Key Learnings

* Practical handling of **time-series ML problems**
* Feature engineering for temporal data
* Deploying ML models into **real-world Flask applications**
* Dynamic visualization using Matplotlib
* Writing clean, scalable project structure

---

