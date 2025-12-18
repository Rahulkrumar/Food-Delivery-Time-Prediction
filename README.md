# 🍕 Food Delivery Time Prediction

Machine learning system to predict food delivery time (ETA) in minutes using distance, traffic, weather, and order-time features.

> **Note:** This project is designed as a **fresher-level, industry-aligned** ML project (Zomato/Swiggy-style).

---

## 🎯 Problem Statement

Predict **delivery time (in minutes)** for a food order based on:

* Distance between restaurant and customer
* Traffic conditions
* Weather conditions
* Time of order (peak hours)

**Business Use:** Show accurate ETAs to customers and improve delivery planning.

---

## 📊 Dataset

**Source:** Public, real-world inspired food delivery dataset (CSV)

**Features Used:**

* Restaurant & Customer Location (latitude/longitude)
* Traffic Density (Low / Medium / High)
* Weather Conditions (Clear / Rain / Fog / Storm)
* Order Hour (0–23)

**Target Variable:**

* `Time_taken(min)` — delivery time in minutes

> Dataset source is kept generic to focus on modeling approach rather than geography.

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/Rahulkrumar/food-delivery-prediction.git
cd food-delivery-prediction

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Run ML Pipeline

```bash
# Data cleaning & validation
python src/data_processing.py

# Feature engineering
python src/feature_engineering.py

# Model training
python src/train.py
```

### 3️⃣ Run Web App (Optional)

```bash
streamlit run app/app.py
```

bash

# Clone repository

git clone [https://github.com/Rahulkrumar/food-delivery-prediction.git](https:/Rahulkrumar/github.com/food-delivery-prediction.git)
cd food-delivery-prediction

# Install dependencies

pip install -r requirements.txt

````

### 2️⃣ Run ML Pipeline
```bash
# Data cleaning & validation
python src/data_processing.py

# Feature engineering
python src/feature_engineering.py

# Model training
python src/train.py
````

### 3️⃣ Run Web App (Optional)

```bash
streamlit run app/app.py
```

---

## 🔬 Feature Engineering

### 1. Distance Calculation

* **Haversine Formula** to compute distance between restaurant and customer
* Feature: `distance_km`

### 2. Time Features

* `order_hour`: Hour of the day (0–23)
* `is_peak_hour`: Lunch (12–14) or Dinner (19–21)

### 3. Traffic Encoding

* Low → 1
* Medium → 2
* High → 3

### 4. Weather Encoding

* One-hot encoded weather conditions (Clear, Rain, Fog, Storm)

---

## 📈 Model Performance

| Model              | MAE (minutes) | RMSE   | R²        |
| ------------------ | ------------- | ------ | --------- |
| Linear Regression  | 11–12         | ~15    | ~0.70     |
| Random Forest      | **6–7**       | **~9** | **~0.85** |
| XGBoost (optional) | ~7            | ~9.5   | ~0.84     |

**Selected Model:** Random Forest Regressor (best MAE)

### 🔑 Top Features

1. `distance_km`
2. `traffic_density`
3. `is_peak_hour`
4. `weather_features`

---

## 💡 Business Impact

* 📍 **Accurate ETAs** for customers
* 🍽️ **Better preparation planning** for restaurants
* 🚴 **Efficient delivery routing**
* 😊 **Improved customer satisfaction**

---

## 📁 Project Structure

```
food-delivery-prediction/
│
├── data/
│   └── food_delivery_time_dataset.csv
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   └── train.py
├── models/
│   └── model.pkl
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```

---

## 🔧 Technology Stack

* **Language:** Python 3.8+
* **ML:** scikit-learn (Random Forest, Linear Regression)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib
* **Deployment:** Streamlit

---

## 📄 License

MIT License

---

⭐ If you found this project useful, please consider starring the repository!
