"""
Streamlit Web App for Delivery Time Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2

# Page config
st.set_page_config(
    page_title="🍕 Delivery Time Predictor",
    page_icon="🍕",
    layout="centered"
)

# Title
st.title("🍕 Food Delivery Time Prediction")
st.markdown("Predict delivery time based on distance, traffic, and weather")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_model_model.pkl')
        return model
    except:
        st.error("Model not found! Train the model first using `python src/train.py`")
        return None

model = load_model()

# Haversine function
def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# Input form
st.sidebar.header("📍 Input Parameters")

# Distance input
st.sidebar.subheader("🗺️ Locations")
distance_option = st.sidebar.radio("Distance input:", ["Enter directly", "Enter coordinates"])

if distance_option == "Enter directly":
    distance_km = st.sidebar.slider("Distance (km)", 0.5, 30.0, 5.0, 0.5)
else:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        rest_lat = st.number_input("Restaurant Lat", value=28.5)
        rest_lon = st.number_input("Restaurant Lon", value=77.0)
    with col2:
        cust_lat = st.number_input("Customer Lat", value=28.6)
        cust_lon = st.number_input("Customer Lon", value=77.1)
    
    distance_km = calculate_distance(rest_lat, rest_lon, cust_lat, cust_lon)
    st.sidebar.success(f"Calculated distance: {distance_km:.2f} km")

# Time
st.sidebar.subheader("⏰ Order Time")
order_hour = st.sidebar.slider("Hour of day", 0, 23, 12)
is_weekend = st.sidebar.checkbox("Weekend order")

# Traffic
st.sidebar.subheader("🚦 Traffic")
traffic = st.sidebar.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"])
traffic_encoded = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}[traffic]

# Weather
st.sidebar.subheader("🌦️ Weather")
weather = st.sidebar.selectbox("Weather Conditions", ["Clear", "Cloudy", "Fog", "Rain", "Storm"])
weather_encoded = {"Clear": 0, "Cloudy": 1, "Fog": 2, "Rain": 3, "Storm": 4}[weather]

# Calculate features
is_peak_hour = 1 if (11 <= order_hour <= 14) or (18 <= order_hour <= 21) else 0
distance_traffic = distance_km * traffic_encoded
peak_traffic = is_peak_hour * traffic_encoded

# Prediction button
if st.sidebar.button("🎯 Predict Delivery Time", type="primary"):
    if model is not None:
        # Prepare features
        features = pd.DataFrame({
            'distance_km': [distance_km],
            'order_hour': [order_hour],
            'is_peak_hour': [is_peak_hour],
            'is_weekend': [1 if is_weekend else 0],
            'traffic_encoded': [traffic_encoded],
            'weather_encoded': [weather_encoded],
            'distance_traffic': [distance_traffic],
            'peak_traffic': [peak_traffic]
        })
        
        # Predict
        prediction = model.predict(features)[0]
        
        # Display result
        st.success("✅ Prediction Complete!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Distance", f"{distance_km:.1f} km")
        with col2:
            st.metric("Traffic", traffic)
        with col3:
            st.metric("Weather", weather)
        
        # Main prediction
        st.markdown("---")
        st.markdown("### 🕐 Estimated Delivery Time")
        st.markdown(f"# {prediction:.0f} minutes")
        
        # Additional info
        if is_peak_hour:
            st.info("⚠️ Peak hour - May take longer")
        if traffic_encoded >= 3:
            st.warning("🚦 High traffic - Delivery may be delayed")
        if weather_encoded >= 2:
            st.warning("🌧️ Bad weather - Extra time required")

# Info section
st.markdown("---")
st.markdown("### 📊 Model Information")
st.markdown("""
**Model**: Random Forest Regressor  
**Performance**: MAE < 7 minutes  
**Features Used**:
- Distance (Haversine formula)
- Traffic density
- Weather conditions
- Time of day (peak hours)
- Day type (weekend/weekday)
""")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | [GitHub](https://github.com/yourusername/food-delivery-prediction)")
