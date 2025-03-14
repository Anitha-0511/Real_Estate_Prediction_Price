import numpy as np
import pandas as pd
import streamlit as st
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# File Paths
DATA_FILE = "Realtor_data.csv"
MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"

# Load Data
@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        st.error(f"Data file not found: {DATA_FILE}")
        return None
    try:
        data = pd.read_csv(DATA_FILE)
        data.drop(columns=["brokered_by", "zip_code", "prev_sold_date"], inplace=True, errors="ignore")
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Train Model
@st.cache_resource
def train_model():
    data = load_data()
    if data is None:
        return None, None, None  # Exit if data is missing

    X = data[["bed", "bath", "house_size"]]
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model and scaler
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(model, MODEL_FILE)

    # Model evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return mae, mse, r2

# Load Model & Scaler Safely
def load_model():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        st.error("Model files not found. Please train the model first.")
        return None, None
    
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

# Predict Price
def predict_price(bed, bath, house_size):
    model, scaler = load_model()
    if model is None or scaler is None:
        return "Model not available. Please train the model first."

    input_data = np.array([[bed, bath, house_size]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return f"${prediction[0]:,.2f}"

# Streamlit UI
st.title("🏡 House Price Prediction App")

st.sidebar.header("Enter House Details")
bed = st.sidebar.number_input("Number of Bedrooms", min_value=1, step=1, value=3)
bath = st.sidebar.number_input("Number of Bathrooms", min_value=1, step=1, value=2)
house_size = st.sidebar.number_input("House Size (sqft)", min_value=500, step=100, value=1500)

if st.sidebar.button("Predict Price"):
    price = predict_price(bed, bath, house_size)
    st.sidebar.success(f"Estimated Price: {price}")

# Train model only once
if "trained" not in st.session_state:
    st.session_state["trained"] = True
    mae, mse, r2 = train_model()
else:
    mae, mse, r2 = None, None, None

# Display Metrics
st.header("📊 Model Training Metrics")
if mae is not None:
    st.write(f"**Mean Absolute Error:** {mae:,.2f}")
    st.write(f"**Mean Squared Error:** {mse:,.2f}")
    st.write(f"**R² Score:** {r2:.2f}")
else:
    st.write("✅ Model already trained and cached.")
