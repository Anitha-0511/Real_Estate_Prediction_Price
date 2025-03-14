# 🏡 House Price Prediction App

This is a Streamlit-based web application that predicts house prices based on the number of bedrooms, bathrooms, and house size. The app trains a Linear Regression model using real estate data and provides an estimated price.

## 📌 Features
- Train a **Linear Regression Model** on real estate data.
- **Predict house prices** based on user inputs.
- **Load and preprocess** data automatically.
- **Cache model & scaler** to improve performance.
- **Show model evaluation metrics** (MAE, MSE, R² Score).

---

## 🚀 Getting Started
### 1️⃣ **Install Dependencies**
Ensure you have Python installed, then install the required dependencies:
```bash
pip install numpy pandas streamlit joblib scikit-learn
```

### 2️⃣ **Run the App**
Execute the following command to start the Streamlit app:
```bash
streamlit run app.py
```

### 3️⃣ **Enter House Details**
- Input the **number of bedrooms, bathrooms, and house size (sqft)** in the sidebar.
- Click **"Predict Price"** to see the estimated house price.

---

## 📂 File Structure
```
📁 Project Directory
│-- app.py                 # Streamlit application script
│-- Realtor_data.csv       # Dataset for training
│-- model.pkl              # Trained Linear Regression model (generated after training)
│-- scaler.pkl             # StandardScaler for feature scaling (generated after training)
│-- requirements.txt       # Required dependencies
│-- README.md              # Project documentation (this file)
```

---

## 🛠 How It Works
1. **Loads and Cleans Data:**
   - Reads `Realtor_data.csv`.
   - Drops irrelevant columns and handles missing values.
2. **Trains the Model (if needed):**
   - Splits data into training & test sets.
   - Scales the data using `StandardScaler`.
   - Fits a `LinearRegression` model.
   - Saves the trained model & scaler.
3. **Predicts House Prices:**
   - Loads the saved model & scaler.
   - Transforms input data and makes predictions.

---

## 📊 Model Performance Metrics
After training, the model displays:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**

---

## 🔗 Additional Information
- **Developed using**: Python, Streamlit, Scikit-learn, Pandas, NumPy.
- **Dataset:** Ensure `Realtor_data.csv` is present in the project directory.
- **Future Improvements:**
  - Add more features (location, year built, etc.).
  - Use advanced models (Random Forest, XGBoost).
  - Deploy the model on a cloud platform.

---

<img width="472" alt="image" src="https://github.com/user-attachments/assets/c977c97b-dce1-43a6-a53d-e89886c5d485" />


<img width="346" alt="image" src="https://github.com/user-attachments/assets/c6e027ca-03e6-4fa3-9aff-245fdb4b1275" />



🎯 **Enjoy predicting house prices with this app!** 🚀




