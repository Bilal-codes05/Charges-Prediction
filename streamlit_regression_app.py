import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset (ensure it's in the same directory or provide path)
data = pd.read_csv("insurance.csv")

# Convert categorical columns to category type
data['sex'] = data['sex'].astype('category')
data['smoker'] = data['smoker'].astype('category')
data['region'] = data['region'].astype('category')

# Encoding categorical columns
data['sex'] = data['sex'].cat.codes
data['smoker'] = data['smoker'].cat.codes
data['region'] = data['region'].cat.codes

# Define features and target
X = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = data['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit Interface
st.title("Insurance Charges Prediction App")

st.write("### Enter the details to predict the insurance charges:")

# Input fields for the user to enter data
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=2)
smoker = st.selectbox("Smoker", ["Yes", "No"])
region = st.selectbox("Region", ["Northeast", "Southeast", "Southwest", "Northwest"])

# Convert input data to match the model format
sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0
region_dict = {"Northeast": 0, "Southeast": 1, "Southwest": 2, "Northwest": 3}
region = region_dict.get(region, 0)

# Prepare the input data for prediction
input_data = np.array([[age, sex, bmi, children, smoker, region]])
input_data_scaled = scaler.transform(input_data)

# Predict using the trained model
prediction = model.predict(input_data_scaled)

# Display the predicted charge
st.write(f"### Predicted Insurance Charges: ${prediction[0]:.2f}")

# Display the performance metrics
st.write(f"### Model Performance Metrics:")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Make the app more interactive and visually appealing with some styling
st.markdown("""
<style>
    .stApp {
        background-color: #f7f7f7;
    }
    h1 {
        color: #4CAF50;
    }
    .stTextInput, .stNumberInput, .stSelectbox {
        background-color: #f0f0f0;
    }
    .stButton {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)
