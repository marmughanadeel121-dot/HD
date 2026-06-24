import streamlit as str
import pickle
import numpy as np

# 1. Set up the page configuration
st.set_page_config(
    page_title="Model Predictor App",
    page_icon="🤖",
    layout="centered"
)

# 2. Load the trained model (.pkl file)
# @st.cache_resource ensures the model loads once and stays in memory
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ 'model.pkl' not found! Please ensure it's in the same directory as this script.")
    st.stop()

# 3. App Title and Description
st.title("🤖 Machine Learning Prediction App")
st.write("Enter the required features below to get a prediction from the trained model.")

st.divider()

# 4. User Input Fields 
# (Change these labels and defaults to match your model's actual features!)
st.subheader("Input Features")

feature_1 = st.number_input("Feature 1 (e.g., Age)", min_value=0, max_value=100, value=25)
feature_2 = st.number_input("Feature 2 (e.g., Income)", min_value=0, value=50000)
feature_3 = st.number_input("Feature 3 (e.g., Score)", min_value=0.0, max_value=10.0, value=7.5)

# 5. Prediction Logic
if st.button("Predict", type="primary"):
    # Arrange inputs into the format your model expects (usually a 2D array)
    input_data = np.array([[feature_1, feature_2, feature_3]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    st.divider()
    st.subheader("Prediction Result")
    st.success(f"🎉 The model predicted: **{prediction[0]}**")
