# app.py
import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "iris_model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "Iris.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found! Make sure iris_model.pkl is in the repo.")
        st.stop()
    return joblib.load(MODEL_PATH)

df    = load_data()
model = load_model()

st.title("🌸 Iris Species Predictor")
st.write("### Dataset Preview")
st.write(df)

st.sidebar.header("Input Features")
sepal_length = st.sidebar.number_input("Sepal Length (cm)", 4.0, 8.0, step=0.1)
sepal_width  = st.sidebar.number_input("Sepal Width (cm)",  2.0, 5.0, step=0.1)
petal_length = st.sidebar.number_input("Petal Length (cm)", 1.0, 7.0, step=0.1)
petal_width  = st.sidebar.number_input("Petal Width (cm)",  0.1, 2.5, step=0.1)

feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
input_data   = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=feature_cols
)

if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)
    st.write("### Prediction:")
    st.write(prediction[0])
    st.success("✅ Successfully Predicted!")