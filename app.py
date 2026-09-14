import streamlit as st
import joblib

# Load trained model
model = joblib.load("demand_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Smart Retail Demand Prediction",
    page_icon="🛒",
    layout="centered"
)

# Title
st.title("🛒 Smart Retail Demand Prediction")

st.write(
    "This application uses a Machine Learning model "
    "to predict retail product demand."
)

st.divider()

# Input section
st.subheader("📋 Enter Product Details")

store = st.number_input(
    "Store Number",
    min_value=1,
    value=1
)

item = st.number_input(
    "Item Number",
    min_value=1,
    value=1
)

year = st.number_input(
    "Year",
    min_value=2017,
    max_value=2030,
    value=2017
)

month = st.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=1
)

day = st.number_input(
    "Day",
    min_value=1,
    max_value=31,
    value=1
)

day_of_week = st.number_input(
    "Day of Week (0 = Monday, 6 = Sunday)",
    min_value=0,
    max_value=6,
    value=0
)

previous_day_sales = st.number_input(
    "Previous Day Sales",
    min_value=0.0,
    value=20.0
)

st.divider()

# Prediction
if st.button("🔮 Predict Demand", use_container_width=True):

    input_data = [[
        store,
        item,
        year,
        month,
        day,
        day_of_week,
        previous_day_sales
    ]]

    prediction = model.predict(input_data)

    st.success(
        f"📦 Predicted Demand: {prediction[0]:.0f} units"
    )

    st.info(
        "The prediction is generated using the trained "
        "Random Forest Machine Learning model."
    )

st.divider()

st.caption("Smart Retail Demand Prediction | Machine Learning Project")
