import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/pulse_model.joblib")

df = pd.read_csv("data/processed/training_data_with_history.csv")
df["date"] = pd.to_datetime(df["date"])

st.title("Pulse")
st.write("Hard Drive Failure Prediction")

# Gets every different drive in the dataset
serial_numbers = df["serial_number"].unique()

# Creates a selectbox for each of the drives
selected_serial = st.selectbox("Select a drive", serial_numbers)


drive_data = df[df["serial_number"] == selected_serial]
selected_date = st.selectbox("Select a date", drive_data["date"].sort_values(ascending=False))

selected_row = drive_data[drive_data["date"] == selected_date].iloc[[0]]
model_features = selected_row[model.feature_names_in_]

failure_probability = model.predict_proba(model_features)[0, 1]

if failure_probability >= 0.3:
    risk = "HIGH"
else:
    risk = "LOW"

st.subheader("Prediction")
st.metric("Failure Probability Within 30 Days", f"{failure_probability * 100:.1f}%")
st.write("Risk:", risk)