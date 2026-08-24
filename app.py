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