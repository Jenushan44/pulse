import streamlit as st
import pandas as pd
import joblib


model = joblib.load("models/pulse_model.joblib")

df = pd.read_csv("data/processed/training_data_with_history.csv")
df["date"] = pd.to_datetime(df["date"])

st.title("Pulse")
st.caption("Predicts hard drive failure risk within 30 days using S.M.A.R.T. data.")


# Gets every different drive in the dataset
serial_numbers = df["serial_number"].unique()


st.sidebar.header("Drive Selection")
selected_serial = st.sidebar.selectbox("Select a drive", serial_numbers)

drive_data = df[df["serial_number"] == selected_serial]

selected_date = st.sidebar.selectbox("Select a date", drive_data["date"].sort_values(ascending=False))

selected_row = drive_data[drive_data["date"] == selected_date].iloc[[0]]

# Gets only the features that are used by the saved model
model_features = selected_row[model.feature_names_in_]
failure_probability = model.predict_proba(model_features)[0, 1]

if failure_probability >= 0.3:
    risk = "HIGH"
else:
    risk = "LOW"



st.subheader("Prediction")
col1, col2 = st.columns(2)

with col1: st.metric("Failure Probability", f"{failure_probability * 100:.1f}%")
with col2: st.metric("Risk Level", risk)


# Drive health information
st.subheader("Drive Health")
col1, col2, col3 = st.columns(3)

col1.metric("SMART 5", selected_row["smart_5_raw"].iloc[0])
col2.metric("7-Day Change", selected_row["smart_5_raw_change_7d"].iloc[0])
col3.metric("30-Day Change", selected_row["smart_5_raw_change_30d"].iloc[0])



st.subheader("SMART 5 Recent Trend")

recent_data = drive_data[drive_data["date"] <= selected_date].sort_values("date").tail(30)
st.line_chart(recent_data.set_index("date")["smart_5_raw"])



st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("Precision", "98.5%")
col2.metric("Recall", "76.6%")
col3.metric("F1 Score", "86.2%")

st.caption("Performance measured on the final time-based test data using a 0.30 prediction threshold.")
st.caption("The demo uses the Backblaze drive records included in this project.")