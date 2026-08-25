import streamlit as st
import pandas as pd
import joblib
import altair as alt

st.set_page_config(page_title="Pulse", page_icon="assets/hard-drive-image.png", layout="wide")

st.markdown("""
<style>
.block-container {
  max-width: 1300px;
  padding-top: 2rem;

  }

[data-testid="stMetric"] {
  background-color: #161b22;
  border: 1px solid #30363d;
  border-radius: 10px;
  padding: 18px;
}


section[data-testid="stSidebar"] {
    border-right: 1px solid #30363d;
}

</style>
""", 
unsafe_allow_html=True)

model = joblib.load("models/pulse_model.joblib")

df = pd.read_csv("data/processed/training_data_with_history.csv")
df["date"] = pd.to_datetime(df["date"])

def show_value(value):
  if pd.isna(value):
    return "N/A"
  else: 
    return f"{value:.1f}"

def show_change(value):

  if pd.isna(value):
    return "N/A"
  elif value > 0:
    return f"+{value:.1f}"
  else: 
    return f"{value:.1f}"

st.sidebar.subheader("Filters")

serial_numbers = sorted(df["serial_number"].dropna().unique())
drive_options = ["All"] + list(serial_numbers)

selected_serial = st.sidebar.selectbox("Serial Number", drive_options)

if selected_serial == "All":
    date_options = ["All"]
    selected_date = st.sidebar.selectbox("Observation Date", date_options)
else:
    
    drive_data = df[df["serial_number"] == selected_serial].sort_values("date")
    dates = list(drive_data["date"].sort_values(ascending=False))
    date_options = ["All"] + dates

    selected_date = st.sidebar.selectbox("Observation Date", date_options, format_func=lambda date: "All" if date == "All" else date.strftime("%b %d, %Y"))

st.sidebar.divider()
st.sidebar.subheader("Model")
st.sidebar.write("Random Forest")
st.sidebar.caption("Prediction window: 30 days")
st.sidebar.caption("Risk threshold: 30%")
st.sidebar.caption("Raw + historical S.M.A.R.T. features")

st.title("PULSE")
st.write("Hard Drive Failure Prediction Dashboard")
st.caption("Uses S.M.A.R.T. data and recent drive history to estimate the failure risk within the next 30 days.")

st.divider()
st.subheader("Model Performance")
st.caption("Performance on the final time-based test set.")

precision_col, recall_col, f1_col, roc_col, pr_col = st.columns(5)

precision_col.metric("Precision", "98.5%")
recall_col.metric("Recall", "76.6%")
f1_col.metric("F1 Score", "86.2%")
roc_col.metric("ROC-AUC", "97.9%")
pr_col.metric("PR-AUC", "97.3%")

st.write("")

importance_col, details_col = st.columns([2, 1], gap="large")

with importance_col:
  st.subheader("What Influences the Model?")
  st.caption("Permutation importance from the final model analysis.")

  importance_df = pd.DataFrame({
      "Feature": ["SMART 5 — 30D Average", "SMART 5 — 30D Change", "SMART 5 — 7D Average", "SMART 9 Raw", "SMART 193 Raw", "SMART 5 — 7D Variation", "SMART 197 Raw", "SMART 5 — 1D Change", "SMART 5 — 7D Change", "SMART 5 Raw"],
      "Importance": [0.109606, 0.085660, 0.043715, 0.038237, 0.032907, 0.026483, 0.019379, 0.018794, 0.017166, 0.009741]
  })

  importance_chart = alt.Chart(importance_df).mark_bar().encode(
    x=alt.X("Importance:Q", title="Permutation Importance"),
    y=alt.Y("Feature:N", sort="-x", title=None),

    tooltip=["Feature", alt.Tooltip("Importance", format=".3f")]

  ).properties(height=300)

  st.altair_chart(importance_chart, use_container_width=True)


with details_col:
    st.subheader("Model Details")
    st.write("**Algorithm**")
    st.write("Random Forest")
    st.write("**Prediction Target**")
    st.write("Failure within 30 days")
    st.write("**Decision Threshold**")
    st.write("30%")
    st.write("**Validation**")
    st.write("Time-based split")
    st.write("**Dataset**")
    st.write("Backblaze drive data")

if selected_serial != "All":

  drive_data = df[df["serial_number"] == selected_serial].sort_values("date")

  if selected_date == "All":
      selected_row = drive_data.iloc[[-1]]
      selected_date = selected_row["date"].iloc[0]
  else:
        
      selected_row = drive_data[drive_data["date"] == selected_date].iloc[[0]]

  model_features = selected_row[model.feature_names_in_]
  failure_probability = model.predict_proba(model_features)[0, 1]


  if failure_probability >= 0.3:
      risk = "HIGH"
  else:
      risk = "LOW"

  smart_5 = selected_row["smart_5_raw"].iloc[0]
  change_1d = selected_row["smart_5_raw_change_1d"].iloc[0]
  change_7d = selected_row["smart_5_raw_change_7d"].iloc[0]
  change_30d = selected_row["smart_5_raw_change_30d"].iloc[0]

  st.divider()

  st.subheader("Selected Drive")


  probability_col, risk_col, smart_col, change_col = st.columns(4)

  probability_col.metric("Failure Probability", f"{failure_probability * 100:.1f}%")

  risk_col.metric("Risk Level", risk)
  smart_col.metric("SMART 5", show_value(smart_5))
  change_col.metric("30-Day Change", show_change(change_30d))

  st.caption(f"Drive {selected_serial} • {selected_date.strftime('%b %d, %Y')}")
  st.write("")


  chart_col, snapshot_col = st.columns([2.2, 1], gap="large")


  with chart_col:
    st.subheader("Drive Health Trend")
    st.caption("SMART 5 values over the most recent observations.")


    recent_data = drive_data[drive_data["date"] <= selected_date].sort_values("date").tail(30)
    trend_data = recent_data[["date", "smart_5_raw"]].dropna()

    if len(trend_data) > 1:

      trend_chart = alt.Chart(trend_data).mark_area(opacity=0.25, line=True).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d")),
        y=alt.Y("smart_5_raw:Q", title="SMART 5"),
        tooltip=[alt.Tooltip("date:T", title="Date", format="%b %d, %Y"),alt.Tooltip("smart_5_raw:Q", title="SMART 5")
      ]).properties(height=300)

      st.altair_chart(trend_chart, use_container_width=True)
    else:
      st.info("Not enough SMART 5 history is available for this observation.")

  with snapshot_col:
    st.subheader("Drive Snapshot")
    st.caption("Recent SMART 5 changes")

    snapshot1, snapshot2 = st.columns(2)
    snapshot1.metric("1 Day", show_change(change_1d))
    snapshot2.metric("7 Days", show_change(change_7d))
    snapshot3, snapshot4 = st.columns(2)
    snapshot3.metric("30 Days", show_change(change_30d))
    snapshot4.metric("Current", show_value(smart_5))

    st.write("")

    st.caption("Selected Drive")
    st.code(str(selected_serial), language=None)


else:
    st.info("Select a drive from the sidebar to view its failure prediction and recent S.M.A.R.T. history.")

st.divider()
st.caption("Pulse - Machine Learning Hard Drive Failure Prediction")