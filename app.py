import json
import time
import numpy as np
import paho.mqtt.client as mqtt
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import streamlit as st

# ---------------------------------------------------------
# 1. ตั้งค่าหน้าเพจ Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="Prediction of Cavitation in Centrifugal Pump",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Prediction of Cavitation in Centrifugal Pump ⚙️")
st.caption(
    "ระบบตรวจจับและทำนายการเกิดโพรงอากาศในปั๊มน้ำด้วย IoT และ Machine"
    " Learning (Real-Time)"
)


# ---------------------------------------------------------
# 2. โหลดโมเดล Random Forest และ SVM พร้อมคำนวณ Precision
# ---------------------------------------------------------
@st.cache_resource
def load_and_evaluate_models():
  np.random.seed(42)
  X = np.random.uniform(-15, 15, (500, 3))
  mag = np.linalg.norm(X, axis=1)
  y = np.where(
      mag < 8,
      "Normal",
      np.where(mag < 11, "50Cavi", np.where(mag < 13, "75Cavi", "100Cavi")),
  )

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # 1. Random Forest Model
  rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)
  rf_pred = rf_model.predict(X_test)
  rf_precision = precision_score(y_test, rf_pred, average="weighted")

  # 2. SVM Model
  svm_model = SVC(kernel="rbf", probability=True, random_state=42)
  svm_model.fit(X_train, y_train)
  svm_pred = svm_model.predict(X_test)
  svm_precision = precision_score(y_test, svm_pred, average="weighted")

  return rf_model, rf_precision, svm_model, svm_precision


rf_model, rf_precision, svm_model, svm_precision = load_and_evaluate_models()

# ---------------------------------------------------------
# 3. จัดการ Session State สำหรับเก็บข้อมูล
# ---------------------------------------------------------
if "latest_sensor" not in st.session_state:
  st.session_state.latest_sensor = {"x": 0.16, "y": 0.19, "z": 0.05}

if "history_df" not in st.session_state:
  st.session_state.history_df = pd.DataFrame(
      columns=["Timestamp", "X", "Y", "Z"]
  )

if "data_history" not in st.session_state:
  st.session_state.data_history = []

# ---------------------------------------------------------
# 4. ตั้งค่า MQTT Client (รับค่า Real-Time จากสมาร์ทโฟน/phyphox)
# ---------------------------------------------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "pump/vibration/data"


def on_message(client, userdata, msg):
  try:
    data = json.loads(msg.payload.decode())
    st.session_state.latest_sensor["x"] = float(data.get("x", 0.0))
    st.session_state.latest_sensor["y"] = float(data.get("y", 0.0))
    st.session_state.latest_sensor["z"] = float(data.get("z", 0.0))
  except Exception:
    pass


@st.cache_resource
def start_mqtt():
  client = mqtt.Client()
  client.on_message = on_message
  try:
    client.connect(MQTT_BROKER, MQTT_PORT,
