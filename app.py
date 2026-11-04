import json
import time
import numpy as np
import paho.mqtt.client as mqtt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# --- 1. ตั้งค่า PAGE CONFIG ---
st.set_page_config(
    page_title="Real-Time Cavitation Prediction", page_icon="⚙️", layout="wide"
)

st.title("⚙️ Real-Time Cavitation Prediction in Centrifugal Pump")
st.caption(
    "รับค่า Accelerometer (X, Y, Z) จากสมาร์ทโฟนผ่าน MQTT เพื่อทำนายสภาวะ"
    " Cavitation"
)

# --- 2. ตั้งค่า MQTT BROKER ---
MQTT_BROKER = "broker.hivemq.com"  # Public Broker ฟรี (หรือใช้ Mosquitto / EMQX)
MQTT_PORT = 1883
MQTT_TOPIC = "buu/eng/pump/accelerometer"

# Session State สำหรับเก็บข้อมูลแสดงกราฟ (50 จุดล่าสุด)
if "sensor_data" not in st.session_state:
  st.session_state.sensor_data = pd.DataFrame(
      columns=["timestamp", "x", "y", "z"]
  )

if "latest_payload" not in st.session_state:
  st.session_state.latest_payload = {"x": 0.0, "y": 0.0, "z": 0.0}

# [เพิ่มที่ 1] Session State สำหรับเก็บบันทึกประวัติข้อมูลทั้งหมดเพื่อ Export CSV
if "vibration_history" not in st.session_state:
  st.session_state.vibration_history = []


# --- 3. MQTT CALLBACK FUNCTIONS ---
def on_connect(client, userdata, flags, rc, properties=None):
  if rc == 0:
    client.subscribe(MQTT_TOPIC)


def on_message(client, userdata, msg):
  try:
    payload = json.loads(msg.payload.decode())
    # รองรับโครงสร้าง JSON: {"x": 0.12, "y": -9.81, "z": 0.45}
    x = float(payload.get("x", 0.0))
    y = float(payload.get("y", 0.0))
    z = float(payload.get("z", 0.0))
    st.session_state.latest_payload = {"x": x, "y": y, "z": z}
  except Exception as e:
    pass


# --- 4. START MQTT CLIENT (BACKGROUND THREAD) ---
@st.cache_resource
def start_mqtt_client():
  client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
  client.on_connect = on_connect
  client.on_message = on_message
  client.connect(MQTT_BROKER, MQTT_PORT, 60)
  client.loop_start()
  return client


mqtt_client = start_mqtt_client()


# --- 5. DUMMY ML MODEL (สำหรับการทดสอบ) ---
@st.cache_resource
def load_trained_model():
  # ในใช้งานจริง ให้เปลี่ยนเป็น joblib.load('cavitation_model.pkl')
  X_train = np.random.uniform(-15, 15, (200, 3))
  # กำหนด Logic สมมติ: ถ้า Vector Magnitude สูงเกินเกณฑ์ จะเกิด Cavitation
  mag = np.linalg.norm(X_train, axis=1)
  y_train = np.where(
      mag > 12, "Cavitation Detected (50Cavi)", "Normal Operation"
  )

  model = RandomForestClassifier(n_estimators=50, random_state=42)
  model.fit(X_train, y_train)
  return model


model = load_trained_model()

# --- 6. SIDEBAR CONTROLS ---
st.sidebar.header("📡 Connection Info")
st.sidebar.text(f"Broker: {MQTT_BROKER}")
st.sidebar.text(f"Topic: {MQTT_TOPIC}")

st.sidebar.markdown("---")
st.sidebar.subheader("📱 การส่งข้อมูลจากสมาร์ทโฟน")
st.sidebar.write(
    "ใช้แอป เช่น **Sensor Streamer** หรือ **Phyphox** ส่ง JSON payload"
    " มาที่ Topic ด้านบน:"
)
st.sidebar.code('{"x": 1.25, "y": -9.81, "z": 0.45}', language="json")

auto_refresh = st.sidebar.checkbox(
    "เปิดใช้งาน Real-Time Auto Refresh", value=True
)
refresh_interval = st.sidebar.slider(
    "Refresh Interval (เสี้ยววินาที)", 0.2, 2.0, 0.5
)

# --- 7. MAIN DASHBOARD ---
# ดึงค่าล่าสุด
curr_x = st.session_state.latest_payload["x"]
curr_y = st.session_state.latest_payload["y"]
curr_z = st.session_state.latest_payload["z"]

# ทำนายสภาวะ Cavitation
input_features = np.array([[curr_x, curr_y, curr_z]])
prediction = model.predict(input_features)[0]
proba = model.predict_proba(input_features).max() * 100

# อัปเดตข้อมูลลง DataFrame (สำหรับพล็อตกราฟ 50 จุดล่าสุด)
now_str = pd.Timestamp.now(tz="Asia/Bangkok").strftime("%H:%M:%S")
new_row = pd.DataFrame(
    [{"timestamp": now_str, "x": curr_x, "y": curr_y, "z": curr_z}]
)
st.session_state.sensor_data = pd.concat(
    [st.session_state.sensor_data, new_row], ignore_index=True
).tail(50)

# [เพิ่มที่ 2] บันทึกข้อมูลเข้า Session History เพื่อรอ Export CSV
st.session_state.vibration_history.append({
    "Timestamp": pd.Timestamp.now(tz="Asia/Bangkok").strftime("%Y-%m-%d %H:%M:%S"),
    "X_axis": curr_x,
    "Y_axis": curr_y,
    "Z_axis": curr_z,
    "Status": prediction,
})

# แสดง Metric Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Axis X (m/s²)", f"{curr_x:.2f}")
col2.metric("Axis Y (m/s²)", f"{curr_y:.2f}")
col3.metric("Axis Z (m/s²)", f"{curr_z:.2f}")

if "Normal" in prediction:
  col4.success(f"✅ {prediction}")
else:
  col4.error(f"⚠️ {prediction} ({proba:.1f}%)")

st.markdown("---")

# Real-Time Line Chart (Plotly)
fig = go.Figure()
df_plot = st.session_state.sensor_data

fig.add_trace(
    go.Scatter(
        x=df_plot["timestamp"],
        y=df_plot["x"],
        mode="lines+markers",
        name="X Axis",
        line=dict(color="red"),
    )
)
fig.add_trace(
    go.Scatter(
        x=df_plot["timestamp"],
        y=df_plot["y"],
        mode="lines+markers",
        name="Y Axis",
        line=dict(color="green"),
    )
)
fig.add_trace(
    go.Scatter(
        x=df_plot["timestamp"],
        y=df_plot["z"],
        mode="lines+markers",
        name="Z Axis",
        line=dict(color="blue"),
    )
)

fig.update_layout(
    title="Real-Time Accelerometer Data (Last 50 points)",
    xaxis_title="Timestamp",
    yaxis_title="Acceleration (m/s²)",
    margin=dict(l=20, r=20, t=40, b=20),
    height=400,
)

st.plotly_chart(fig, use_container_width=True)

# [เพิ่มที่ 3] ปุ่ม Export CSV และ Clear Log บน Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Export Data")

if len(st.session_state.vibration_history) > 0:
  df_history = pd.DataFrame(st.session_state.vibration_history)
  csv_data = df_history.to_csv(index=False).encode("utf-8")

  st.sidebar.download_button(
      label="⬇️ Download CSV Log",
      data=csv_data,
      file_name=f'pump_vibration_{pd.Timestamp.now(tz="Asia/Bangkok").strftime("%Y%m%d_%H%M%S")}.csv',
      mime="text/csv",
  )
  st.sidebar.caption(f"บันทึกแล้วทั้งหมด: {len(df_history)} แถว")

  if st.sidebar.button("🗑️ Clear Data Log"):
    st.session_state.vibration_history = []
    st.rerun()
else:
  st.sidebar.info("กำลังรอรับข้อมูล...")

# Loop ให้ระบบ รีเฟรช หน้าอัตโนมัติ
if auto_refresh:
  time.sleep(refresh_interval)
  st.rerun()
