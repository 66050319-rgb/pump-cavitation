import json
import time
from io import BytesIO
import numpy as np
import paho.mqtt.client as mqtt
import pandas as pd
import plotly.graph_objects as go
import qrcode
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
# 4. ตั้งค่า MQTT Client (รับค่าจากสมาร์ทโฟน/phyphox)
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
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(MQTT_TOPIC)
    client.loop_start()
  except Exception as e:
    st.error(f"ไม่สามารถเชื่อมต่อ MQTT Broker ได้: {e}")
  return client


mqtt_client = start_mqtt()


# ---------------------------------------------------------
# 5. ฟังก์ชันสร้าง QR Code สำหรับแอป phyphox
# ---------------------------------------------------------
def generate_phyphox_qr():
  phyphox_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<phyphox version="1.7">
    <title>Pump Vibration (MQTT)</title>
    <category>IoT Sensors</category>
    <description>ส่งค่าการสั่นสะเทือน 3 แกนเข้าสู่วงปั๊มน้ำ Real-Time Dashboard</description>
    <data-containers>
        <container size="1">acc_x</container>
        <container size="1">acc_y</container>
        <container size="1">acc_z</container>
    </data-containers>
    <input>
        <sensor type="accelerometer" average="false">
            <output component="x">acc_x</output>
            <output component="y">acc_y</output>
            <output component="z">acc_z</output>
        </sensor>
    </input>
    <network>
        <mqtt id="mqtt_stream" broker="{MQTT_BROKER}" port="1883" topic="{MQTT_TOPIC}" interval="0.2">
            <send container="acc_x" bind="x" />
            <send container="acc_y" bind="y" />
            <send container="acc_z" bind="z" />
        </mqtt>
    </network>
</phyphox>"""
  qr = qrcode.QRCode(version=1, box_size=6, border=2)
  qr.add_data(phyphox_xml)
  qr.make(fit=True)
  img = qr.make_image(fill_color="black", back_color="white")

  buf = BytesIO()
  img.save(buf, format="PNG")
  return buf.getvalue()


# ---------------------------------------------------------
# 6. ส่วนควบคุมหน้าจอ (Sidebar Controls)
# ---------------------------------------------------------
st.sidebar.header("🎛️ ระบบควบคุม")
auto_refresh = st.sidebar.checkbox(
    "เปิดใช้งาน Real-Time Auto Refresh", value=True
)
refresh_interval = st.sidebar.slider("ความถี่ในการรีเฟรช (วินาที)", 0.5, 3.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("📱 เชื่อมต่อผ่าน Phyphox App")
with st.sidebar.expander("📲 สแกน QR Code เพื่อตั้งค่ามือถือ"):
  st.write("1. เปิดแอป **phyphox** บนสมาร์ทโฟน")
  st.write("2. กดปุ่ม **+** แล้วเลือก **Add experiment from QR code**")
  st.write("3. สแกน QR Code ด้านล่างนี้เพื่อเชื่อมต่ออัตโนมัติ:")
  qr_bytes = generate_phyphox_qr()
  st.image(
      qr_bytes,
      caption="Phyphox MQTT Auto-Config QR Code",
      use_container_width=True,
  )

st.sidebar.markdown("---")
st.sidebar.subheader("📊 ข้อมูลสถานะการเชื่อมต่อ")
st.sidebar.info(f"**Broker:** {MQTT_BROKER}\n\n**Topic:** {MQTT_TOPIC}")

# ---------------------------------------------------------
# 7. ประมวลผลข้อมูลปัจจุบัน และเก็บบันทึก
# ---------------------------------------------------------
curr_x = st.session_state.latest_sensor["x"]
curr_y = st.session_state.latest_sensor["y"]
curr_z = st.session_state.latest_sensor["z"]
abs_accel = np.sqrt(curr_x**2 + curr_y**2 + curr_z**2)

now_time_str = pd.Timestamp.now(tz="Asia/Bangkok").strftime("%H:%M:%S")
now_full_time = pd.Timestamp.now(tz="Asia/Bangkok").strftime(
    "%Y-%m-%d %H:%M:%S"
)

# บันทึกข้อมูลลง History สำหรับทำกราฟ
new_row = pd.DataFrame(
    [{"Timestamp": now_time_str, "X": curr_x, "Y": curr_y, "Z": curr_z}]
)
st.session_state.history_df = pd.concat(
    [st.session_state.history_df, new_row], ignore_index=True
).tail(50)

# ทำนายสภาวะด้วยโมเดล ML ทั้งสองแบบ
input_features = np.array([[curr_x, curr_y, curr_z]])

rf_pred = rf_model.predict(input_features)[0]
rf_proba = rf_model.predict_proba(input_features).max() * 100

svm_pred = svm_model.predict(input_features)[0]
svm_proba = svm_model.predict_proba(input_features).max() * 100

# บันทึกลงตาราง Data Log สำหรับดาวน์โหลด CSV
log_entry = {
    "Timestamp": now_full_time,
    "Acceleration X (m/s²)": curr_x,
    "Acceleration Y (m/s²)": curr_y,
    "Acceleration Z (m/s²)": curr_z,
    "Absolute Accel (m/s²)": round(abs_accel, 4),
    "RF Prediction": rf_pred,
    "RF Conf (%)": round(rf_proba, 1),
    "SVM Prediction": svm_pred,
    "SVM Conf (%)": round(svm_proba, 1),
}
st.session_state.data_history.append(log_entry)

# ---------------------------------------------------------
# 8. แสดงผลข้อมูลบน Main Dashboard
# ---------------------------------------------------------
# 8.1 แสดงค่าแรงสั่นสะเทือนปัจจุบัน
st.markdown("### 📍 ค่าการสั่นสะเทือนปัจจุบัน (Real-Time)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("ค่าการสั่นแนวแกน X (m/s²)", f"{curr_x:.2f}")
m2.metric("ค่าการสั่นแนวแกน Y (m/s²)", f"{curr_y:.2f}")
m3.metric("ค่าการสั่นแนวแกน Z (m/s²)", f"{curr_z:.2f}")
m4.metric("Absolute Accel (m/s²)", f"{abs_accel:.2f}")

# 8.2 กราฟเส้น Real-Time (Plotly)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=st.session_state.history_df["Timestamp"],
        y=st.session_state.history_df["X"],
        mode="lines+markers",
        name="แกน X",
    )
)
fig.add_trace(
    go.Scatter(
        x=st.session_state.history_df["Timestamp"],
        y=st.session_state.history_df["Y"],
        mode="lines+markers",
        name="แกน Y",
    )
)
fig.add_trace(
    go.Scatter(
        x=st.session_state.history_df["Timestamp"],
        y=st.session_state.history_df["Z"],
        mode="lines+markers",
        name="แกน Z",
    )
)
fig.update_layout(
    title="📈 กราฟแสดงแนวโน้มความสั่นสะเทือน Real-Time",
    xaxis_title="เวลา",
    yaxis_title="ความเร่งสั่นสะเทือน (m/s²)",
    height=350,
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# 8.3 ผลการวิเคราะห์และทำนายสภาวะ (เปรียบเทียบ RF vs SVM)
st.markdown("---")
st.markdown("### 🤖 ผลการทำนายสภาวะการเกิด Cavitation")

col_rf, col_svm = st.columns(2)

with col_rf:
  st.subheader(f"🌲 ปั๊มทำงานที่สภาวะ: {rf_pred}")
  st.caption(f"ความมั่นใจ Real-Time: **{rf_proba:.1f}%**")
  st.success(
      f"ตัววิธี Random Forest ค่า precision เท่ากับ: **{rf_precision:.2f}**"
  )

with col_svm:
  st.subheader(f"⚡ ปั๊มทำงานที่สภาวะ: {svm_pred}")
  st.caption(f"ความมั่นใจ Real-Time: **{svm_proba:.1f}%**")
  st.info(
      "ตัววิธี Support Vector Machine ค่า precision เท่ากับ:"
      f" **{svm_precision:.2f}**"
  )

# 8.4 ตาราง Data Log และการส่งออก CSV
st.markdown("---")
st.markdown("### 📋 ตารางบันทึกประวัติการสั่นสะเทือนสด (Live Log)")
if st.session_state.data_history:
  log_df = pd.DataFrame(st.session_state.data_history)
  st.dataframe(log_df.tail(10), use_container_width=True)

  csv_data = log_df.to_csv(index=False).encode("utf-8")
  file_name = (
      "pump_vibration_log_"
      f'{pd.Timestamp.now(tz="Asia/Bangkok").strftime("%Y%m%d_%H%M%S")}.csv'
  )

  st.download_button(
      label="📥 ดาวน์โหลดประวัติข้อมูลทั้งหมดเป็น CSV",
      data=csv_data,
      file_name=file_name,
      mime="text/csv",
  )

# ---------------------------------------------------------
# 9. คำสั่ง Auto Refresh
# ---------------------------------------------------------
if auto_refresh:
  time.sleep(refresh_interval)
  st.rerun()
