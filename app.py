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
    " Learning"
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
# 4. ตั้งค่า MQTT Client (รับค่า Real-Time)
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
# 5. ฟังก์ชันสร้างไฟล์คอนฟิก phyphox XML
# ---------------------------------------------------------
PHYPHOX_XML = f"""<?xml version="1.0" encoding="utf-8"?>
<phyphox version="1.7">
    <title>Pump Cavitation Monitor</title>
    <category>IoT Dashboard</category>
    <data-containers>
        <container size="1">x</container>
        <container size="1">y</container>
        <container size="1">z</container>
    </data-containers>
    <input>
        <sensor type="accelerometer" average="false">
            <output component="x">x</output>
            <output component="y">y</output>
            <output component="z">z</output>
        </sensor>
    </input>
    <network>
        <mqtt service="mqtt/json" broker="{MQTT_BROKER}" port="{MQTT_PORT}" sendTopic="{MQTT_TOPIC}" interval="0.2">
            <send id="x">x</send>
            <send id="y">y</send>
            <send id="z">z</send>
        </mqtt>
    </network>
</phyphox>"""

# ---------------------------------------------------------
# 6. ส่วนควบคุมหน้าจอ (Sidebar Controls)
# ---------------------------------------------------------
st.sidebar.header("🎛️ ระบบควบคุม")
auto_refresh = st.sidebar.checkbox(
    "เปิดใช้งาน Real-Time Auto Refresh", value=True
)
refresh_interval = st.sidebar.slider("ความถี่ในการรีเฟรช (วินาที)", 0.5, 3.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("📱 เชื่อมต่อ phyphox (iOS / Android)")
st.sidebar.download_button(
    label="📥 ดาวน์โหลดไฟล์คอนฟิก Pump_IoT.phyphox",
    data=PHYPHOX_XML,
    file_name="Pump_IoT.phyphox",
    mime="application/xml",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📁 อัปโหลดไฟล์ประวัติจาก phyphox")
uploaded_file = st.sidebar.file_uploader(
    "เลือกไฟล์ CSV ที่บันทึกจาก phyphox", type=["csv"]
)

# ---------------------------------------------------------
# 7. ตรวจสอบแหล่งข้อมูล (CSV vs Real-Time MQTT)
# ---------------------------------------------------------
using_csv = False

if uploaded_file is not None:
  try:
    df_phyphox = pd.read_csv(uploaded_file)
    numeric_cols = df_phyphox.select_dtypes(include=[np.number]).columns
    feature_cols = [
        col for col in numeric_cols if "time" not in col.lower()
    ][:3]

    if len(feature_cols) >= 3:
      using_csv = True
      st.sidebar.success("📂 กำลังวิเคราะห์ข้อมูลจากไฟล์ CSV")

      # ดึงคอลัมน์ X, Y, Z จาก CSV
      X_batch = df_phyphox[feature_cols].values

      # กำหนดให้ค่าปัจจุบันดึงมาจากแถวล่าสุดของ CSV
      last_row = X_batch[-1]
      curr_x, curr_y, curr_z = float(last_row[0]), float(last_row[1]), float(last_row[2])

      # สร้าง Data Log ประวัติทั้งหมดจากไฟล์ CSV
      csv_log = []
      for idx, row in enumerate(X_batch):
        x_val, y_val, z_val = float(row[0]), float(row[1]), float(row[2])
        mag_val = np.sqrt(x_val**2 + y_val**2 + z_val**2)
        feat = np.array([[x_val, y_val, z_val]])

        rf_p = rf_model.predict(feat)[0]
        rf_c = rf_model.predict_proba(feat).max() * 100

        svm_p = svm_model.predict(feat)[0]
        svm_c = svm_model.predict_proba(feat).max() * 100

        # ใช้คอลัมน์ Time จาก CSV หรือสร้าง Timestamp ปลอม
        t_stamp = (
            df_phyphox.iloc[idx, 0]
            if "time" in str(df_phyphox.columns[0]).lower()
            else f"Row {idx+1}"
        )

        csv_log.append({
            "Timestamp": t_stamp,
            "Acceleration X (m/s²)": round(x_val, 4),
            "Acceleration Y (m/s²)": round(y_val, 4),
            "Acceleration Z (m/s²)": round(z_val, 4),
            "Absolute Accel (m/s²)": round(mag_val, 4),
            "RF Prediction": rf_p,
            "RF Conf (%)": round(rf_c, 1),
            "SVM Prediction": svm_p,
            "SVM Conf (%)": round(svm_c, 1),
        })

      # อัปเดตตาราง Log ให้แสดงข้อมูลทั้งหมดจาก CSV
      display_log_df = pd.DataFrame(csv_log)

      # อัปเดตกราฟประวัติ
      chart_df = pd.DataFrame({
          "Timestamp": [item["Timestamp"] for item in csv_log],
          "X": [item["Acceleration X (m/s²)"] for item in csv_log],
          "Y": [item["Acceleration Y (m/s²)"] for item in csv_log],
          "Z": [item["Acceleration Z (m/s²)"] for item in csv_log],
      }).tail(50)

    else:
      st.sidebar.error("ไฟล์ CSV ไม่มีคอลัมน์ X, Y, Z ตัวเลขที่ถูกต้อง")
  except Exception as e:
    st.sidebar.error(f"อ่านไฟล์ CSV ไม่สำเร็จ: {e}")

if not using_csv:
  # ใช้ค่า Real-Time จาก MQTT สด
  curr_x = st.session_state.latest_sensor["x"]
  curr_y = st.session_state.latest_sensor["y"]
  curr_z = st.session_state.latest_sensor["z"]

  now_time_str = pd.Timestamp.now(tz="Asia/Bangkok").strftime("%H:%M:%S")
  now_full_time = pd.Timestamp.now(tz="Asia/Bangkok").strftime(
      "%Y-%m-%d %H:%M:%S"
  )

  # บันทึกข้อมูล Real-time ลง session state
  new_row = pd.DataFrame(
      [{"Timestamp": now_time_str, "X": curr_x, "Y": curr_y, "Z": curr_z}]
  )
  st.session_state.history_df = pd.concat(
      [st.session_state.history_df, new_row], ignore_index=True
  ).tail(50)
  chart_df = st.session_state.history_df

# ---------------------------------------------------------
# 8. คำนวณผลการทำนายด้วย ML โมเดล
# ---------------------------------------------------------
abs_accel = np.sqrt(curr_x**2 + curr_y**2 + curr_z**2)
input_features = np.array([[curr_x, curr_y, curr_z]])

rf_pred = rf_model.predict(input_features)[0]
rf_proba = rf_model.predict_proba(input_features).max() * 100

svm_pred = svm_model.predict(input_features)[0]
svm_proba = svm_model.predict_proba(input_features).max() * 100

if not using_csv:
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
  display_log_df = pd.DataFrame(st.session_state.data_history)

# ---------------------------------------------------------
# 9. แสดงผลบน Main Dashboard
# ---------------------------------------------------------
# 9.1 ค่าแรงสั่นสะเทือนปัจจุบัน
st.markdown("### 📍 ค่าการสั่นสะเทือน (ปัจจุบัน/จาก CSV)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("ค่าการสั่นแนวแกน X (m/s²)", f"{curr_x:.2f}")
m2.metric("ค่าการสั่นแนวแกน Y (m/s²)", f"{curr_y:.2f}")
m3.metric("ค่าการสั่นแนวแกน Z (m/s²)", f"{curr_z:.2f}")
m4.metric("Absolute Accel (m/s²)", f"{abs_accel:.2f}")

# 9.2 ผลการทำนายสภาวะ Cavitation (RF & SVM)
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

# 9.3 กราฟเส้นแสดงแนวโน้มความสั่นสะเทือน
st.markdown("---")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=chart_df["Timestamp"],
        y=chart_df["X"],
        mode="lines+markers",
        name="แกน X",
    )
)
fig.add_trace(
    go.Scatter(
        x=chart_df["Timestamp"],
        y=chart_df["Y"],
        mode="lines+markers",
        name="แกน Y",
    )
)
fig.add_trace(
    go.Scatter(
        x=chart_df["Timestamp"],
        y=chart_df["Z"],
        mode="lines+markers",
        name="แกน Z",
    )
)
fig.update_layout(
    title="📈 กราฟแสดงแนวโน้มความสั่นสะเทือน",
    xaxis_title="เวลา / ลำดับข้อมูล",
    yaxis_title="ความเร่งสั่นสะเทือน (m/s²)",
    height=350,
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# 9.4 ตารางบันทึกประวัติการสั่นสะเทือน (Live Log / CSV Log)
st.markdown("---")
st.markdown("### 📋 ตารางบันทึกประวัติการสั่นสะเทือนสด (Live Log)")
if not display_log_df.empty:
  st.dataframe(display_log_df.tail(10), use_container_width=True)

  csv_data = display_log_df.to_csv(index=False).encode("utf-8")
  file_name = (
      "pump_vibration_predicted_log_"
      f'{pd.Timestamp.now(tz="Asia/Bangkok").strftime("%Y%m%d_%H%M%S")}.csv'
  )

  st.download_button(
      label="📥 ดาวน์โหลดประวัติข้อมูลทั้งหมดเป็น CSV",
      data=csv_data,
      file_name=file_name,
      mime="text/csv",
  )

# ---------------------------------------------------------
# 10. คำสั่ง Auto Refresh (ทำงานเฉพาะเมื่อไม่ได้ใช้ CSV)
# ---------------------------------------------------------
if auto_refresh and not using_csv:
  time.sleep(refresh_interval)
  st.rerun()
