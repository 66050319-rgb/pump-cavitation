import numpy as np
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
    "ระบบประมวลผลและทำนายการเกิดโพรงอากาศในปั๊มน้ำด้วย Machine Learning"
    " (จากไฟล์ CSV)"
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

  rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)
  rf_pred = rf_model.predict(X_test)
  rf_precision = precision_score(y_test, rf_pred, average="weighted")

  svm_model = SVC(kernel="rbf", probability=True, random_state=42)
  svm_model.fit(X_train, y_train)
  svm_pred = svm_model.predict(X_test)
  svm_precision = precision_score(y_test, svm_pred, average="weighted")

  return rf_model, rf_precision, svm_model, svm_precision


rf_model, rf_precision, svm_model, svm_precision = load_and_evaluate_models()

# ---------------------------------------------------------
# 3. ส่วนอัปโหลดไฟล์ CSV (Sidebar)
# ---------------------------------------------------------
st.sidebar.header("📁 อัปโหลดไฟล์ประวัติจาก phyphox")
uploaded_file = st.sidebar.file_uploader(
    "เลือกไฟล์ CSV ที่บันทึกจาก phyphox", type=["csv"]
)

# ---------------------------------------------------------
# 4. ประมวลผลและแสดงผลข้อมูลจากไฟล์ CSV
# ---------------------------------------------------------
if uploaded_file is not None:
  try:
    df_phyphox = pd.read_csv(uploaded_file)

    # ค้นหาคอลัมน์ X, Y, Z
    x_col = [
        c
        for c in df_phyphox.columns
        if "x" in c.lower() and "abs" not in c.lower()
    ]
    y_col = [
        c
        for c in df_phyphox.columns
        if "y" in c.lower() and "abs" not in c.lower()
    ]
    z_col = [
        c
        for c in df_phyphox.columns
        if "z" in c.lower() and "abs" not in c.lower()
    ]

    if not (x_col and y_col and z_col):
      num_cols = df_phyphox.select_dtypes(include=[np.number]).columns
      if len(num_cols) >= 4:
        x_col, y_col, z_col = [num_cols[1]], [num_cols[2]], [num_cols[3]]
      elif len(num_cols) >= 3:
        x_col, y_col, z_col = [num_cols[0]], [num_cols[1]], [num_cols[2]]

    if x_col and y_col and z_col:
      st.sidebar.success("📂 อ่านข้อมูลจากไฟล์ CSV สำเร็จ!")

      col_x_name, col_y_name, col_z_name = x_col[0], y_col[0], z_col[0]

      x_vals = df_phyphox[col_x_name].values
      y_vals = df_phyphox[col_y_name].values
      z_vals = df_phyphox[col_z_name].values

      # ดึงค่าแถวล่าสุด
      curr_x, curr_y, curr_z = (
          float(x_vals[-1]),
          float(y_vals[-1]),
          float(z_vals[-1]),
      )
      abs_accel = np.sqrt(curr_x**2 + curr_y**2 + curr_z**2)

      # ประมวลผลข้อมูลทั้งไฟล์
      csv_log = []
      time_col = [c for c in df_phyphox.columns if "time" in c.lower()]

      for i in range(len(df_phyphox)):
        xv, yv, zv = float(x_vals[i]), float(y_vals[i]), float(z_vals[i])
        magv = np.sqrt(xv**2 + yv**2 + zv**2)
        feat = np.array([[xv, yv, zv]])

        rf_p = rf_model.predict(feat)[0]
        rf_c = rf_model.predict_proba(feat).max() * 100

        svm_p = svm_model.predict(feat)[0]
        svm_c = svm_model.predict_proba(feat).max() * 100

        t_val = df_phyphox[time_col[0]].iloc[i] if time_col else f"Point {i+1}"

        csv_log.append({
            "Timestamp": t_val,
            "Acceleration X (m/s²)": round(xv, 4),
            "Acceleration Y (m/s²)": round(yv, 4),
            "Acceleration Z (m/s²)": round(zv, 4),
            "Absolute Accel (m/s²)": round(magv, 4),
            "RF Prediction": rf_p,
            "RF Conf (%)": round(rf_c, 1),
            "SVM Prediction": svm_p,
            "SVM Conf (%)": round(svm_c, 1),
        })

      display_log_df = pd.DataFrame(csv_log)

      # ---------------------------------------------------------
      # คำนวณ Precision จริงจากไฟล์ CSV (ถ้ามีคอลัมน์เฉลย/Label อยู่ในไฟล์)
      # ---------------------------------------------------------
      label_col = [
          c
          for c in df_phyphox.columns
          if c.lower() in ["label", "status", "target", "class", "actual"]
      ]

      if label_col:
        y_true = df_phyphox[label_col[0]]
        y_rf_all = display_log_df["RF Prediction"]
        y_svm_all = display_log_df["SVM Prediction"]

        current_rf_precision = precision_score(
            y_true, y_rf_all, average="weighted", zero_division=0
        )
        current_svm_precision = precision_score(
            y_true, y_svm_all, average="weighted", zero_division=0
        )
        precision_title = "Precision จริงจากไฟล์นี้"
      else:
        current_rf_precision = rf_precision
        current_svm_precision = svm_precision
        precision_title = "Model Precision (อ้างอิง)"

      # แสดงค่าของแถวล่าสุด
      st.markdown("### 📍 ค่าการสั่นสะเทือน (จากไฟล์ CSV แถวล่าสุด)")
      m1, m2, m3, m4 = st.columns(4)
      m1.metric("ค่าการสั่นแนวแกน X (m/s²)", f"{curr_x:.2f}")
      m2.metric("ค่าการสั่นแนวแกน Y (m/s²)", f"{curr_y:.2f}")
      m3.metric("ค่าการสั่นแนวแกน Z (m/s²)", f"{curr_z:.2f}")
      m4.metric("Absolute Accel (m/s²)", f"{abs_accel:.2f}")

      # คำนวณทำนายผลของแถวล่าสุด
      input_features = np.array([[curr_x, curr_y, curr_z]])
      rf_pred = rf_model.predict(input_features)[0]
      rf_proba = rf_model.predict_proba(input_features).max() * 100
      svm_pred = svm_model.predict(input_features)[0]
      svm_proba = svm_model.predict_proba(input_features).max() * 100

      # แสดงผลการทำนาย
      st.markdown("---")
      st.markdown("### 🤖 ผลการทำนายสภาวะการเกิด Cavitation")

      col_rf, col_svm = st.columns(2)

      with col_rf:
        st.subheader(f"🌲 ปั๊มทำงานที่สภาวะ: {rf_pred}")
        st.caption(f"ความมั่นใจ: **{rf_proba:.1f}%**")
        st.success(
            f"ตัววิธี Random Forest {precision_title} เท่ากับ:"
            f" **{current_rf_precision:.2f}**"
        )

      with col_svm:
        st.subheader(f"⚡ ปั๊มทำงานที่สภาวะ: {svm_pred}")
        st.caption(f"ความมั่นใจ: **{svm_proba:.1f}%**")
        st.info(
            f"ตัววิธี Support Vector Machine {precision_title} เท่ากับ:"
            f" **{current_svm_precision:.2f}**"
        )

      # กราฟแสดงแนวโน้มความสั่นสะเทือน
      st.markdown("---")
      fig = go.Figure()
      fig.add_trace(
          go.Scatter(
              x=display_log_df["Timestamp"],
              y=display_log_df["Acceleration X (m/s²)"],
              mode="lines+markers",
              name="แกน X",
          )
      )
      fig.add_trace(
          go.Scatter(
              x=display_log_df["Timestamp"],
              y=display_log_df["Acceleration Y (m/s²)"],
              mode="lines+markers",
              name="แกน Y",
          )
      )
      fig.add_trace(
          go.Scatter(
              x=display_log_df["Timestamp"],
              y=display_log_df["Acceleration Z (m/s²)"],
              mode="lines+markers",
              name="แกน Z",
          )
      )
      fig.update_layout(
          title="📈 กราฟแสดงแนวโน้มความสั่นสะเทือนจากไฟล์ CSV",
          xaxis_title="เวลา / ลำดับข้อมูล",
          yaxis_title="ความเร่งสั่นสะเทือน (m/s²)",
          height=350,
          margin=dict(l=20, r=20, t=40, b=20),
      )
      st.plotly_chart(fig, use_container_width=True)

      # แสดงตารางประวัติผลทำนาย
      st.markdown("---")
      st.markdown("### 📋 ตารางผลการวิเคราะห์ไฟล์ CSV ย้อนหลัง")
      st.dataframe(display_log_df, use_container_width=True)

      csv_data = display_log_df.to_csv(index=False).encode("utf-8")
      st.download_button(
          label="📥 ดาวน์โหลดประวัติผลการวิเคราะห์ทั้งหมดเป็น CSV",
          data=csv_data,
          file_name="phyphox_predicted_result.csv",
          mime="text/csv",
      )
    else:
      st.error("ไม่พบคอลัมน์ X, Y, Z ในไฟล์ CSV กรุณาตรวจสอบไฟล์อีกครั้ง")
  except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ CSV: {e}")
else:
  st.info("👈 กรุณาอัปโหลดไฟล์ CSV จาก phyphox ที่แถบด้านซ้ายเพื่อเริ่มวิเคราะห์")
