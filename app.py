import streamlit as st
import pandas as pd
import joblib

# 网页基础配置
st.set_page_config(page_title="Relapse Predictor", layout="centered")
st.title("🩺 Relapse Risk Calculator (RSF Model)")
st.markdown("This calculator is based on a Random Survival Forest model trained on 152 NMOSD patients.")

# 固定8个核心特征（和训练完全一致）
FEATURES = [
    "NLR",
    "MLR",
    "Qalb",
    "QIgM",
    "QIgG",
    "EDSS score",
    "Serum AQP4-IgG",
    "Maintenance Therapy"
]

# 加载新训练的模型
@st.cache_resource
def load_model():
    return joblib.load('rsf_web_model.pkl')

try:
    pack = load_model()
    model = pack['model']
    medians = pack['medians']

    st.sidebar.header("Patient Clinical Characteristics")
    inputs = {}

    # 连续变量
    inputs["NLR"] = st.sidebar.number_input("NLR", value=float(medians["NLR"]))
    inputs["MLR"] = st.sidebar.number_input("MLR", value=float(medians["MLR"]))
    inputs["Qalb"] = st.sidebar.number_input("Qalb", value=float(medians["Qalb"]))
    inputs["QIgM"] = st.sidebar.number_input("QIgM", value=float(medians["QIgM"]))
    inputs["QIgG"] = st.sidebar.number_input("QIgG", value=float(medians["QIgG"]))
    inputs["EDSS score"] = st.sidebar.number_input("EDSS score", value=float(medians["EDSS score"]))

    # AQP4‑IgG：0=阴性，1=阳性
    aqp4_map = {"Negative (阴性)": 0, "Positive (阳性)": 1}
    selected_aqp4 = st.sidebar.selectbox("Serum AQP4‑IgG", list(aqp4_map.keys()))
    inputs["Serum AQP4-IgG"] = aqp4_map[selected_aqp4]

    # 维持治疗三分类（规范学术命名）
    therapy_map = {
        "Untreated (未治疗)": 0,
        "Conventional immunosuppressive therapy (传统激素/免疫抑制治疗)": 1,
        "Targeted biologic therapy (靶向生物制剂治疗)": 2
    }
    selected_therapy = st.sidebar.selectbox("Maintenance Therapy", list(therapy_map.keys()))
    inputs["Maintenance Therapy"] = therapy_map[selected_therapy]

    # 预测
    if st.button("Predict Risk"):
        input_df = pd.DataFrame([inputs])[FEATURES]
        prediction = model.predict(input_df)[0]

        st.write("---")
        st.subheader("Prediction Results")
        st.metric("Estimated Median Relapse‑Free Time", f"{round(prediction, 2)} Months")

        if prediction < 24:
            st.error("⚠️ High Risk of Relapse")
        elif prediction < 60:
            st.warning("🟠 Moderate Risk of Relapse")
        else:
            st.success("✅ Low Risk of Relapse")

except Exception as e:
    st.error(f"Running Error: {e}")

st.markdown("---")
st.caption("Note: This prediction tool is for research purposes only, not for direct clinical decision‑making.")
