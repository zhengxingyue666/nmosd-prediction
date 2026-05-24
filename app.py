import streamlit as st
import pandas as pd
import joblib

# 网页配置
st.set_page_config(page_title="Relapse Predictor", layout="centered")

st.title("🩺 Relapse Risk Calculator (RSF Model)")
st.markdown("This calculator is based on a Random Survival Forest model trained on 152 patients.")

# 固定使用SHAP筛选的前8个核心特征
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

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load('rsf_model.pkl')

try:
    pack = load_model()
    model = pack['model']
    medians = pack['medians']

    st.sidebar.header("Patient Characteristics")
    
    inputs = {}

    # 1. 连续数值变量
    inputs["NLR"] = st.sidebar.number_input("NLR", value=float(medians.get("NLR", 3.0)))
    inputs["MLR"] = st.sidebar.number_input("MLR", value=float(medians.get("MLR", 0.5)))
    inputs["Qalb"] = st.sidebar.number_input("Qalb", value=float(medians.get("Qalb", 40)))
    inputs["QIgM"] = st.sidebar.number_input("QIgM", value=float(medians.get("QIgM", 1.2)))
    inputs["QIgG"] = st.sidebar.number_input("QIgG", value=float(medians.get("QIgG", 12)))
    inputs["EDSS score"] = st.sidebar.number_input("EDSS score", value=float(medians.get("EDSS score", 2.0)))

    # 2. 二分类：Serum AQP4-IgG（0=阴性，1=阳性）
    aqp4_map = {"Negative (阴性, 0)": 0, "Positive (阳性, 1)": 1}
    selected_aqp4 = st.sidebar.selectbox("Serum AQP4-IgG", options=list(aqp4_map.keys()))
    inputs["Serum AQP4-IgG"] = aqp4_map[selected_aqp4]

    # 3. 三分类：Maintenance Therapy（严格匹配指南分类）
therapy_map = {
    "Untreated (未治疗, 0)": 0,
    "Conventional immunosuppressive therapy (传统激素/免疫抑制治疗, 1)": 1,
    "Targeted biologic therapy (靶向生物制剂治疗, 2)": 2
}
selected_therapy = st.sidebar.selectbox("Maintenance Therapy", options=list(therapy_map.keys()))
inputs["Maintenance Therapy"] = therapy_map[selected_therapy]

    # 预测按钮
    if st.button("Predict Risk"):
        input_df = pd.DataFrame([inputs])[FEATURES]  # 严格按特征顺序输入
        prediction = model.predict(input_df)[0]
        
        st.write("---")
        st.subheader("Results")
        
        # 显示预测的数值结果
        st.metric("Estimated Median Relapse-Free Time", f"{round(prediction, 2)} Months")
        
        # 风险分层显示
        if prediction < 24:
            st.error("⚠️ High Risk of Relapse")
        elif prediction < 60:
            st.warning("🟠 Moderate Risk of Relapse")
        else:
            st.success("✅ Low Risk of Relapse")

except Exception as e:
    st.error(f"Error: {e}. Please ensure rsf_model.pkl is uploaded and features match the top 8 SHAP features.")

st.markdown("---")
st.caption("Note: For research use only. Not for clinical decision-making.")
