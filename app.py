import streamlit as st
import pandas as pd
import joblib

# 网页基础配置
st.set_page_config(page_title="Relapse Predictor", layout="centered")

st.title("🩺 Relapse Risk Calculator (RSF Model)")
st.markdown("This calculator is based on a Random Survival Forest model trained on 152 NMOSD patients.")

# 固定使用SHAP筛选的前8个核心预测特征（严格顺序）
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

# 加载训练好的RSF模型
@st.cache_resource
def load_model():
    return joblib.load('rsf_model.pkl')

try:
    pack = load_model()
    model = pack['model']
    medians = pack['medians']

    st.sidebar.header("Patient Clinical Characteristics")
    
    inputs = {}

    # 1. 连续型变量输入
    inputs["NLR"] = st.sidebar.number_input("NLR (中性粒细胞/淋巴细胞比值)", value=float(medians.get("NLR", 3.0)))
    inputs["MLR"] = st.sidebar.number_input("MLR (单核细胞/淋巴细胞比值)", value=float(medians.get("MLR", 0.5)))
    inputs["Qalb"] = st.sidebar.number_input("Qalb (白蛋白比值)", value=float(medians.get("Qalb", 40)))
    inputs["QIgM"] = st.sidebar.number_input("QIgM (IgM比值)", value=float(medians.get("QIgM", 1.2)))
    inputs["QIgG"] = st.sidebar.number_input("QIgG (IgG比值)", value=float(medians.get("QIgG", 12)))
    inputs["EDSS score"] = st.sidebar.number_input("EDSS score (扩展残疾状态量表)", value=float(medians.get("EDSS score", 2.0)))

    # 2. 二分类：血清AQP4‑IgG（0=阴性，1=阳性）
    aqp4_map = {
        "Negative (阴性, 0)": 0,
        "Positive (阳性, 1)": 1
    }
    selected_aqp4 = st.sidebar.selectbox("Serum AQP4‑IgG (水通道蛋白4抗体)", options=list(aqp4_map.keys()))
    inputs["Serum AQP4-IgG"] = aqp4_map[selected_aqp4]

    # 3. 规范三分类维持治疗（NMOSD临床指南标准分类）
    therapy_map = {
        "Untreated (未治疗, 0)": 0,
        "Conventional immunosuppressive therapy (传统激素/免疫抑制治疗, 1)": 1,
        "Targeted biologic therapy (靶向生物制剂治疗, 2)": 2
    }
    selected_therapy = st.sidebar.selectbox("Maintenance Therapy (维持治疗方案)", options=list(therapy_map.keys()))
    inputs["Maintenance Therapy"] = therapy_map[selected_therapy]

    # 预测触发按钮
    if st.button("Predict Risk"):
        # 严格匹配模型特征顺序，避免维度错误
        input_df = pd.DataFrame([inputs])[FEATURES]
        prediction = model.predict(input_df)[0]
        
        st.write("---")
        st.subheader("Prediction Results")
        
        # 输出中位无复发时间
        st.metric("Estimated Median Relapse‑Free Time", f"{round(prediction, 2)} Months")
        
        # 复发风险分层（临床通用分层标准）
        if prediction < 24:
            st.error("⚠️ High Risk of Relapse")
        elif prediction < 60:
            st.warning("🟠 Moderate Risk of Relapse")
        else:
            st.success("✅ Low Risk of Relapse")

except Exception as e:
    st.error(f"Running Error: {e}. Please confirm rsf_model.pkl file is uploaded and feature order matches the top 8 SHAP features.")

# 免责声明
st.markdown("---")
st.caption("Note: This prediction tool is for research purposes only, not for direct clinical decision‑making.")
