import streamlit as st
import pandas as pd
import joblib

# 网页配置
st.set_page_config(page_title="Relapse Predictor", layout="centered")

st.title("🩺 Relapse Risk Calculator (RSF Model)")
st.markdown("This calculator is based on a Random Survival Forest model trained on 152 patients.")

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load('rsf_model.pkl')

try:
    pack = load_model()
    model = pack['model']
    features = pack['features']
    medians = pack['medians']

    st.sidebar.header("Patient Characteristics")
    
    # 动态生成输入框
    inputs = {}
    
    # 定义 Maintenance Therapy 的映射关系
    therapy_map = {
        "Untreated (未治疗)": 0,
        "Steroid therapy (激素治疗)": 1,
        "Novel immunosuppressants (新型免疫抑制剂)": 2
    }

    for col in features:
        val = float(medians.get(col, 0))
        
        # 1. 特别处理 Maintenance Therapy
        if "Maintenance Therapy" in col:
            selected_label = st.sidebar.selectbox(
                f"{col}", 
                options=list(therapy_map.keys()),
                index=0
            )
            inputs[col] = therapy_map[selected_label]
            
        # 2. 处理其他二分类变量 (sex, group, status)
        elif any(key in col.lower() for key in ["sex", "group", "status"]):
            inputs[col] = st.sidebar.selectbox(f"{col}", [0, 1], index=int(val))
            
        # 3. 处理连续数值变量 (NLR, MLR, age, PLR, etc.)
        else:
            inputs[col] = st.sidebar.number_input(f"{col}", value=val)

    # 预测按钮
    if st.button("Predict Risk"):
        input_df = pd.DataFrame([inputs])
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
    st.error(f"Error: {e}. Please ensure rsf_model.pkl is uploaded and matches the features.")

st.markdown("---")
st.caption("Note: For research use only. Not for clinical decision-making.")
