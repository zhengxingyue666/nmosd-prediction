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
    for col in features:
        val = float(medians.get(col, 0))
        if "sex" in col.lower() or "group" in col.lower() or "status" in col.lower():
            inputs[col] = st.sidebar.selectbox(f"{col}", [0, 1], index=int(val))
        else:
            inputs[col] = st.sidebar.number_input(f"{col}", value=val)

    # 预测按钮
    if st.button("Predict Risk"):
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)[0]
        
        st.write("---")
        st.subheader("Results")
        st.metric("Estimated Median Relapse-Free Time", f"{round(prediction, 2)} Months")
        
        if prediction < 24:
            st.error("High Risk of Relapse")
        elif prediction < 60:
            st.warning("Moderate Risk of Relapse")
        else:
            st.success("Low Risk of Relapse")

except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure rsf_model.pkl is uploaded.")

st.markdown("---")
st.caption("Note: For research use only. Not for clinical decision-making.")