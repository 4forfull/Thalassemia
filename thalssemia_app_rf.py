# Import Important Library
import streamlit as st
from PIL import Image
import pandas as pd
import pickle  # 新增
# 或者用 joblib
# import joblib

# Set Page Title
st.set_page_config(page_title="Thalassemia")

# ===== Load Models (.pkl) =====
with open('RF_model_YN.pkl', 'rb') as f:
    model = pickle.load(f)

with open('RF_model_ab.pkl', 'rb') as f:
    model1 = pickle.load(f)

# 如果模型是 joblib 保存的：
# model = joblib.load('.../catboost_model_smote_YN.pkl')
# model1 = joblib.load('.../catboost_model_smote_ab.pkl')

# ===== Load Dataset =====
df_main = pd.read_csv('D:/临床病例/地贫最终数据/软件/test.csv')

# ===== Load Image =====
image = Image.open('img.jpg')


# Streamlit Function For Building Button & app.
def main():
    st.image(image)

    gender = st.selectbox("Gender (Male:0, Female:1).", df_main['gender'].unique())
    age = st.number_input('Age: 0-100.', value=None)
    RBC = st.number_input('Red Blood Cell\n(RBC: 0-15).', value=None)
    Hb = st.number_input('Hemoglobin (Hb: 0-200).', value=None)
    HCT = st.number_input('Hematocrit (HCT: 0-1).', value=None)
    MCV = st.number_input('Mean Corpuscular Volume (MCV: 50-200).', value=None)
    MCH = st.number_input('Mean Corpuscular Hemoglobin (MCH: 10-100).', value=None)
    MCHC = st.number_input('Mean Corpuscular Hemoglobin Concentration (MCHC: 100-1000).', value=None)
    RDW_CV = st.number_input('Red Cell Distribution Width-Coefficient of Variation (RDW_CV: 0-0.5).', value=None)
    RDW_SD = st.number_input('Red Cell Distribution Width-Standard Deviation (RDW_SD: 20-150).', value=None)
    input_data = [gender, age, RBC, Hb, HCT, MCV, MCH, MCHC, RDW_CV, RDW_SD]

    if st.button('Predict'):
        result = prediction(input_data)
        st.markdown(
            f"<div style='background-color:grey; padding:8px'><h1 style='color: white; text-align: center;'>{result}</h1></div>",
            unsafe_allow_html=True)


# Prediction Function
def prediction(input_data):
    test = [input_data]
    predict = model.predict(test)
    predict1 = model1.predict(test)
    if predict == 0:
        return "Diagnosis: Non-thalassemia"
    else:
        if predict1 == 0:
            return "Diagnosis: α-Thalassemia"
        else:
            return "Diagnosis: β-Thalassemia"


if __name__ == '__main__':
    main()
