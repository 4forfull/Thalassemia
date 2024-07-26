# Import Important Library.
import joblib
import streamlit as st
from PIL import Image
import pandas as pd
from catboost import CatBoostClassifier

# Load Model
model = CatBoostClassifier()
model.load_model('./catboost_model_smote_是否.cbm')

model1 = CatBoostClassifier()
model1.load_model('./catboost_model_smote_ab.cbm')

# load datase
df_main = pd.read_csv('./test.csv')

# Load Image
image = Image.open('img.jpg')


# Streamlit Function For Building Button & app.
def main():
    st.image(image)

    gender = st.selectbox("Gender (Male:0, Female:1).", df_main['gender'].unique())
    age = st.number_input('Age: 0-100.', value=None)
    RBC = st.number_input('Red Blood Cell\n(RBC: 0-15).', value=None)
    Hb = st.number_input('Hemoglobin (Hb: 0-200).', value=None)
    HCT = st.number_input('Hematocrit (HCT: 0-1).', value=None)
    MCV = st.number_input('Mean Corpusular Volume (MCV: 50-200).', value=None)
    MCH = st.number_input('EMean Corpusular Hemoglobin (MCH: 10-100).', value=None)
    MCHC = st.number_input('Mean Corpuscular Hemoglobin Concentration (MCH: 100-1000).', value=None)
    RDW_CV = st.number_input('ERed Cell Distribution Width-Coefficient of Variation (RDW_CV: 0-0.5).', value=None)
    RDW_SD = st.number_input('Red Cell Distribution Width-Standard Deviation (RDW_CV: 20-150).', value=None)
    input = [gender, age, RBC, Hb, HCT, MCV, MCH, MCHC, RDW_CV, RDW_SD]
    result = ''
    if st.button('Predict', ''):
        result = prediction(input)
    temp = '''
     <div style='background-color:grey; padding:8px'>
     <h1 style='color: white  ; text-align: center;'>{}</h1>
     </div>
     '''.format(result)
    st.markdown(temp, unsafe_allow_html=True)


# Prediction Function to predict from model.
def prediction(input):
    test = [input]
    predict = model.predict(test)
    predict1 = model1.predict(test)
    if predict == 0:
        print("Diagnosis: Thalassemia")
        if predict1 == 0:
            print("Diagnosis: α-Thalassemia")
        else:
            print("Diagnosis: β-Thalassemia")
    else:
        print(
            "Diagnosis: Non-thalassemia")


if __name__ == '__main__':
    main()


