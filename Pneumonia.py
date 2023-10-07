from base64 import a85decode
from copyreg import pickle
import streamlit as st
import numpy as np
from numpy import array
from numpy import argmax
from numpy import genfromtxt
import pandas as pd
import shap
import catboost as cb
from catboost.datasets import titanic
import plotly.graph_objs as go 
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediction probability of pneumonia after hip surgery for elderly patients", layout="wide")

plt.style.use('default')

df=pd.read_csv('traindata1.csv',encoding='utf8')
trainy=df.Pneumonia
trainx=df.drop('Pneumonia',axis=1)

Cb = cb.CatBoostClassifier(
         iterations=200,
         od_type='Iter',
         od_wait=600,
         max_depth=4,
         learning_rate=0.02,
         l2_leaf_reg=12,
         random_seed=1,
         metric_period=50,
         fold_len_multiplier=1.2,
         loss_function='Logloss',
         logging_level='Verbose')
Cb.fit(trainx, trainy)

###side-bar
def user_input_features():
    st.title("Prediction probability of pneumonia after hip surgery for elderly patients")
    st.sidebar.header('User input parameters below')
    a1=st.sidebar.number_input("Age",min_value=65,max_value=120)
    a2=st.sidebar.number_input("CRP(mg/L)",min_value=0.01,max_value=None,step=0.01)
    a3=st.sidebar.number_input("Preoperative length of stay(day)",min_value=0,max_value=None)
    a5=st.sidebar.selectbox('Smoking',('Never smoking','Former smoking','Current smoking'))
    a6=st.sidebar.selectbox('Preoperative SpO2',('≥96%','<96%'))
    a7=st.sidebar.selectbox('ASA physical status',('Ⅰ/Ⅱ','Ⅲ/Ⅳ/Ⅴ'))
    st.sidebar.markdown('The calculation of mFI-5')    
    m1=st.sidebar.selectbox('Functional status',('Independent','Dependent'))
    m2=st.sidebar.selectbox('Diabetes mellitus',('No','Yes'))
    m3=st.sidebar.selectbox('Chronic obstructive pulmonary disease',('No','Yes'))
    m4=st.sidebar.selectbox('Heart failure',('No','Yes'))
    m5=st.sidebar.selectbox('Hypertension',('No','Yes'))
    st.sidebar.write("Note: Never smoking=0, Former smoking=1, Current smoking=2, ≥96%=0, <96%=1, Ⅰ/Ⅱ=0, Ⅲ/Ⅳ/Ⅴ=1, Independent=0, Dependent=1, No=0, Yes=1")
    result=""
    if m1=="Dependent":
        m1=1
    else: 
        m1=0 
    if m2=="Yes":
        m2=1
    else: 
        m2=0 
    if m3=="Yes":
        m3=1
    else: 
        m3=0 
    if m4=="Yes":
        m4=1
    else: 
        m4=0 
    if m5=="Yes":
        m5=1
    else: 
        m5=0 
    a4=m1+m2+m3+m4+m5
    if a5=="Never smoking":
        a5=0
    elif a5=='Former smoking':
        a5=1
    else: 
        a5=2 
    if a6=="<96%":
        a6=1
    else: 
        a6=0 
    if a7=="Ⅲ/Ⅳ/Ⅴ":
        a7=1
    else: 
        a7=0 
   
    output=[a1,a2,a3,a4,a5,a6,a7]
    int_features=[int(x) for x in output]
    final_features=np.array(int_features)
    patient1=pd.DataFrame(final_features)
    patient=pd.DataFrame(patient1.values.T,columns=trainx.columns)
    prediction=Cb.predict_proba(patient)
    prediction=float(prediction[:, 1])
    def predict_PPCs():
        prediction=round(user_input_features[:, 1],3)
        return prediction
    result=""
    if st.button("Predict"):
        st.success('The probability of pneumonia for the patient: {:.1f}%'.format(prediction*100))
        if prediction>0.151:
            b="High risk"
        else:
            b="Low risk"
        st.success('The risk group'+": "+b)
        explainer_Cb = shap.TreeExplainer(Cb)
        shap_values= explainer_Cb(patient)
        shap.plots.waterfall(shap_values[0])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write("Waterfall plot analysis of Pneumonia for the patient:")
        st.pyplot(bbox_inches='tight')
        st.write("Abbreviations: CRP, C-reactive protein; mFI-5, modified five-item frailty index; SpO2, Peripheral capillary oxygen saturation; ASA, American Society of Anesthesiologists.")
        st.write("Note: The mFI-5 score based on five variables including functional status, diabetes mellitus, chronic obstructive pulmonary disease, heart failure, and hypertension")
    if st.button("Reset"):
        st.write("")
    st.markdown("*Statement: this website will not record or store any information inputed.")
    st.write("2023 Nanjing First Hospital, Nanjing Medical University. All Rights Reserved ")
    st.write("✉ Contact Us: zoujianjun100@126.com")

if __name__ == '__main__':
    user_input_features()