
import pandas as pd
import numpy as np
import streamlit as st
import sklearn
import imblearn
import joblib

## Create Get_Log Function
def Get_Log(v):
    return np.log(v.astype(float))

Model = joblib.load('../Data/Model.pkl')
Inputs = joblib.load('../Data/Inputs.pkl')

def Predict(Gender, Married, Dependents, Education, Self_Employed, CoapplicantIncome, Credit_History, Property_Area, Loan_Per_Month, Income_After_Loan, Income_Exceeds_Loan):
    df_test = pd.DataFrame(columns=Inputs)
    df_test.at[0,'Gender'] = Gender
    df_test.at[0,'Married'] = Married
    df_test.at[0,'Dependents'] = Dependents
    df_test.at[0,'Education'] = Education
    df_test.at[0,'Self_Employed'] = Self_Employed
    
    df_test.at[0,'CoapplicantIncome'] = CoapplicantIncome
    df_test.at[0,'Credit_History'] = Credit_History
    df_test.at[0,'Property_Area'] = Property_Area
    df_test.at[0,'Loan_Per_Month'] = Loan_Per_Month
    df_test.at[0,'Income_After_Loan'] = Income_After_Loan
    df_test.at[0,'Income_Exceeds_Loan'] = Income_Exceeds_Loan
    
    res = Model.predict(df_test)
    return res[0]
    
    
def Main():
    st.markdown('<p style="font-size:50px;text-align:center;"><strong>Loan Detection</strong></p>',unsafe_allow_html=True)
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Married = st.selectbox('Married',['Yes','No'])
    Dependents = st.selectbox('Number of Dependents',[0,1,2,3])
    Education = st.selectbox('Education',['Graduate', 'Not Graduate'])
    Self_Employed = st.selectbox('Self_Employed',['Yes','No'])
    ApplicantIncome = st.slider("ApplicantIncome ",min_value=100, max_value=90000, step=100, value=10000)
    CoapplicantIncome = st.slider("CoapplicantIncome ",min_value=0, max_value=90000, step=10, value=10000)
    LoanAmount = st.slider("LoanAmount ",min_value=20000, max_value=900000, step=10, value=30000)
    Loan_Amount_Term = st.slider("Loan_Amount_Term ",min_value=10, max_value=500, step=1, value=100)
    Credit_History = st.selectbox('Credit History',[0,1])
    Property_Area = st.selectbox('Property Area',['Urban', 'Rural', 'Semiurban'])
    ## Feature Engineering
    ## Calculate Loan_Per_Month
    Loan_Per_Month = LoanAmount / Loan_Amount_Term
    Loan_Per_Month = np.round(Loan_Per_Month,0)
    
    ## Calculate Income_After_Loan
    Income_After_Loan = ApplicantIncome - Loan_Per_Month
    
    ## Extract Income_Exceeds_Loan
    if ApplicantIncome > Loan_Per_Month:
        Income_Exceeds_Loan= 1
    else: 
        Income_Exceeds_Loan = 0
    
    ## Call Predict Function
    if st.button("Predict"):
        res = Predict(Gender, Married, Dependents, Education, Self_Employed, CoapplicantIncome, Credit_History, Property_Area, Loan_Per_Month, Income_After_Loan, Income_Exceeds_Loan)
        res_dict = {0:'Refused', 1:'Accepted'}
        st.text(f'Your Loan Request is {res_dict[res]}')
    
    
Main()
