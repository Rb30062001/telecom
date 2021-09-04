import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
pickle_in = open("pietmodel.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv',nrows=1000)
X = dataset.iloc[:,1:20].values

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
for i in range(5,17):
  X[:, i] = labelencoder_X.fit_transform(X[:, i])


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

def predict_note_authentication(gender	,SeniorCitizen,	Partner,	Dependents,	tenure,	PhoneService,
                                MultipleLines,	InternetService,	OnlineSecurity,	OnlineBackup,	DeviceProtection,	TechSupport,
                                StreamingTV	,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges):
  output= model.predict(sc.transform([[gender	,SeniorCitizen,	Partner,	Dependents,	tenure,	PhoneService,
                                MultipleLines,	InternetService,	OnlineSecurity,	OnlineBackup,	DeviceProtection,	TechSupport,
                                StreamingTV	,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges]]))
  print("Customer will leave =", output)
  if output==[1]:
    prediction="Customer will Leave"
  else:
    prediction="Customer will not Leave"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:brown;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:45px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning Project</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Telecom Customer Churn Prediction")

    gender=st.number_input('Insert 0 For Male 1 For Female ',0,1)
    SeniorCitizen=st.number_input('Insert a SeniorCitizen 0 For No 1 For Yes',0,1)
    Partner=st.number_input('Insert a Partner 0 For No 1 For Yes',0,1)
    Dependents=st.number_input('Insert a Dependents 0 For No 1 For Yes',0,1)
    tenure=st.number_input('Insert a tenure',0)
    PhoneService=st.number_input('Insert a PhoneService 0 For No 1 For Yes',0,1)
    MultipleLines=st.number_input('Insert a MultipleLines 0 For No 1 For Yes 2 For No phone service',0,1)
    InternetService=st.number_input('Insert a InternetService 0 For No 1 For DSL 2 For Fiber Optics ',0,2)
    OnlineSecurity=st.number_input('Insert a OnlineSecurity 0 For No 1 For Yes 2 No Internet service ',0,2)
    OnlineBackup=st.number_input('Insert a OnlineBackup 0 For No 1 For Yes 2 No Internet service ',0,2)
    DeviceProtection=st.number_input('Insert a DeviceProtection 0 For No 1 For Yes 2 No Internet service ',0,2)
    TechSupport=st.number_input('Insert a TechSupport 0 For No 1 For Yes 2 No Internet service ',0,2)
    StreamingTV=st.number_input('Insert a StreamingTV 0 For No 1 For Yes 2 No Internet service ',0,2)
    StreamingMovies=st.number_input('Insert a StreamingMovies 0 For No 1 For Yes 2 No Internet service ',0,2)
    Contract=st.number_input('Insert a Contract 0 For Month-to-month 1 For one year 2 For Two year ',0,2)
    PaperlessBilling=st.number_input('Insert a PaperlessBilling 0 For No 1 For Yes',0,1)
    PaymentMethod=st.number_input('Insert a PaymentMethod 0 For Electronic check 1 For Mailed check 2 For Bank transfer 3 For Credit card ',0,3)
    MonthlyCharges=st.number_input('Insert a MonthlyCharges',0)
    TotalCharges=st.number_input('Insert a TotalCharges',0)
    
    
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(gender	,SeniorCitizen,	Partner,	Dependents,	tenure,	PhoneService,
                                MultipleLines,	InternetService,	OnlineSecurity,	OnlineBackup,	DeviceProtection,	TechSupport,
                                StreamingTV	,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges)
      st.success('Model has predicted that the{}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Rishav Bansal")
      st.subheader("Section-C,PIET")

if __name__=='__main__':
  main()
   
