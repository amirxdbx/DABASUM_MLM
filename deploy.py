### Libraries
import pandas as pd
import numpy as np
import streamlit as st
from pickle import load
################################################################

st.write("""
# XGBoost, RF, and GB Models to predict Shear strength of RC beam strengthened with EBR-FRP
This app predicts the **V_R** 
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Model=st.sidebar.radio('Select your model:',('XGBoost','RF','Gradiant Boosting'),key = "4")
    Cross_section = st.sidebar.radio(
     "What is the  type of cross section?",
     ('Rectangular', 'T_section'),key = "1")
    st.sidebar.title('Cross sections properties:')
    BW = st.sidebar.slider('BW', 75, 400, 200)
    HW = st.sidebar.slider('HW', 150, 650, 400)
    a_d = st.sidebar.slider('a_d', 1.2, 5.0, 1.56)
    RHOSWV = st.sidebar.number_input('RHOSWV', 0.0, 0.0122, 0.00164)
    RHOSL = st.sidebar.number_input('RHOSL', 0.003, 0.075, 0.023)
    FCM = st.sidebar.slider('FCM', 11.0, 60.0, 32.9)
    st.sidebar.title('FRP reinforcement properties:') 
    Rho_f = st.sidebar.number_input('Rho_f', 0.00014, 0.017, 0.00045)
    E_fm = st.sidebar.slider('E_fm', 67, 392, 235)
    w_s = st.sidebar.slider('w_s', 0.125, 1.0, 0.4)
    HF_C_EBR = st.sidebar.slider('HF_C_EBR', 150, 650, 366)
    alpha = st.sidebar.slider('alpha', 30, 90, 90)
    Full_wrap = st.sidebar.radio("Full_wrap or U-wrap?",
     ('Full_wrap', 'U-wrap'),key = "2")
    Continuous = st.sidebar.radio("Discrete or Continuous?",
     ('Discrete', 'Continuous'),key = "3")
    
   
    
    data = {'Type of Cross section': Cross_section,
            'Width of beam':BW,
            'Web height of beam': HW,
            'a/d':a_d,
            'transversal steel ratio': RHOSWV,
            'Longitudinal steel ratio':RHOSL,
            'Compressive strength of concrete':FCM,
            'FRP reinforcement':Rho_f,
            'Elastic modulus of FRP': E_fm,
            'width to spacing of FRP strips (1 in case of continuous)':w_s,
            'Height of FRP (equal to HW in case of Full wrap)':HF_C_EBR,
            'Angel between FRP fibers and beam axis':alpha,
            'Type of Wrapping':Full_wrap,
            'continuity':Continuous
            }
    if Cross_section=='Rectangular':
        cross=1
    else:
        cross=0
    if Full_wrap=='Full_wrap':
        Full_wrap=1
    else:
        Full_wrap=0
    if Continuous=='Continuous':
        Continuous=1
    else:
        Continuous=0
    X = {'Rectangular':cross  ,
            'BW':BW,
            'HW': HW,
            'a/d':a_d,
            'RHOSWV': RHOSWV,
            'RHOSL':RHOSL,
            'FCM':FCM,
            'Rho_f':Rho_f,
            'E_fm': E_fm*1000,
            'w/s':w_s,
            'HF_C_EBR':HF_C_EBR,
            'alpha':np.radians(alpha),
            'Full wrap':Full_wrap,
            'Continuous':Continuous
            }
    X=pd.Series(X)    
    X = pd.DataFrame(np.array(X).reshape(1,-1),columns=X.index)
    return data,Model,X

data,Model,X = user_input_features()

st.subheader('User Input parameters')
st.write(data)

#load ML model 
if Model=='XGBoost': 
    st.write('XGBoost model is loaded!')
    Selected_Model=(r'XGboost.pkl')
elif Model=='RF':
    st.write('RF model is loaded!')
    Selected_Model=(r'Random Forest.pkl')
elif Model=='Gradiant Boosting':
    st.write('Gradiant Boosting model is loaded!')
    Selected_Model=(r'Gradient Bossting.pkl')

model=open(Selected_Model,"rb")
st.subheader('Prediction of Shear strength:')
scaled_outputs=model.predict(X)
st.write('$V_R= $', round(scaled_outputs[0],1),'kN')