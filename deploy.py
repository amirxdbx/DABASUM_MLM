from pycaret.regression import *
import pandas as pd 
import numpy as np
import streamlit as st


st.set_page_config(page_title="FRP contribution to Shear resistance", page_icon=":guardsman:", layout="wide")

@st.cache_data  
def load_data(url):
    Main_data = pd.read_csv(url)
    return Main_data

Main_data = load_data('Main_data.csv')

@st.cache_resource
def load_model_():
    model = load_model('$Xgboost$.pkl')
    return model

tuned_model_ = load_model_()

def cot(x): 
    return 1/np.tan(x)

def unscalery(value,Label,NewDataset):
    return np.exp(((value-0.1)*(np.log(NewDataset[Label]).max()-np.log(NewDataset[Label]).min()))+np.log(NewDataset[Label]).min())

def calculate(values):    
    st.write(values)
    SFI = float(values.Af)/float(values.sf)+(float(values.Es)/float(values.Ef))*float(values.As)/float(values.ss)
    if values.Config [0]== 'Fully-wrapped':
        S_U_O = 0
    elif values.Config [0]== 'U-wrapped':
        S_U_O = 1
    else: 
        S_U_O = 2

    dataSample=pd.DataFrame(data={'SFI_factor':[SFI],'alpha':[float(values.alpha)],
                                  'w_s':[float(values.wf_sf)],'hf':[float(values.hf)],'S_U_O':[S_U_O],
                                  'fcm':[float(values.fcm)]})
    
    e_fe = unscalery((predict_model(tuned_model_,dataSample).prediction_label),'e_fe',Main_data)[0]
    result = e_fe*float(values.get('Ef'))*float(values.get('Af'))/float(values.get('sf'))*np.sin(np.radians(dataSample.alpha))*dataSample.hf*(1+cot(np.radians(dataSample.alpha)))*np.sin(np.radians(dataSample.alpha))
    return result

st.write('Enter your beam data:')
col1, col2, col3 = st.columns([2,2, 3])

with col1:
    Af= st.number_input("Area of FRP (mm2):", value=42)
    sf= st.number_input("sf (mm):", value=114)
    Es= st.number_input("Elasticity modulus of Steel (GPa):", value=200)
    Ef= st.number_input("Elasticity modulus of FRP (GPa):", value=218.4)
    As= st.number_input("Area of stirrups (mm2):", value=56.5)
    ss= st.number_input("ss (mm):", value=300)
with col2:
    
    alpha= st.number_input("FRP orientation:", value=90)
    wf_sf= st.number_input("width to spacing ratio:", value=0.53)
    hf= st.number_input("Height of FRP reinforcement (mm):", value=300)
    fcm= st.number_input("Concrete compressive strength (MPa)", value=39.7)
    Config= st.radio('',options=("Fully-wrapped","U-wrapped","Side-bonded"),index=1) 

    
values=pd.DataFrame({
    'Af': [Af],
    'sf': [sf],
    'Es': [Es],
    'Ef': [Ef],
    'As': [As],
    'ss': [ss],
    'alpha': [alpha],
    'wf_sf': [wf_sf],
    'hf': [hf],
    'fcm': [fcm],
    'Config': [Config],
})

with col3:
    st.empty()
    st.empty()
    st.empty()
    st.button('Calculate', key='Calculate')
    out = st.empty()
    if st.session_state.get('Calculate'):
        result = np.round(calculate(values), 2)
        out.text(f"Contribution of FRP to shear resistance: {result[0]} kN")
