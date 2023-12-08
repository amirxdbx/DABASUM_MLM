from pycaret.regression import *
import pandas as pd 
import numpy as np
import streamlit as st


st.set_page_config(page_title="FRP contribution to Shear resistance", page_icon=":guardsman:", layout="wide")

# @st.cache_resource
# def load_model_():
#     model = load_model('$gb$.pkl')
#     return model

# tuned_model_ = load_model_()

# def cot(x): 
#     return 1/np.tan(x)

# def unscalery(value):
#     return np.exp(((value-0.1)*3.667910806940382)-7.900217131033494)

# def calculate(values):    
#     st.write(values)
#     A_fpl = float(values.Af)/float(values.sf)
#     A_spl= float(values.As)/float(values.ss)
#     b_fl_b_w= float(values.b_fl)/float(values.b_w)
    
#     if values.Config [0]== 'Fully-wrapped':
#         S_U_O = 0
#     elif values.Config [0]== 'U-wrapped':
#         S_U_O = 1
#     else: 
#         S_U_O = 2

    # dataSample=pd.DataFrame(data={'A_fpl':[A_fpl],'A_spl':[A_spl],'alpha':[float(values.alpha)],
    #                               'w_s':[float(values.wf_sf)],'hf':[float(values.hf)],'S_U_O':[S_U_O],
    #                               'E_f':[float(values.E_f)]})
    
    # e_fe = unscalery((predict_model(tuned_model_,dataSample).prediction_label))[0]
    # result = e_fe*float(values.get('Ef'))*float(values.get('A_fpl'))*np.sin(np.radians(dataSample.alpha))*dataSample.hf*(1+cot(np.radians(dataSample.alpha)))*np.sin(np.radians(dataSample.alpha))
#     return result

st.write('Enter your beam data:')
col1, col2, col3 = st.columns([2,2,3])

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
#     out = st.empty()
#     if st.session_state.get('Calculate'):
#         result = np.round(calculate(values), 2)
#         out.text(f"Contribution of FRP to shear resistance: \n {result[0]} kN")
