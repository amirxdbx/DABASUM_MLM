from pycaret.regression import *
import pandas as pd 
import numpy as np
import streamlit as st
import pickle


st.set_page_config(page_title="FRP contribution to Shear resistance", page_icon=":guardsman:", layout="wide")

@st.cache_resource
def load_model_():
    model = load_model('$gb$.pkl')
    return model

tuned_model_ = load_model_()

# def cot(x): 
#     return 1/np.tan(x)

# def unscalery(value):
#     return np.exp(((value-0.1)*3.667910806940382)-7.900217131033494)

# def calculate(values):    
#     st.write(values)
#     A_fpl = float(values.Af)/float(values.sf)
#     A_spl= float(values.As)/float(values.ss)
#     b_fl_b_w= float(values.b_fl)/float(values.b_w)
    

    # dataSample=pd.DataFrame(data={'A_fpl':[A_fpl],'A_spl':[A_spl],'alpha':[float(values.alpha)],
    #                               'w_s':[float(values.wf_sf)],'hf':[float(values.hf)],'S_U_O':[S_U_O],
    #                               'E_f':[float(values.E_f)]})
    
    # e_fe = unscalery((predict_model(tuned_model_,dataSample).prediction_label))[0]
    # result = e_fe*float(values.get('Ef'))*float(values.get('A_fpl'))*np.sin(np.radians(dataSample.alpha))*dataSample.hf*(1+cot(np.radians(dataSample.alpha)))*np.sin(np.radians(dataSample.alpha))
#     return result

st.write('Enter your beam data:')
col1, col2, col3 = st.columns([2,2,3])

with col1:
    tf= st.number_input("Thickness of FRP (mm):", value=0.1)
    sf= st.number_input("sf (mm):", value=114)
    wf= st.number_input("wf (mm):", value=114)
    A_fpl=2*tf*wf/sf
    Ef= st.number_input("Elasticity modulus of FRP (GPa):", value=218.4)
    Asw= st.number_input("Area of stirrups (mm2):", value=56.5)
    ss= st.number_input("spacing of stirrups (mm):", value=300)
    A_spl=Asw/ss
with col2:
    alpha_options = [45, 90]
    alpha = st.selectbox("FRP orientation:", options=alpha_options, index=alpha_options.index(90))
    wf_sf= wf/sf
    hf= st.number_input("Height of FRP reinforcement (mm):", value=300)
    b_fl= st.number_input("Width of beam flange (mm):", value=300)
    b_w= st.number_input("Width of beam web(mm):", value=300)
    b_fl_b_w=b_fl/b_w
    
values=pd.DataFrame({
    'A_fpl':[A_fpl],
    'Ef': [Ef],
    'A_spl': [A_spl],
    'alpha': [alpha],
    'wf_sf': [wf_sf],
    'hf': [hf],
    'b_fl_b_w': [b_fl_b_w]
    
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
