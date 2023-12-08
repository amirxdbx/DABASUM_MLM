from pycaret.regression import *
import pandas as pd 
import numpy as np
import streamlit as st
import pickle


st.set_page_config(page_title="FRP contribution to Shear resistance", page_icon=":guardsman:", layout="wide")

@st.cache_resource
def load_model_():
    model = load_model('$gb$')
    return model

tuned_model_ = load_model_()

def cot(x): 
    return 1/np.tan(np.radians(x))

def unscalery(value):
    return np.exp(((value-0.1)*3.667910806940382)-7.900217131033494)

def calculate(values):    
    st.write(values)
    A_fpl = float(values.A_fpl)
    A_spl= float(values.A_spl)
    b_fl_b_w= float(values.b_fl_b_w)
    

    Sample=pd.DataFrame(data={'A_fpl':[A_fpl],'w_s':[float(values.wf_sf)],'':[float(values.b_fl_b_w)],
                                  'a_d':[float(values.a_d)],
                                  'A_spl':[A_spl],'alpha':[float(values.alpha)],
                                  ,'E_f':[float(values.E_f)]})
    
    e_fe = unscalery((predict_model(tuned_model_,Sample).prediction_label))[0]
    result = e_fe*float(values.get('Ef'))*float(values.get('A_fpl'))*
            float(values.get('hf'))*(cot(45)+cot(float(values.get('alpha'))))*np.sin(np.radians(float(values.get('alpha'))))
    return result

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
    a_d= st.number_input("Shear span to depth ratio:", value=3)
    b_fl_b_w=b_fl/b_w
    
values=pd.DataFrame({
    'A_fpl':[A_fpl],
    'Ef': [Ef],
    'A_spl': [A_spl],
    'alpha': [alpha],
    'wf_sf': [wf_sf],
    'hf': [hf],
    'b_fl_b_w': [b_fl_b_w],
    'a_d':a_d
})

with col3:
    st.empty()
    st.empty()
    st.empty()
    st.button('Calculate', key='Calculate')
    out = st.empty()
    if st.session_state.get('Calculate'):
        result = np.round(calculate(values), 2)
        out.text(f"Contribution of FRP to shear resistance: \n {result[0]} kN")
