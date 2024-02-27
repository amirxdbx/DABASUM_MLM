from PIL import Image
from pycaret.regression import *
import pandas as pd 
import numpy as np
import streamlit as st
import pickle

def resize_image(image, max_size=(600, 400)):
    original_size = image.size
    ratio = min(max_size[0] / original_size[0], max_size[1] / original_size[1])
    new_size = tuple(int(dim * ratio) for dim in original_size)
    return image.resize(new_size)
    
# Load and resize the image
image = Image.open('cross_section-ML.png')
resized_image = resize_image(image, max_size=(600, 400))

st.set_page_config(page_title="FRP contribution to Shear resistance", page_icon=":guardsman:", layout="wide")

@st.cache_resource
def load_model_():
    model = load_model('synth_xgboost')
    return model

tuned_model_ = load_model_()

def cot(x): 
    return 1/np.tan(x)

def unscalery(value):
    return np.exp(((value-0.001)*3.3803681237888172)-7.902757481871264)

def calculate(values):    
    st.write(values)

    Sample=pd.DataFrame(data={'E_f':[float(values.E_f)],
                              'Rho_f':[float(values.Rho_f)],
                              'fcm':[float(values.fcm)],
                              'Rho_sw':[float(values.Rho_sw)],
                              'Rho_sl':[float(values.Rho_sl)],
                              'hf':[float(values.hf)],
                              'b_fl/bw':[float(values['b_fl_bw'])],
                              'S_U_O':[float(values.S_U_O)],
                              'alpha':[float(values.alpha)],
                              'f_yy':[float(values.f_yy)]})
    
    e_fe = unscalery(predict_model(tuned_model_,Sample).prediction_label[0])
    result = e_fe*float(values.get('E_f'))*float(values.get('A_fpl'))* float(values.get('hf'))*(1+cot(float(values.get('alpha'))))*np.sin(float(values.get('alpha')))
    return result
    
# Center the image
st.markdown("<h1 style='text-align: center;'>Beams characteristic</h1>", unsafe_allow_html=True)
st.image(resized_image, caption='', use_column_width ='auto')
st.markdown("<h1 style='text-align: center;'>Enter your beam data:</h1>", unsafe_allow_html=True)

col1, col2, col3, col4,col5= st.columns([2,2,2,2,4])

with col1:
    tf= st.number_input("Thickness of FRP (mm):", value=0.352)
    sf= st.number_input("sf (mm):", value=114)
    wf= st.number_input("wf (mm):", value=60)
    A_fpl=2*tf*wf/sf
    hf= st.number_input("Height of FRP reinforcement (mm):", value=300)
with col2: 
    E_f= st.number_input("Elasticity modulus of FRP (GPa):", value=218.4)    
    alpha_options = [45, 90]
    alpha = st.selectbox("FRP orientation:", options=alpha_options, index=alpha_options.index(90))
    config_options = ['Fully wrapped', 'U-wrapped', 'Side-bonded']
    S_U_O = st.selectbox("FRP configuration:", options=config_options, index=config_options.index('Fully wrapped'))

with col3:
    Asw = st.number_input("Area of stirrups (mm2):", value=56.5)
    ss = st.number_input("Spacing of stirrups (mm):", value=300)
    f_yy = st.number_input("Steel yield strength (MPa):", value=400)
    if ss==0:
        Rho_sw = 0
    else: 
        A_spl =Rho_sw / ss
    
with col4:
    b_fl= st.number_input("Width of beam flange (mm):", value=450)
    b_w= st.number_input("Width of beam web(mm):", value=180)
    fcm= st.number_input("Concrete compressive strength (MPa):", value=30)
    
    b_fl_bw=b_fl/b_w
    Rho_sl= st.number_input("Ratio of longitudinal steel(mm):", value=0.05)
    
values=pd.DataFrame({
    'A_fpl':[A_fpl],
    'E_f': [E_f],
    'Rho_sw': [Rho_sw],
    'Rho_sl': [Rho_sw],
    'alpha': [np.radians(alpha)],
    'hf': [hf],
    'b_fl_bw': [b_fl_bw],
    'S_U_O':[np.where(S_U_O=='Fully wrapped',0,np.where(S_U_O=='U-wrapped',1,2))],
    'Rho_f':[A_fpl/b_w],
    'fcm':[fcm],
    'f_yy': [f_yy]
})   

with col5:
    st.button('Calculate', key='Calculate')
    out = st.empty()
    if st.session_state.get('Calculate'):
        result = np.round(calculate(values), 2)
        out.text(f"Contribution of FRP to shear resistance: \n {result} kN")
