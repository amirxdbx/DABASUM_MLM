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
    model = load_model('gbr')
    return model

tuned_model_ = load_model_()

def cot(x): 
    return 1/np.tan(x)

def unscalery(value):
    return np.exp(((value-0.1)*3.667910806940572)-7.900217131033686)

def calculate(values):    
    st.write(values)

    Sample=pd.DataFrame(data={'A_fpl':[A_fpl],'w_s':[float(values.wf_sf)],'b_fl/bw':[float(values['b_fl_bw'])],
                                  'a_d':[float(values.a_d)],
                                  'A_spl':[A_spl],
                                  'E_f':[float(values.E_f)]})
    
    e_fe = unscalery(predict_model(tuned_model_,Sample).prediction_label[0])
    result = e_fe*float(values.get('E_f'))*float(values.get('A_fpl'))* float(values.get('hf'))*(1+cot(float(values.get('alpha'))))*np.sin(float(values.get('alpha')))
    return result
    
# Center the image
st.markdown("<h1 style='text-align: center;'>Beams characteristic</h1>", unsafe_allow_html=True)
st.image(resized_image, caption='', use_column_width ='auto')
st.markdown("<h1 style='text-align: center;'>Enter your beam data:</h1>", unsafe_allow_html=True)

col1, col2, col3, col4,col5= st.columns([2,2,2,2,4])

with col1:
    tf= st.number_input("Thickness of FRP (mm):", value=0.168)
    sf= st.number_input("sf (mm):", value=150)
    wf= st.number_input("wf (mm):", value=150)
    A_fpl=2*tf*wf/sf
    wf_sf= wf/sf
    hf= st.number_input("Height of FRP reinforcement (mm):", value=250)
with col2: 
    E_f= st.number_input("Elasticity modulus of FRP (GPa):", value=230)    
    alpha_options = [45, 90]
    alpha = st.selectbox("FRP orientation:", options=alpha_options, index=alpha_options.index(90))

with col3:
    Asw= st.number_input("Area of stirrups (mm2):", value=0)
    ss= st.number_input("spacing of stirrups (mm):", value=0)
    if ss==0:
        A_spl =0
    else: 
        A_spl =Asw / ss
    
with col4:
    b_fl= st.number_input("Width of beam flange (mm):", value=150)
    b_w= st.number_input("Width of beam web(mm):", value=150)
    a_d= st.number_input("Shear span to depth ratio:", value=2.27)
    b_fl_bw=b_fl/b_w
    
values=pd.DataFrame({
    'A_fpl':[A_fpl],
    'E_f': [E_f],
    'A_spl': [A_spl],
    'alpha': [np.radians(alpha)],
    'wf_sf': [wf_sf],
    'hf': [hf],
    'b_fl_bw': [b_fl_bw],
    'a_d':a_d
})   
with col5:
    st.button('Calculate', key='Calculate')
    out = st.empty()
    if st.session_state.get('Calculate'):
        result = np.round(calculate(values), 2)
        out.text(f"Contribution of FRP to shear resistance: \n {result} kN")


