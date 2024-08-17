from PIL import Image
import pandas as pd 
import numpy as np
import streamlit as st
import xgboost as xgb
import pickle

# Function to resize image
def resize_image(image, max_size=(600, 400)):
    original_size = image.size
    ratio = min(max_size[0] / original_size[0], max_size[1] / original_size[1])
    new_size = tuple(int(dim * ratio) for dim in original_size)
    return image.resize(new_size)

# Load and resize the image
image = Image.open('cross_section-ML.png')
resized_image = resize_image(image, max_size=(600, 400))

# Set page configuration
st.set_page_config(page_title="FRP contribution to Shear resistance", page_icon=":guardsman:", layout="wide")

# Function to load cached model
@st.cache_resource
def load_model_syn():
    model_syn = pickle.load(open('syn.pkl', "rb"))
    return model_syn

@st.cache_resource
def load_model_real():
    model_real = pickle.load(open('real.pkl', "rb"))
    return model_real

# Load models
tuned_model_syn = load_model_syn()
tuned_model_real = load_model_real()

# Mathematical functions
def cot(x): 
    return 1/np.tan(x)

def unscalery(value):
    return np.exp(((value-0.001)*3.380368123788582)-7.902757481871019)

# Function to calculate FRP contribution to shear resistance
def calculate(values):    
    st.write(values)
    Sample=pd.DataFrame(data={
        'Rho_f':[float(values.Rho_f)],
        'fcm':[float(values.fcm)],
        'E_f':[float(values.E_f)],
        'Rho_sw':[float(values.Rho_sw)],
        'Rho_sl':[float(values.Rho_sl)],
        'S_U_O':[int(values.S_U_O)],
        'hf':[float(values.hf)],
        'f_yy':[float(values.f_yy)],
        'alpha':[float(values.alpha)],
        'b_fl/bw':[float(values['b_fl_bw'])] 
    })
    prediction = tuned_model_.predict(Sample)
    e_fe = unscalery(prediction)
    result = e_fe * float(values.get('E_f')) * float(values.get('A_fpl')) * float(values.get('hf')) * (1 + cot(float(values.get('alpha')))) * np.sin(float(values.get('alpha')))
    return result[0]

# Initialize the DataFrame in session_state if not already present
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

# Center the image
st.markdown("<h1 style='text-align: center;'>Beams characteristic</h1>", unsafe_allow_html=True)
st.image(resized_image, caption='', use_column_width ='auto')
st.markdown("<h1 style='text-align: center;'>Enter
