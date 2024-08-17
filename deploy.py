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
@st.cache(allow_output_mutation=True)
def load_model_syn():
    model_syn = pickle.load(open('syn.pkl', "rb"))
    return model_syn

@st.cache(allow_output_mutation=True)
def load_model_real():
    model_real = pickle.load(open('real.pkl', "rb"))
    return model_real

# Load models
tuned_model_syn = load_model_syn()
tuned_model_real = load_model_real()

# Mathematical functions
def cot(x): 
    return 1 / np.tan(x)

def unscalery(value):
    return np.exp(((value - 0.001) * 3.380368123788582) - 7.902757481871019)

# Function to calculate FRP contribution to shear resistance
def calculate(values):
    Sample = pd.DataFrame(data={
        'fcm': [float(values['fcm'])],
        'E_f': [float(values['E_f'])],
        'Rho_sw': [float(values['Rho_sw'])],
        'Rho_sl': [float(values['Rho_sl'])],
        'S_U_O': [(values['S_U_O'])],
        'hf': [float(values['hf'])],
        'f_yy': [float(values['f_yy'])],
        'alpha': [float(values['alpha'])],
        'b_fl/bw': [float(values['b_fl_bw'])]
    })

    prediction = tuned_model_.predict(Sample)
    e_fe = unscalery(prediction)
    result = e_fe * float(values.get('E_f')) * float(values.get('A_fpl')) * float(values.get('hf')) * (1 + cot(float(values.get('alpha')))) * np.sin(float(values.get('alpha')))
    return result[0]

# UI layout
st.markdown("<h1 style='text-align: center;'>Beams characteristic</h1>", unsafe_allow_html=True)
st.image(resized_image, caption='', use_column_width='auto')
st.markdown("<h1 style='text-align: center;'>Enter your beam data:</h1>", unsafe_allow_html=True)

# Column layout
col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 4])

# Load last stored values or initialize defaults
session_values = st.session_state.get('user_values', {
    'tf': 0.352,
    'sf': 114,
    'wf': 60,
    'hf': 300,
    'E_f': 218.4,
    'alpha': 90,
    'S_U_O': 'U-wrapped',
    'b_fl': 450,
    'b_w': 180,
    'fcm': 39.7,
    'Asw': 56.5,
    'ss': 300,
    'f_yy': 542
})

# User inputs
with col1:
    Model_options = ['Xgboost_real', 'XGBoost_syn']
    model = st.selectbox("Model:", options=Model_options, index=Model_options.index('XGBoost_syn'))
    if model == 'XGBoost_syn':
        tuned_model_ = tuned_model_syn
    else: 
        tuned_model_ = tuned_model_real
    tf = st.number_input("Thickness of FRP (mm):", value=session_values['tf'])
    sf = st.number_input("sf (mm):", value=session_values['sf'])
    wf = st.number_input("wf (mm):", value=session_values['wf'])
    A_fpl = 2 * tf * wf / sf
    hf = st.number_input("Height of FRP reinforcement hf (mm):", value=session_values['hf'])

with col2: 
    E_f = st.number_input("Elasticity modulus of FRP Ef (GPa):", value=session_values['E_f'])    
    alpha_options = [45, 90]
    alpha = st.selectbox("Fibres orientation:", options=alpha_options, index=alpha_options.index(session_values['alpha']))
    config_options = ['Fully wrapped', 'U-wrapped', 'Side-bonded']
    S_U_O = st.selectbox("FRP configuration:", options=config_options, index=config_options.index(session_values['S_U_O']))

with col3:
    b_fl = st.number_input("Width of beam flange (mm):", value=session_values['b_fl'])
    b_w = st.number_input("Width of beam web(mm):", value=session_values['b_w'])
    fcm = st.number_input("Concrete compressive strength (MPa):", value=session_values['fcm'])
    b_fl_bw = b_fl / b_w
    Rho_sl = st.number_input("Ratio of longitudinal steel(mm):", value=0.038397)

with col4:
    Asw = st.number_input("Area of stirrups (mm2):", value=session_values['Asw'])
    ss = st.number_input("Spacing of stirrups (mm):", value=session_values['ss'])
    f_yy = st.number_input("Steel yield strength fswy (MPa):", value=session_values['f_yy'])
    if ss == 0:
        Rho_sw = 0
    else: 
        Rho_sw = Asw / ss / b_w

# Store current values in session_state
st.session_state.user_values = {
    'tf': tf,
    'sf': sf,
    'wf': wf,
    'hf': hf,
    'E_f': E_f,
    'alpha': alpha,
    'S_U_O':([np.where(S_U_O=='Fully wrapped',0,np.where(S_U_O=='U-wrapped',1,2))])[0],    
    'b_fl': b_fl,
    'b_w': b_w,
    'fcm': fcm,
    'Asw': Asw,
    'ss': ss,
    'f_yy': f_yy,
    'Rho_sw':Rho_sw,
    'Rho_sl': Rho_sl
}

# Calculate button and output
with col5:
    st.button('Calculate', key='Calculate')
    out = st.empty()
    if st.session_state.get('Calculate'):
        result = calculate(st.session_state.user_values)
        out.text(f"Contribution of FRP to shear resistance: \n {result:.2f} kN")
