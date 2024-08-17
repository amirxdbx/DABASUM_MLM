from PIL import Image
import pandas as pd 
import numpy as np
import streamlit as st
import xgboost as xgb
import pickle
import io

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
    return 1 / np.tan(x)

def unscalery(value):
    return np.exp(((value - 0.001) * 3.380368123788582) - 7.902757481871019)

# Function to calculate FRP contribution to shear resistance
def calculate(values):    
    st.write(values)
    Sample = pd.DataFrame(data={
        'Rho_f': [float(values.Rho_f)],
        'fcm': [float(values.fcm)],
        'E_f': [float(values.E_f)],
        'Rho_sw': [float(values.Rho_sw)],
        'Rho_sl': [float(values.Rho_sl)],
        'S_U_O': [int(values.S_U_O)],
        'hf': [float(values.hf)],
        'f_yy': [float(values.f_yy)],
        'alpha': [float(values.alpha)],
        'b_fl/bw': [float(values['b_fl_bw'])] 
    })
    prediction = tuned_model_.predict(Sample)
    e_fe = unscalery(prediction)
    result = e_fe * float(values.get('E_f')) * float(values.get('A_fpl')) * float(values.get('hf')) * (1 + cot(float(values.get('alpha')))) * np.sin(float(values.get('alpha')))
    return result[0]

def ACI(values):
    row=values.copy(deep=True)
    row['full']=row['S_U_O']
    ## ACI 440.2R-17
    alpha = row.alpha
    Alpha_factor = np.sin(alpha)+np.cos(alpha)
    C_E = 0.95   
    eps_fu = C_E*row['eps_fu']
    fck=row['fcm']-8
    ###   e_fe_predicted  #########################################
    L_e = 23300/(row['tf']*row['E_f']*1000)**0.58
    k_1 = (fck/27)**(2/3)
    k_2 = np.where(row['full']==1,(row['d_fv']-L_e)/row['d_fv'],
                    np.where(row.full==2,np.maximum(0,(row['d_fv']-2*L_e)/row['d_fv']),0))
    row['k_v'] = np.where(row['full'] == 0, 0.75,(k_1*k_2*L_e)/(11900*eps_fu))
    # e_fe_predicted
    e_fe_ACI = row['k_v']*eps_fu
    e_fe_ACI = np.where(e_fe_ACI > 0.004, 0.004,e_fe_ACI)
    # f_fe
    f_fe = row['E_f']*1000*e_fe_ACI
    # V_f_predicted
    Sai_f = np.where(row['full'] == 0, 0.95, 0.85)
    A_fv = 2*row['tf']*np.where(row['wf'] == 1, np.sin(alpha), row['wf']/row['sf'])
    row['V_f_model'] = Sai_f*(f_fe*Alpha_factor*row['d_fv']*A_fv)*0.001        
    return  row['V_f_model']

# Initialize the DataFrame in session_state if not already present
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

# Center the image
st.markdown("<h1 style='text-align: center;'>Beams characteristic</h1>", unsafe_allow_html=True)
st.image(resized_image, caption='', use_column_width='auto')
st.markdown("<h1 style='text-align: center;'>Enter your beam data:</h1>", unsafe_allow_html=True)

# UI layout with column configuration
col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 4])

# Input fields
with col1:
    Model_options = ['Xgboost_real', 'XGBoost_syn']
    model = st.selectbox("Model:", options=Model_options, index=Model_options.index('XGBoost_syn'))
    tuned_model_ = tuned_model_syn if model == 'XGBoost_syn' else tuned_model_real
    tf = st.number_input("Thickness of FRP (mm):", value=0.352)
    sf = st.number_input("sf (mm):", value=114)
    wf = st.number_input("wf (mm):", value=60)
    A_fpl = 2 * tf * wf / sf
    hf = st.number_input("Height of FRP reinforcement hf (mm):", value=300)
    d_fv= st.number_input("Effective height of FRP (mm):", value=260)

with col2: 
    eps_fu = st.number_input("Ultimate strength of FRP Ef (MPa):", value=2862.9)    
    E_f = st.number_input("Elasticity modulus of FRP Ef (GPa):", value=218.4)    
    alpha_options = [45, 90]
    alpha = st.selectbox("Fibres orientation:", options=alpha_options, index=alpha_options.index(90))
    config_options = ['Fully wrapped', 'U-wrapped', 'Side-bonded']
    S_U_O = st.selectbox("FRP configuration:", options=config_options, index=config_options.index('U-wrapped'))

with col3:
    h= st.number_input("Total height of beam (mm):", value=400)
    b_fl = st.number_input("Width of beam flange (mm):", value=450)
    b_w = st.number_input("Width of beam web(mm):", value=180)
    fcm = st.number_input("Concrete compressive strength (MPa):", value=39.7)
    b_fl_bw = b_fl / b_w
    Rho_sl = st.number_input("Ratio of longitudinal steel(mm):", value=0.038397)

with col4:
    Asw = st.number_input("Area of stirrups (mm2):", value=56.5)
    ss = st.number_input("Spacing of stirrups (mm):", value=300)
    f_yy = st.number_input("Steel yield strength fswy (MPa):", value=542)
    Rho_sw = 0 if ss == 0 else Asw / ss / b_w

# Prepare values for calculation
values = pd.DataFrame({
    'A_fpl': [A_fpl],
    'E_f': [E_f],
    'Rho_sw': [Rho_sw],
    'Rho_sl': [Rho_sl],
    'alpha': [np.radians(int(alpha))],
    'tf':[tf],
    'sf':[sf],
    'wf':[wf],
    'hf': [hf],
    'h':[h],
    'd_fv':[d_fv],
    'eps_fu':[eps_fu],
    'b_fl_bw': [b_fl_bw],
    'S_U_O': [np.where(S_U_O == 'Fully wrapped', 0, np.where(S_U_O == 'U-wrapped', 1, 2))],
    'Rho_f': [A_fpl / b_w],
    'fcm': [fcm],
    'f_yy': [f_yy]
})  

# Calculate and save results
with col5:
    if st.button('Calculate'):
        result = calculate(values)
        values['result'] = result
        values['model'] = model
        st.session_state.df = pd.concat([st.session_state.df, values], ignore_index=True)
        st.write(f"Contribution of FRP to shear resistance (Based on XGBoost model): {result:.2f} kN")

        ACI_result= ACI(values)
        values['ACI_result'] = ACI_result
        st.write(f"Contribution of FRP to shear resistance (based on ACI 440.2R): {ACI_result} kN")
    
    if st.button('Clear Logs'):
        st.session_state.df = pd.DataFrame()  # Clear the DataFrame
        st.write("Logs cleared.")
    
    if st.button('Download Log'):
        # Convert DataFrame to CSV
        csv = st.session_state.df.to_csv(index=False)
        # Create a download link for the CSV
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='frp_contribution_logs.csv',
            mime='text/csv'
        )

# Display the DataFrame with saved results
st.write("Log of results:")
st.write(st.session_state.df)
