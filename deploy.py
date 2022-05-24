### Libraries
import pandas as pd
import numpy as np
import streamlit as st
import pickle
@st.cache(hash_funcs={"MyUnhashableClass": lambda _: None})
          
def load_model():
    XGB=pickle.load(open('XGboost.pkl','rb'))
    RF=pickle.load(open('Random Forest.pkl','rb'))
    GBoost=pickle.load(open('Gradient Bossting.pkl','rb'))
    return XGB,RF,GBoost

XGB,RF,GBoost=load_model()
################################################################
