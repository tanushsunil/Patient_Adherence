import pandas as pd
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import web_app.sample_app as sample_app
from PIL import Image
import web_app.Model_tester as Model_tester

st.title("Patient Adherence")

Selected = option_menu(
    menu_title=None,
    options=["Home", "Models Tester", "PA Finder", "About us"],
    icons=["house-fill", "speedometer2", "hexagon-half", "people-fill"],
    default_index=0,
    orientation="horizontal"
)

if Selected == "PA Finder":
    global model_version
    model_version = option_menu(
        menu_title=None,
        options=["Individual Patient", "Predict for group"],
        icons=["hexagon", "hexagon-fill"],
        default_index=0,
        orientation="horizontal")
    if model_version == "Individual Patient":
        sample_app.main()
    else:
        st.title('Coming soon! :)')

elif Selected == "Home":
    st.header("What are we solving?")
    st.text("The problem addressed by this model is finding out if the patient is medically\nadherent or not. clinical studies have demonstrated that only 50-70% of patients\nadhere properly to prescribed drug therapy. This behavior of adherence failure\ncan cause health issues, hospitalization risk and even death. Patient Adherence\ninsights can prove useful for \n\n     1.Doctors who prescribe drugs\n     2.Drug / Medication producers\n     3.Government")
    image = Image.open("web_app\Images\Home_embed_image.png")
    st.image(image)
    st.title("What is our model focused on?")
    st.text("This model is focused on the prediction of adherence behavior with individual\nselection. The dataset utilized is historically captured through a medication\nevent monitoring system. When the group who are prone to be non-adherent is\naccurately identified and targeted, it makes the way for improving patient care\nand helps Healthcare workers to assess and develop new strategies.")

elif Selected == "Models Tester":
    Model_tester.main()



        
