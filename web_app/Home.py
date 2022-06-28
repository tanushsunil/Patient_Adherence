import streamlit as st
from PIL import Image


st.header("What are we solving?")
st.text("The problem addressed by this model is finding out if the patient is medically\nadherent or not. clinical studies have demonstrated that only 50-70% of patients\nadhere properly to prescribed drug therapy. This behavior of adherence failure\ncan cause health issues, hospitalization risk and even death. Patient Adherence\ninsights can prove useful for \n\n     1.Doctors who prescribe drugs\n     2.Drug / Medication producers\n     3.Government")
image = Image.open("web_app\Images\Home_embed_image.png")
st.image(image)
st.title("What is our model focused on?")
st.text("This model is focused on the prediction of adherence behavior with individual\nselection. The dataset utilized is historically captured through a medication\nevent monitoring system. When the group who are prone to be non-adherent is\naccurately identified and targeted, it makes the way for improving patient care\nand helps Healthcare workers to assess and develop new strategies.")