import pandas as pd
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

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
        model = pickle.load(open('Logistic_Regression.sav', 'rb'))

# creating a function for Prediction
        def adherence_prediction(input_data):

    # changing the input_data to numpy array
           input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
           input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

           prediction = model.predict(input_data_reshaped)
           print(prediction)

           if (prediction[0] == 0):
              return 'The patient is most likely to not adhere to the medication'
           else:
              return 'The patient will likely be adhered to the medication'


        def main():

           # getting the input data from the user
           Age = st.text_input('Age')
           Prescription_Days = st.text_input('Prescription Days')
           Gender = st.selectbox('Select you Gender', ('Male', 'Female'))
           Insurance_1 = st.selectbox('Do you have Insurance?', ('Yes', 'No'))
           Marital_status_1 = st.selectbox('Is the patient married?', ('Married', 'Unmarried'))
           Diabetes_1 = st.selectbox('Does the patient have Diabetes?', ('Yes', 'No'))
           Alcoholism_1 = st.selectbox('Does the patient have Alcoholism?', ('Yes', 'No'))
           HyperTension_1 = st.selectbox('Does the patient have HyperTension?', ('Yes', 'No'))
           Smokes_1 = st.selectbox('Does the patient smoke?', ('Yes', 'No'))
           Tuberculosis_1 = st.selectbox('Does the patient have Tuberculosis?', ('Yes', 'No'))
           Drug_cost_1 = st.selectbox('Cost of the drug', ('High', 'Low'))

           if Gender == 'Male':
              Male, Female = 1, 0
           else:
              Male, Female = 0, 1

           if Insurance_1 == 'Yes':
              Insurance = 1
           else:
              Insurance = 0

           if Alcoholism_1 == 'Yes':
              Alcoholism = 1
           else:
              Alcoholism = 0

           if Tuberculosis_1 == 'Yes':
              Tuberculosis = 1
           else:
              Tuberculosis = 0

           if Diabetes_1 == 'Yes':
              Diabetes = 1
           else:
              Diabetes = 0

           if HyperTension_1 == 'Yes':
              HyperTension = 1
           else:
              HyperTension = 0

           if Smokes_1 == 'Yes':
              Smokes = 1
           else:
              Smokes = 0

           if Marital_status_1 == 'Married':
              Married, Unmarried = 1, 0
           else:
              Married, Unmarried = 0, 1

           if Drug_cost_1 == 'High':
              High, Low = 1, 0
           else:
              High, Low = 0, 1

           # code for Prediction
           adherence_diagnosis = ''

        # creating a button for Prediction

           if st.button('Get Adherence Test Result'):
              adherence_diagnosis = adherence_prediction([Age, Prescription_Days, Male, Female, Married, Unmarried, HyperTension, Diabetes, Alcoholism, Smokes, High, Low, Insurance, Tuberculosis])
        

           st.success(adherence_diagnosis)


           if __name__ == '__main__':
              main()
    else:
        st.title('Coming soon! :)')

elif Selected == "Home":
    st.header("What are we solving?")
    st.text("The problem addressed by this model is finding out if the patient is medically\nadherent or not. clinical studies have demonstrated that only 50-70% of patients\nadhere properly to prescribed drug therapy. This behavior of adherence failure\ncan cause health issues, hospitalization risk and even death. Patient Adherence\ninsights can prove useful for \n\n     1.Doctors who prescribe drugs\n     2.Drug / Medication producers\n     3.Government")
    st.image("https://imgur.com/a/9adYla7")
    st.title("What is our model focused on?")
    st.text("This model is focused on the prediction of adherence behavior with individual\nselection. The dataset utilized is historically captured through a medication\nevent monitoring system. When the group who are prone to be non-adherent is\naccurately identified and targeted, it makes the way for improving patient care\nand helps Healthcare workers to assess and develop new strategies.")

elif Selected == "Models Tester":
    st.write("Coming Soon!")



        
