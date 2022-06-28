import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics


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
    Marital_status_1 = st.selectbox(
        'Is the patient married?', ('Married', 'Unmarried'))
    Diabetes_1 = st.selectbox('Does the patient have Diabetes?', ('Yes', 'No'))
    Alcoholism_1 = st.selectbox(
        'Does the patient have Alcoholism?', ('Yes', 'No'))
    HyperTension_1 = st.selectbox(
        'Does the patient have HyperTension?', ('Yes', 'No'))
    Smokes_1 = st.selectbox('Does the patient smoke?', ('Yes', 'No'))
    Tuberculosis_1 = st.selectbox(
        'Does the patient have Tuberculosis?', ('Yes', 'No'))
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
