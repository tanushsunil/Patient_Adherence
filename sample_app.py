import numpy as np
import pickle
import streamlit as st


Model_op = st.selectbox('Which model?',
                        ('Logistic Regression Classifier', 'K-Nearest Neighbor'))

if Model_op == 'Logistic Regression Classifier':
    model = pickle.load(open('Mode\Logistic_Regression.sav', 'rb'))
    st.write('You selected:', Model_op)
else:
    model = pickle.load(open('Model\K-Nearest_Neighbor.sav', 'rb'))
    st.write('You selected:', Model_op)


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
    Gender = st.selectbox('Select you Gender',
                          ('Male', 'Female'))

    if Gender == 'Male':
        Male, Female = 1, 0
        st.write('You selected:', Gender)
    else:
        Male, Female = 0, 1
        st.write('You selected:', Gender)

    # code for Prediction
    adherence_diagnosis = ''

    # creating a button for Prediction

    if st.button('Get Adherence Test Result'):
        adherence_diagnosis = adherence_prediction(
            [Age, Prescription_Days, Male, Female])

    st.success(adherence_diagnosis)


if __name__ == '__main__':
    main()
