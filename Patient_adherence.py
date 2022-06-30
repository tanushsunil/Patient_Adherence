import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
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
   model = st.selectbox('choose a model',
      ('Decision tree Classifier', 'K-Nearest Neighbor', 'Logistic Regression', 'Random Forest Classifier'))

   if model =='Decision tree Classifier':
       m = pickle.load(open('Decision_Tree_Classifier.sav', 'rb'))
   elif model =='K-Nearest Neighbor':
       m = pickle.load(open('K-Nearest Neighbor.sav', 'rb'))
   elif model == 'Logistic Regression':
       m = pickle.load(open('Logistic_Regression.sav', 'rb'))
   else:     
       m = pickle.load(open('Random_Forest_Classifier.sav', 'rb'))
    
   # creating a function for Prediction
   def adherence_prediction(input_data):

    # changing the input_data to numpy array
           input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
           input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

           prediction = m.predict(input_data_reshaped)
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

elif Selected == "Home":
    st.header("What are we solving?")
    st.text("The problem addressed by this model is finding out if the patient is medically\nadherent or not. clinical studies have demonstrated that only 50-70% of patients\nadhere properly to prescribed drug therapy. This behavior of adherence failure\ncan cause health issues, hospitalization risk and even death. Patient Adherence\ninsights can prove useful for \n\n     1.Doctors who prescribe drugs\n     2.Drug / Medication producers\n     3.Government")
    st.image("https://www.omadahealth.com/hs-fs/hubfs/BlogIllustrations_medicationbarriers6.png?width=2860&name=BlogIllustrations_medicationbarriers6.png")
    st.title("What is our model focused on?")
    st.text("This model is focused on the prediction of adherence behavior with individual\nselection. The dataset utilized is historically captured through a medication\nevent monitoring system. When the group who are prone to be non-adherent is\naccurately identified and targeted, it makes the way for improving patient care\nand helps Healthcare workers to assess and develop new strategies.")

elif Selected == "Models Tester":
    
    def model_selection():

      Model_op = st.selectbox('Which model?',('Logistic Regression Classifier', 'K-Nearest Neighbor', 'Decision Tree Classifier', 'Random Forest Classifier'))

      global model

      if Model_op == 'Logistic Regression Classifier':
          model = pickle.load(open('Logistic_Regression.sav', 'rb'))
      elif Model_op == 'K-Nearest Neighbor':
          model = pickle.load(open('K-Nearest_Neighbor.sav', 'rb'))
      elif Model_op == 'Decision Tree Classifier':
          model = pickle.load(open('Decision_Tree_Classifier.sav', 'rb'))
      else:
          model = pickle.load(open('Random_Forest_Classifier.sav', 'rb'))


    def main():
        global dataset, x, y, x_test, x_train, y_test, y_train, y_pred
        dataset = pd.read_csv(r'patient_data_processed.csv')

        dataset_2 = pd.read_csv(r'patient_data.csv')   
 
 
        st.header('Test the performance of our models:')
        st.write('These are our trained models which are appiled on our sample dataset. You can select any one out of our four models and click on the test button to get the accuracy. :smile:')
        st.subheader('Sample dataset:')
        st.dataframe(data=dataset_2)   
        model_selection()


        le = LabelEncoder()
        dataset['Adherence_n'] = le.fit_transform(dataset['Adherence'])
        dataset = dataset.drop(['Unnamed: 0', 'Adherence'], axis='columns')

        x = dataset.iloc[:, [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 14, 15, 16, 17]]
        y = dataset.iloc[:, 9]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        y_pred = " "
        score = " "
    


        if st.button("Test the model"):
            y_pred = model.predict(x_test)
            score = metrics.accuracy_score(y_pred, y_test)
              
        st.write("The accuracy of the selected model is:")
        st.success(score*100)
        
    if __name__ == '__main__':
     main()
    
    
elif Selected == 'About us':
    
    st.title('About our team:')
    st.write('This team consists of a trio of data science freshers and our main goal is to provide the user with accurate results and a super user-friendly app. ')
    
    st.header('Team Members:')
    
    Selected = option_menu(
    menu_title='Team Members:',
    options=["Tanush Sunil", "Abinayasree", "Harshithaa"],
    icons=["lightning-charge", "gem", "emoji-laughing"],
    menu_icon=['people-fill'],    
    default_index=0,
    orientation="horizontal")
    
    
    if Selected == 'Tanush Sunil':
         st.title('Hello, I\'m Tanush Sunil')
         st.subheader('My perspective on this project')
         st.write('')
         st.write('[Instagram](https://www.instagram.com/tanush_sunil/)')
         st.write('[GitHub](https://github.com/tanushsunil)')
    elif Selected == 'Abinayasree':
         st.title('Hi, I\'m Abinayasree')
         st.subheader('My perspective on this project')
         st.write('')
         st.write('[Instagram](https://www.instagram.com/abbiee__s/)')
    else:
         st.title('Hi, I\'m Harshithaa')
         st.subheader('My perspective on this project')
         st.write('')
         st.write('[Instagram](https://www.instagram.com/___hershy______/)')
    
    
    
    
    



        
