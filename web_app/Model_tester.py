import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pickle


def model_selection():

    Model_op = st.selectbox('Which model?',('Logistic Regression Classifier', 'K-Nearest Neighbor', 'Decision Tree Classifier'))

    global model

    if Model_op == 'Logistic Regression Classifier':
        model = pickle.load(open('Logistic_Regression.sav', 'rb'))
    elif Model_op == 'K-Nearest Neighbor':
        model = pickle.load(open('K-Nearest_Neighbor.sav', 'rb'))
    else:
        model = pickle.load(open('Decision_Tree_Classifier.sav', 'rb'))


def main():
 global dataset, x, y, x_test, x_train, y_test, y_train, y_pred
 dataset = pd.read_csv(
    r'C:\Users\tanus\Desktop\patient_adherence\Dataset\patient_data_processed.csv')

 dataset_2 = pd.read_csv(r'C:\Users\tanus\Desktop\patient_adherence\Dataset\patient_data.csv')   
 
 
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
 

