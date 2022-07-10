# Packages---------------------------->Importing all the required packages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pickle

# Importing Data------------>>Reading .CSV & doing a light data processing
dataset = pd.read_csv(r'C:\Users\tanus\Desktop\patient_adherence\Dataset\patient_data_processed.csv')

le = LabelEncoder()
dataset['Adherence_n'] = le.fit_transform(dataset['Adherence'])
dataset = dataset.drop(['Unnamed: 0', 'Adherence'], axis='columns')

# Feature Selection---->>Selecting the required features to feed the model
x = dataset.iloc[:, [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 14, 15, 16, 17]]
y = dataset.iloc[:, 9]

# Data splitting------>>Splitting data into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)


# Logistic Regression CLassification----------------->>Type 1 Classifier
def lr_prediction():    
    global lr
    lr = LogisticRegression(C=0.5)
    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)

    score = metrics.accuracy_score(y_test, y_pred)
    print("\nAccuracy of Logistic Regression Model: ", score*100, '%')

    cm = confusion_matrix(y_test, y_pred)
    print('The confusion matrix for this model is \n', cm)


# K-Nearest Neighbor Classifier---------------------->>Type 2 Classifier
    
def knn_prediction():   
    global knn 
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)

    score = metrics.accuracy_score(y_test, y_pred)
    print("\nAccuracy of KNN Model: ", score*100, '%')

    cm = confusion_matrix(y_test, y_pred)
    print('The confusion matrix for this model is \n', cm)


# Decision Tree Classifier---------------------------->>Type 3 Classifier
def deci_prediction():
    global deci_tree
    deci_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    deci_tree.fit(x_train, y_train)

    y_pred = deci_tree.predict(x_test)

    score = metrics.accuracy_score(y_pred, y_test)
    print('\nAccuracy of Decision Tree : ', score*100, "%")

    cm = confusion_matrix(y_test, y_pred)
    print('The confusion matrix for this model is \n', cm)

    

# Random Forest Classifier---------------------------->>Type 4 Classifier
def random_forest():
    global rf_pred
    rf_pred = RandomForestClassifier(n_estimators=2, criterion = 'entropy', random_state= 0)
    rf_pred.fit(x_train, y_train)

    y_pred = rf_pred.predict(x_test)

    score = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy of Random Forest Classifer: ", score*100, "%")

    from sklearn.metrics import confusion_matrix
    cm= confusion_matrix(y_test, y_pred)
    print("Confusion Matrix")
    print(cm)

# Predicting results
lr_prediction()
knn_prediction()
deci_prediction()
random_forest()


# Pickling Process----------------->>Exporting the Models into .sav files
"""filename = ('K-Nearest_Neighbor.sav')
pickle.dump(knn, open(filename,'wb'))

filename_2 = ('Logistic_Regression.sav')
pickle.dump(lr, open(filename_2,'wb'))

filename_3 = ('Decision_Tree_Classifier.sav')
pickle.dump(deci_tree, open(filename_3,'wb'))"""

filename_4 = ('Random_Forest_Classifier.sav')
pickle.dump(rf_pred, open(filename_4,'wb'))


# Motivation :)----------------------------------------->>The way to go!!
print("\n\nThat's the spirit! Only deployment is pending.\nHope it's a piece of cake for you!")

# What are you looking for? Thats the end!
#Credits - Abinayasree, Tanush, Harshithaa
