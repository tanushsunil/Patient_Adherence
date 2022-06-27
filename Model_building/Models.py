#Packages--------------------------------------------------------------------------->Importing all the required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pickle

#Importing Data------------------------------->>Reading .CSV & doing a light processing (Enna ramalingam epidi irukka, data processing la paathadhu)
dataset = pd.read_csv(r'C:\Users\tanus\Desktop\patient_adherence\Dataset\patient_data_processed.csv')

le = LabelEncoder()
dataset['Adherence_n'] = le.fit_transform(dataset['Adherence'])
dataset = dataset.drop(['Unnamed: 0', 'Adherence'], axis='columns')

#Feature Selection--------------------------------------------------->>Selecting the required features to feed the model
x = dataset.iloc[:, [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17]]
y = dataset.iloc[:, 9]

#Data splitting------------------------------------------------------>>Splitting data into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


#Logistic Regression CLassification------------------------------------------------------------------>>Type 1 Classifier
lr = LogisticRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

score = metrics.accuracy_score(y_test, y_pred)
print("\nAccuracy of Logistic Regression Model: ", score*100, '%')

cm = confusion_matrix(y_test, y_pred)
print('The confusion matrix for this model is \n', cm)

#K-Nearest Neighbor Classifier---------------------------------------------------------------------->>Type 2 Classifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

score = metrics.accuracy_score(y_test, y_pred)
print("\nAccuracy of KNN Model: ", score*100, '%')

cm = confusion_matrix(y_test, y_pred)
print('The confusion matrix for this model is \n', cm)

#Decision Tree Classifier--------------------------------------------------------------------------->>Type 3 Classifier
deci_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
deci_tree.fit(x_train, y_train)

y_pred = deci_tree.predict(x_test)

score = metrics.accuracy_score(y_pred, y_test)
print('\nAccuracy of Decision Tree : ', score*100,"%")

cm = confusion_matrix(y_test, y_pred)
print('The confusion matrix for this model is \n', cm)

#Random Forest Classifier--------------------------------------------------------------------------->>Type 4 Classifier


#Pickling Process---------------------------------------------------------------->>Exporting the Models into .sav files
filename = ('K-Nearest_Neighbor.sav')
pickle.dump(knn, open(filename,'wb'))

filename_2 = ('Logistic_Regression.sav')
pickle.dump(lr, open(filename_2,'wb'))

filename_3 = ('Decision_Tree_Classifier.sav')
pickle.dump(deci_tree, open(filename_3,'wb'))


#Motivation :)--------------------------------------------------------------------------------------->>The way to go!!
print("That's the spirit! Only deployment is pending.\nHope it's a piece of cake for you!")

#What are you looking for? Thats the end!
#Credits - Abinayasree, Tanush, Harshithaa