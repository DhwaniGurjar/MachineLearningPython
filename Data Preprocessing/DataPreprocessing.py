#Importing Libraries 
import numpy as np                  #Math and Array
import matplotlib.pyplot as plt     #Plot charts
import pandas as pd                 #Import datasets

#Importing Dataset
dataset = pd.read_csv('Data.csv')     
x = dataset.iloc[:,:-1].values              #x = Matrix of Features - mostly in first column
y = dataset.iloc[:,-1].values               #y = Dependent variables vector - mostly last column
print(x)
print(y)

#Taking Care of Missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
print(x)

#Encoding Categorical Data
#Encoding the indepedent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))          #X should be numpy array
print(x)
#Encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

#Feature Scaling - Standardisation / Normalisation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
print(x)

#Splitting dataset into the Training Set(80% of data to train ML) and the Test Set(20% of data to evaluate prediction on new observations)
from sklearn.model_selection import train_test_split    #tts is a function
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)
 