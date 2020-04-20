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