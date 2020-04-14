#Importing Libraries
 
import numpy as np                  #Math and Array
import matplotlib.pyplot as plt     #Plot charts
import pandas as pd                 #Import datasets

#Importing Dataset

dataset = pd.read_csv('Data.csv');      
x = dataset.iloc[:,:-1].values              #x = Matrix of Features - mostly in first column
y = dataset.iloc[:,-1].values               #y = Dependent variables vector - mostly last column
print(x)
print(y)