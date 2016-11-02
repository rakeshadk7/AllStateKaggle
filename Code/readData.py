import sklearn
import pandas as pd
import numpy as np

train = pd.read_csv("..\\Data\\train.csv") #read train dataset
test = pd.read_csv("..\\Data\\test.csv")   #read test dataset

#print train.describe()

train = train.drop(['id'], axis=1) #id is not a feature, so drop it
test = test.drop(['id'], axis=1)

train["loss"] = np.log1p(train["loss"]) #Aplly log 1 + loss

#Now, we need to split the dataset's columns into categorical features and continuous features and use hot encode all categorical features




