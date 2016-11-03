import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


train = pd.read_csv("..\\Data\\train.csv") #read train dataset
test = pd.read_csv("..\\Data\\test.csv")   #read test dataset

#print train.describe()

x = train.drop(['loss','id'], 1)
y = np.log1p(train["loss"]) #Apply log 1 + loss. Note: remeber to subtract shift when getting end result
ids = test['id'] #needed for writing results.csv

#Now, we need to split the dataset's columns into categorical features and continuous features and hot encode all categorical features

cat = []
cont = []
for colName in x.columns:
    if colName.startswith("cat"):
        cat.append(colName)
    elif colName.startswith("cont"):
        cont.append(colName)
    else:
        print "Oops! Unknown ColName encountered"
        

        









