import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("..\\Data\\train.csv") #read train dataset
test = pd.read_csv("..\\Data\\test.csv")   #read test dataset

#print train.describe()

x = train.drop(['loss','id'], 1)
y = np.log1p(train["loss"]) #Apply log 1 + loss. Note: remeber to subtract shift when getting end result
ids = test['id'] #needed for writing results.csv

'''
Split the features into continuous and categorical,
Simpler way to do this is to just read first 116 columns as categorical and the remaining as continious.
But I hate hardcoding, so .... 
'''
cat = []
cont = []
for colName in x.columns:
    if colName.startswith("cat"):
        cat.append(colName)
    elif colName.startswith("cont"):
        cont.append(colName)
    else:
        print "Oops! Unknown ColName encountered"

'''    
le = LabelEncoder() 
oHe = OneHotEncoder()
x[cat] = x[cat].apply(le.fit_transform, axis = 1) #Apply encoder to all columns with categorical features 

'''








