#C:\Users\RAdhikesavan\Documents\Personal\Kaggle\Code

import os

#Change working directory
os.chdir("C:\Users\RAdhikesavan\Documents\Personal\Kaggle\Code")

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

cat = []
cont = []
columns = list(train)

cat = [x for x in columns if x.startswith("cat")]
con = [x for x in columns if x.startswith("con")]


#Get the unique set of category labels for each column
labels = {}
for colName in cat:
    trainLabels = train[colName].unique() 
    testLabels = test[colName].unique()
    labels[colName] = (list(set(trainLabels) | set(testLabels))) 

le = LabelEncoder()
catTrainFeatures = []
catTestFeatures = []

for col in cat:
    #Label encode Train and Test with the same encoder    
    le.fit(labels[col])
    trainFeature = le.transform(train[col])
    trainFeature = trainFeature.reshape(train.shape[0],1)
    
    testFeature = le.transform(test[col])
    testFeature = testFeature.reshape(test.shape[0],1)
    
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[col]))
    trainFeature = onehot_encoder.fit_transform(trainFeature)
    
    catTrainFeatures.append(trainFeature)
    catTestFeatures.append(testFeature)
    
catTrainFeatures = np.column_stack(catTrainFeatures)
catTestFeatures = np.column_stack(catTestFeatures)

trainData = np.concatenate((catTrainFeatures, train[cont]),axis=1)
testData = np.concatenate((catTestFeatures, test[cont]),axis=1)




