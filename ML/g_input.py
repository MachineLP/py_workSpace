# -*- coding: utf-8 -*-
"""
Created on 2017 11.17
@author: liupeng
"""

import pandas as pd  
import numpy as np  
from sklearn.preprocessing import LabelEncoder  
from sklearn.preprocessing import StandardScaler  
  
train_data = pd.read_csv("bank-full.csv")  

train_data = train_data['age;"job";"marital";"education";"default";"balance";"housing";"loan";"contact";"day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"'].str.split(';',expand=True)  

text = '"job";"marital";"education";"default";"balance";"housing";"loan";"contact";"day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"'
text = str(text)
text = text.split(';')

train_data = pd.DataFrame(train_data.values[0::, 1::], index=train_data.values[0::, 0].astype(np.int), columns = text)  
train_data.index.name = "age" 
train_data.to_csv("input.csv")  

# print (train_data)


train_data = pd.read_csv("input.csv")  

data = []
y = train_data['age']
data.append(y)
train_data['age'] = y
y = train_data['"job"']
y = LabelEncoder().fit(y).transform(y)  
data.append(y)
train_data['"job"'] = y
y = train_data['"marital"']
y = LabelEncoder().fit(y).transform(y)  
data.append(y)
train_data['"marital"'] = y
y = train_data['"education"']
y = LabelEncoder().fit(y).transform(y)  
data.append(y)
train_data['"education"'] = y
y = train_data['"default"']
y = LabelEncoder().fit(y).transform(y)  
data.append(y)
train_data['"default"'] = y
y = train_data['"balance"']
data.append(y)
train_data['"balance"'] = y
y = train_data['"housing"']
y = LabelEncoder().fit(y).transform(y)  
data.append(y)
train_data['"housing"'] = y
y = train_data['"loan"']
y = LabelEncoder().fit(y).transform(y)  
data.append(y)
train_data['"loan"'] = y
y = train_data['"contact"']
y = LabelEncoder().fit(y).transform(y)  
data.append(y)
train_data['"contact"'] = y
y = train_data['"day"']
data.append(y)
train_data['"day"'] = y
y = train_data['"month"']
y = LabelEncoder().fit(y).transform(y)  
data.append(y)
train_data['"month"'] = y
y = train_data['"duration"']
data.append(y)
train_data['"duration"'] = y
y = train_data['"campaign"']
data.append(y)
train_data['"campaign"'] = y
y = train_data['"pdays"']
data.append(y)
train_data['"pdays"'] = y
y = train_data['"previous"']
data.append(y)
train_data['"previous"'] = y
y = train_data['"poutcome"']
y = LabelEncoder().fit(y).transform(y)  
data.append(y)
train_data['"poutcome"'] = y
y = train_data['"y"']
y = LabelEncoder().fit(y).transform(y)  
data.append(y)
train_data['"y"'] = y

# 将每一列数据进行归一化
train_data = train_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))  #方法一  
print ('train_data.values[0::, 0]', train_data.values[0::, 0])

train_data = pd.DataFrame(train_data.values[0::, 1::], index=train_data.values[0::, 0], columns = text)  
train_data.index.name = "age" 
train_data.to_csv("train_data.csv")  

print (train_data)

