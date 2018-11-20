
# coding: utf-8

# In[168]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score


# In[169]:


data = pd.read_csv("zomato.csv")
data.head()


# In[170]:


#storing all different currencies of different countries
cur=data['Currency']
cur = set(cur)
cur = list(cur)


# In[171]:


#dictionary of different currencies and their conversion rate to dollars
dic={}
dic[cur[0]] = 0.094 #currency in dollars
dic[cur[1]] = 0.19
dic[cur[2]] = 0.000068
dic[cur[3]] = 0.27
dic[cur[4]] = 1.28
dic[cur[5]] = 0.69
dic[cur[6]] = 1
dic[cur[7]] = 0.27
dic[cur[8]] = 0.27
dic[cur[9]] = 0.071
dic[cur[10]] = 0.0057
dic[cur[11]] = 0.014


# In[172]:


#converting all currencies to dollars
for i in range(9551):
    data['Average Cost for two'][i] = data['Average Cost for two'][i] * dic[data['Currency'][i]]
data.head()


# In[173]:


#dropping all the outliers with avg cost for two>$80 it won't effect the data too much and it would make it more logical
data = data[data['Average Cost for two']<80]
#dropping all votes >1000 as they are working as outliers
data = data[data['Votes']<1000]
#splitting the data between train data and test data
train_Y,test_Y,train_X,test_X = train_test_split(data['Aggregate rating'],data[['Average Cost for two','Votes','Price range']],test_size = 0.2,random_state = 2)
train_X.describe()


# In[174]:


data[['Votes','Average Cost for two','Price range','Aggregate rating']].describe()


# In[175]:


# plot of 'Aggregate rating' vs 'Average Cost for two'
hy = data['Aggregate rating']
hx = data['Average Cost for two']
plt.xlabel('Average Cost for two (in dollars)', fontsize=10)
plt.ylabel('Aggregate rating (out of 5)', fontsize=10)
plt.plot(hx,hy,"r.")
plt.show()


# In[176]:


# plot of 'Aggregate rating' vs 'Votes'
hy = data['Aggregate rating']
hx = data['Votes']
plt.xlabel('Votes', fontsize=10)
plt.ylabel('Aggregate rating (out of 5)', fontsize=10)
plt.plot(hx,hy,"b.")
plt.show()


# In[177]:


# plot of 'Aggregate rating' vs 'Price range'
hy = data['Aggregate rating']
hx = data['Price range']
plt.xlabel('Price range', fontsize=10)
plt.ylabel('Aggregate rating (out of 5)', fontsize=10)
plt.plot(hx,hy,"g.")
plt.show()


# In[178]:


#plot of rating with respect to features Price range,Average Cost of two,votes
plt.figure(figsize=(10,10))
plt.title('Values of training data')
plt.xlabel("Values of features Price range,Average Cost of two,votes")
plt.ylabel("Values of training data output")
plt.plot(train_X,train_Y,"b.")
plt.show()


# In[179]:


#training of the model
regressor = RandomForestRegressor(n_estimators = 300)
regressor.fit(train_X, train_Y)


# In[180]:


# K-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = train_X, y = train_Y, cv = 10)
mean_accuracy = accuracies.mean()
std_accuracy = accuracies.std()
std_accuracy


# In[181]:


#prediction of test data
y_pred = regressor.predict(test_X)
y_pred


# In[182]:


#expected output
test_Y


# In[183]:


#our accuracy
regressor.score(test_X, test_Y)


# In[184]:


#plot between y actual and y_predicted with respect to test input 
plt.figure(figsize=(10,10))
plt.plot(test_X,test_Y,"b.")
plt.plot(test_X,y_pred,"r.")
plt.title('Values of test data')
plt.xlabel("Values of features Price range,Average Cost of two,votes")
plt.ylabel("Values of expected output and actua output data output")
plt.show()

