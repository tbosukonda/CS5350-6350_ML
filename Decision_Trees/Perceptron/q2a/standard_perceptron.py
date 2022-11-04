#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


train = pd.read_csv("train.csv")
train.head()


# In[3]:


class Perceptron:
    def __init__(self):
        self.weights = []
        self.curr_weight = [0,0,0,0]
    
    def predict(self, x_test):
        y_pred = []
        for x in x_test.values: #predicting for each and every example in the test dataset
            if np.dot(self.curr_weight.transpose(), x) <= 0:
                pred = -1 #negative class
            else:
                pred = 1  #positive class
            y_pred.append(pred)
        return np.array(y_pred)
    
    def fit(self, t, x_train, y, r, w):
        for i in range(t):
            for x_sample, y_sample in zip(x_train.values, y):
                if y_sample * np.dot(w.transpose(), x_sample) <= 0: #error in classification
                    w = w + r*y_sample*x_sample  #updating the weights
                self.weights.append(w)
                self.curr_weight = w
            
    def calc_error(self, y_pred, y_test):
        cnt = 0
        for pred, true in zip(y_pred, y_test):
            if pred != true:
                cnt = cnt+1
        print("Number of misclassifications : ", cnt)
        return cnt/len(y_pred)
        


# In[4]:


x_train = train.iloc[:,:4]
x_train.head()


# In[5]:


print(x_train.shape[1])


# In[6]:


y_train = train.iloc[:,4]
y_train = y_train.replace(0, -1)
y_train.head()


# In[7]:


weights = [0]*x_train.shape[1]
#print(weights)
p_mod = Perceptron()
p_mod.fit(10, x_train, y_train, 0.1, np.array(weights).T)


# In[8]:


#print(p_mod.weights)


# In[9]:


test = pd.read_csv("test.csv")
test.head()


# In[10]:


x_test = test.iloc[:,:4]
x_test.head()


# In[11]:


y_test = test.iloc[:,4]
y_test = y_test.replace(0, -1)
y_test.head()


# In[12]:


y_predictions = p_mod.predict(x_test)


# In[13]:


test_error = p_mod.calc_error(y_predictions, y_test)
print("Average test error : ", test_error)


# In[153]:


print("Learned weights : ", p_mod.curr_weight)


# In[ ]:




