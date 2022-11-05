#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


train = pd.read_csv("train.csv")
train.head()


# In[3]:


x_train = train.iloc[:,:4]
x_train.head()


# In[4]:


y_train = train.iloc[:,4]
y_train = y_train.replace(0, -1)
y_train.head()


# In[5]:


test = pd.read_csv("test.csv")
test.head()


# In[6]:


x_test = test.iloc[:,:4]
x_test.head()


# In[7]:


y_test = test.iloc[:,4]
y_test = y_test.replace(0, -1)
y_test.head()


# In[9]:


class Average_Perceptron:
    def __init__(self):
        self.weights = []
        self.curr_weight = [0,0,0,0]
        self.a = []
    
    def predict(self, x_test):
        y_pred = []
        for x in x_test.values:
            if np.dot(self.a[-1], x) <= 0:
                pred = -1
            else:
                pred = 1
            y_pred.append(pred)
        return np.array(y_pred)
    
    def fit(self, t, x_train, y, r, w):
        a = x_train.shape[1]*[0]
        for i in range(t):
            for x_sample, y_sample in zip(x_train.values, y):
                if y_sample * np.dot(w, x_sample) <= 0:
                    w = w + r*y_sample*x_sample
                a = a + w
                self.weights.append(w)
                self.curr_weight = a
                self.a.append(a)
            
    def calc_error(self, y_pred, y_test):
        cnt = 0
        for pred, true in zip(y_pred, y_test):
            if pred != true:
                cnt = cnt+1
        print("test misclassifications : ",cnt)
        return cnt/len(y_pred)
        


# In[10]:


weights = [0]*x_train.shape[1]
print(weights)
ap_mod = Average_Perceptron()
ap_mod.fit(10, x_train, y_train, 0.1, np.array(weights))


# In[12]:


y_predictions = ap_mod.predict(x_test)


# In[13]:


#print(y_predictions)


# In[14]:


test_error = ap_mod.calc_error(y_predictions, y_test)
print("average test error : ",test_error)


# In[18]:


print("learned weight vector : ",ap_mod.a[-1])


# In[19]:


#print(ap_mod.a)


# In[ ]:




