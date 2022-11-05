#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np


# In[14]:


train = pd.read_csv("train.csv")
train.head()


# In[15]:


x_train = train.iloc[:,:4]
x_train.head()


# In[16]:


y_train = train.iloc[:,4]
y_train = y_train.replace(0, -1)
y_train.head()


# In[17]:


test = pd.read_csv("test.csv")
test.head()


# In[18]:


x_test = test.iloc[:,:4]
x_test.head()


# In[19]:


y_test = test.iloc[:,4]
y_test = y_test.replace(0, -1)
y_test.head()


# In[62]:


class Voted_Perceptron:
    def __init__(self):
        self.weights = []
        self.curr_weight = [0,0,0,0]
        self.cnt = []
        self.m = 0  
        self.weight_count = []
    
    def predict(self, x_test):
        y_pred = []
        for x in x_test.values:
            pred = 0
            for cnt in self.cnt:
                pred = pred + cnt * np.sign(np.dot(self.curr_weight, x))
                
            y_pred.append(np.sign(1 if pred>=0 else -1))
            
        return np.array(y_pred)
    
    def fit(self, t, x_train, y, r, w):
        m = 0
        cnt = [0]
        
        for i in range(t):
            c_m = 0
            for x_sample, y_sample in zip(x_train.values, y):
                
                if y_sample * np.dot(w, x_sample) <= 0:                    
                    w = w + r*y_sample*x_sample
                    c_m = 1
                    cnt.append(1)
                    m = m+1
                else:
                    cnt[m] = cnt[m] + 1
                    c_m = c_m + 1
                    
            self.weight_count.append((w,c_m))        
            self.weights.append(w)
            self.curr_weight = w
            self.cnt = cnt
            self.m = m
            
    def calc_error(self, y_pred, y_test):
        cnt = 0
        for pred, true in zip(y_pred, y_test):
            if pred != true:
                cnt = cnt+1
        print("test misclassification : ", cnt)
        return cnt/len(y_pred)
        


# In[63]:


weights = [0]*x_train.shape[1]
print(weights)
vp_mod = Voted_Perceptron()
vp_mod.fit(10, x_train, y_train, 0.1, np.array(weights))


# In[64]:


print(vp_mod.curr_weight)


# In[65]:


y_predictions = vp_mod.predict(x_test)


# In[66]:


#print(y_predictions)


# In[67]:


print("correct train classifications : ", vp_mod.m)
test_error = vp_mod.calc_error(y_predictions, y_test)
print("average test error : ", test_error)


# In[69]:


print("weight vectors and their count : ",vp_mod.weight_count)


# In[ ]:




