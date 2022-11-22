#!/usr/bin/env python
# coding: utf-8

# In[1]:


#loading the requires libraries
import pandas as pd
import numpy as np


# In[2]:


train_data = pd.read_csv("train.csv")
#train_data.head()


# In[3]:


#splitting into dependednt and independent feature 
x_train = train_data.iloc[:,:4]
y_train = train_data.iloc[:,4]


# In[4]:


#converting [0,1] feature values into [-1,1] values
y_train = y_train.replace(0,-1)


# In[5]:


#loading test dataset
test_data = pd.read_csv("test.csv")
#test_data.head()


# In[6]:


#splitting into dependednt and independent feature 
x_test = test_data.iloc[:,:4]
y_test = test_data.iloc[:,4]


# In[7]:


#converting [0,1] feature values into [-1,1] values
y_test = y_test.replace(0,-1)


# In[8]:


#initial weights and bias
initial_wt = np.array([0,0,0,0])
initial_bias = 0
initial_values = (initial_wt, initial_bias)


# In[9]:


weight_overview = []
bias_overview = []

def learning_rate_1(gamma_zero, a, t):
    learning_rate = gamma_zero/(1 + (gamma_zero/a) * t)
    return learning_rate

def learning_rate_2(gamma_zero, t):
    learning_rate = gamma_zero/(1 + t)

def stocasticSubgradientDescent(x_train, y_train, intial_values, C, gamma_zero, a, N, lr_model, T=1): #add the learning_rate model
    weight, bias = intial_values
    
    for epoch in range(T):
        learning_rate = learning_rate_1(gamma_zero, a, epoch)
        
        sub_gradient = subgradients(x_train, y_train, weight, bias, C, learning_rate, N)
        
        weight = weight - (learning_rate * sub_gradient[0]) #weight update
        bias = bias - (learning_rate * sub_gradient[1]) #bias update
        weight_overview.append(weight)
        bias_overview.append(bias)
        
    return (weight, bias)

def subgradients(x_train, y_train, weight, bias, C, lr_t, N):
    subgradient_weight = 0
    subgradient_bias = 0
    
    for xi, yi in zip(x_train.values, y_train.values):
        
        f_xi = np.dot(weight.T, xi) + bias
        
        
        decision_value = yi * f_xi
        
        if decision_value<=1:
            temp1 = lr_t * weight
            temp2 = lr_t * C * N * yi * xi
            subgradient_weight -= temp1 + temp2
            subgradient_bias -= lr_t * C * N * 1 * yi
            
        else:
            subgradient_weight += (1 - lr_t) * weight
            subgradient_bias += 0
            
        return (subgradient_weight, subgradient_bias)


# In[10]:


def predict(weight, x_test):
    y_pred = []
    for x in x_test.values:
        if np.dot(weight.T, x)<=1:
            pred = -1
        else:
            pred = 1
        y_pred.append(pred)
    return y_pred

def calc_error(y_pred, y_test):
    cnt = 0
    for pred, true in zip(y_pred, y_test):
        if pred != true:
            cnt = cnt+1
    return cnt/(len(y_test))


# In[11]:


#length of dataset
print(len(y_train))


# For gamma = gamma_zero/(1 + (gamma_zero/a)*t)
# 
# 
# For C = 100/873 

# In[12]:


weight, bias = stocasticSubgradientDescent(x_train, y_train, initial_values, (100/873), 0.01, 0.1, 871, "learning_rate_1", 100)


# In[13]:


print("learned weight : ", weight)
print("learned bias : ", bias)


# In[63]:


#print(weight_overview)


# In[64]:


#print(bias_overview)


# In[14]:


train_y_predictions = predict(weight, x_train)
print("train error : ", calc_error(train_y_predictions, y_train))


# In[15]:


y_predictions = predict(weight, x_test)
print("test error : ", calc_error(y_predictions, y_test))


# For C = 500/873

# In[16]:


weight, bias = stocasticSubgradientDescent(x_train, y_train, initial_values, (500/873), 0.01, 0.1, 871, "learning_rate_1", 100)


# In[17]:


print("learned weight : ", weight)
print("learned bias : ", bias)


# In[18]:


train_y_predictions = predict(weight, x_train)
print("train error : ", calc_error(train_y_predictions, y_train))


# In[19]:


y_predictions = predict(weight, x_test)
print("test error : ", calc_error(y_predictions, y_test))


# For C = 700/873

# In[20]:


weight, bias = stocasticSubgradientDescent(x_train, y_train, initial_values, (700/873), 0.01, 0.1, 871, "learning_rate_1", 100)


# In[21]:


print("learned weight : ", weight)
print("learned bias : ", bias)


# In[22]:


train_y_predictions = predict(weight, x_train)
print("train error : ", calc_error(train_y_predictions, y_train))


# In[23]:


y_predictions = predict(weight, x_test)
print("test error : ", calc_error(y_predictions, y_test))


# For gamma = gamma_zero/(1 + t)
# 
# 
# For C = 100/873 

# In[24]:


weight, bias = stocasticSubgradientDescent(x_train, y_train, initial_values, (100/873), 0.01, 0.1, 871, "learning_rate_2", 100)


# In[25]:


print("learned weight : ", weight)
print("learned bias : ", bias)


# In[26]:


train_y_predictions = predict(weight, x_train)
print("train error : ", calc_error(train_y_predictions, y_train))


# In[27]:


y_predictions = predict(weight, x_test)
print("test error : ", calc_error(y_predictions, y_test))


# For C = 500/873

# In[28]:


weight, bias = stocasticSubgradientDescent(x_train, y_train, initial_values, (500/873), 0.01, 0.1, 871, "learning_rate_2", 100)


# In[29]:


print("learned weight : ", weight)
print("learned bias : ", bias)


# In[30]:


train_y_predictions = predict(weight, x_train)
print("train error : ", calc_error(train_y_predictions, y_train))


# In[31]:


y_predictions = predict(weight, x_test)
print("test error : ", calc_error(y_predictions, y_test))


# For C = 700/873

# In[32]:


weight, bias = stocasticSubgradientDescent(x_train, y_train, initial_values, (700/873), 0.01, 0.1, 871, "learning_rate_2", 100)


# In[33]:


print("learned weight : ", weight)
print("learned bias : ", bias)


# In[34]:


train_y_predictions = predict(weight, x_train)
print("train error : ", calc_error(train_y_predictions, y_train))


# In[35]:


y_predictions = predict(weight, x_test)
print("test error : ", calc_error(y_predictions, y_test))

