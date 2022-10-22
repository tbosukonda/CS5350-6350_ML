#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


train_df = pd.read_csv(r"train.csv", names = ["Cement", "Slag", "Fly_ash", "Water", "SP",
                                                       "Coarse_Aggr", "Fine_Aggr", "Output"])


# In[5]:


#train_df.head()


# In[157]:


class Gradient_Descent:
    def __init__(self, learning_rate = 1):
        self.lr = learning_rate
        self.weights = np.zeros(7)
        self.bias = 0
        self.cost = []
        self.steps = 0        
        
    def learn(self, X, y):
        while(True):
            y_pred = self._predict(X) #making prediction
            error, d_weight, d_bias = self._modify_gradient(X, y, y_pred) #computing error, dJ/dW, and dJ,dB
            converge = self._update_params(d_weight, d_bias) #updating the parameters
            cost = self.calc_cost(error) #calculating the cost
            self.steps = self.steps+1
            self.cost.append(cost)
            if converge == 1:
                break         
           
            
    def _predict(self, X):
        res = self.bias + np.dot(X, self.weights)
        return res
    
    def _modify_gradient(self, X, y, y_pred):
        error = y_pred - y
        d_weight = np.dot(X.T, error) #sum over all the examples (y_i - w^T.x_i)x_ij
        d_bias = np.sum(error) #sum over all the examples (y_i - w^T.x_i)
        
        return error, d_weight, d_bias
        
    def _update_params(self, d_weight, d_bias):
        converge = 0
        new_weights = self.weights - (self.lr * d_weight) #w^t+1 = w^t - r.dJ/dW
        new_bias = self.bias - (self.lr * d_bias) #r^t+1 = r^t - r.dJ/dB
        
        if np.sum(new_weights) - np.sum(self.weights) < (1/1000000):
            converge = 1
        else:
            self.weights = new_weights
            self.bias = new_bias
            self.lr = self.lr/2
        
        return converge
    
    def predict(self, X):
        y_pred = self._predict(X)
        return y_pred
    
    def calc_cost(self, error):
        c = 0.5 * np.sum(np.square(error))
        return c       
        


# In[6]:


test_df = pd.read_csv(r"test.csv", names = ["Cement", "Slag", "Fly_ash", "Water", "SP",
                                                       "Coarse_Aggr", "Fine_Aggr", "Output"])


# In[7]:


x_train = train_df.iloc[:,:7]
#x_train.head()


# In[160]:


y_train = train_df["Output"]
#y_train.head()


# In[169]:


gd_model = Gradient_Descent(learning_rate = 0.01)
gd_model.learn(x_train, y_train)


# In[170]:


x_test = test_df.iloc[:,:7]
#x_test.head()


# In[171]:


y_test = test_df["Output"]
#y_test.head()


# In[172]:


y_pred = gd_model.predict(x_test)


# In[173]:


def acc(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true)**2))

print(acc(y_test, y_pred))


# In[174]:


print("learned weight vector : ", gd_model.weights)
print("learned bias : ", gd_model.bias)
print("learned learning rate : ", gd_model.lr)


# In[175]:


cost = gd_model.cost
steps = [i+1 for i in range(gd_model.steps)]


# In[176]:


import matplotlib.pyplot as plt
plt.plot(steps, cost)


# In[153]:


print("cost vaue for tet data : ", gd_model.calc_cost(y_pred - y_test))


# In[ ]:




