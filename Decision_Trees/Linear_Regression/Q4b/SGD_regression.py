#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[283]:


class Stochastic_Gradient_Descent:
    def __init__(self, learning_rate = 1):
        self.lr = learning_rate
        self.weights = np.zeros(7)
        self.bias = 0
        self.lr_curve = [self.lr]
        self.steps = 0
        self.cost = [1]
        
    def learn(self, X_train, y_train):
        
        while(True):
            converge = 0
            c_row = 1
            n = X_train.shape[0]
            old_cost = 1

            while(c_row < n):
                x = X_train.iloc[c_row, :]  
                y = y_train.iloc[c_row]     

                y_pred = self._predict(x) #making prediction
                error, d_w, d_b = self._modify_gradient(x, y, y_pred) #computing error, dJ/dW, and dJ,dB   
                cost = self.calc_cost(error) #finding the cost
                #print("old cost:", old_cost)
                #print("new cost:", cost)
                diff_cost = cost-old_cost
                #print("diff_cost", diff_cost)
                n_w, n_b, n_lr = self._update_params(d_w, d_b) #computing updates parameters
                
                if abs(diff_cost) < (1/1000): #converging once the cost difference between old and new value is less than 0.001
                    converge = 1
                    
                if converge == 1:
                    break
                
                #if the model doesn't converge, then updating the parameters
                old_cost = cost
                self.cost.append(cost)
                self.weights = n_w
                self.bias = n_b
                self.lr = n_lr
                self.lr_curve.append(self.lr)
                c_row = c_row+1
                self.steps = self.steps+1
                
            if converge == 1:
                    break
                
            
    def _predict(self, x):
        res = self.bias + np.dot(x, self.weights) #y_pred = b+ w^T.x
        return res
    
    def _modify_gradient(self, x, y, y_pred):
        error = y - y_pred #error = actual- predicted
        d_weight = np.dot(x.T, error) #sum over all the examples (y_i - w^T.x_i)x_ij
        d_bias = error #sum over all the examples (y_i - w^T.x_i)
        
        return error, d_weight, d_bias
        
    def _update_params(self, d_weight, d_bias):
        
        new_weights = self.weights + (self.lr * d_weight) #w^t+1 = w^t - r.dJ/dW
        new_bias = self.bias + (self.lr * d_bias) #r^t+1 = r^t - r.dJ/dB        
        new_lr = self.lr/2        
        return new_weights, new_bias, new_lr
        
    
    def predict(self, X):
        y_pred = self._predict(X)
        return y_pred
    
    def calc_cost(self, error):
        c = 0.5 * np.sum(np.square(error))
        return c


# In[284]:


train_df = pd.read_csv(r"train.csv", names = ["Cement", "Slag", "Fly_ash", "Water", "SP",
                                                       "Coarse_Aggr", "Fine_Aggr", "Output"])


# In[285]:


#train_df.head()


# In[286]:


x = train_df.iloc[1, :]
#print(x)


# In[287]:


x_train = train_df.iloc[:,:7]
#x_train.head()


# In[288]:


y_train = train_df["Output"]
#y_train.head()


# In[289]:


sgd_model = Stochastic_Gradient_Descent(learning_rate = 0.1)
sgd_model.learn(x_train, y_train)


# In[290]:


test_df = pd.read_csv(r"test.csv", names = ["Cement", "Slag", "Fly_ash", "Water", "SP",
                                                       "Coarse_Aggr", "Fine_Aggr", "Output"])


# In[291]:


x_test = test_df.iloc[:,:7]
#x_test.head()


# In[292]:


y_test = test_df["Output"]
#y_test.head()


# In[293]:


y_pred = sgd_model.predict(x_test)


# In[294]:


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true)**2))

print(rmse(y_test, y_pred))


# In[295]:


print("learned weight vector : ", sgd_model.weights)


# In[296]:


print("learned bias value : ", sgd_model.bias)


# In[297]:


print("learning rate : ", sgd_model.lr)


# In[298]:


lr_curve = sgd_model.lr_curve
steps = [i for i in range(sgd_model.steps+1)]

import matplotlib.pyplot as plt

plt.plot(steps, lr_curve)
plt.xlabel("Iterations")
plt.ylabel("Learning rate")
plt.title("Learning rate curve")
plt.show()


# In[299]:


cost_curve = sgd_model.cost
steps = [i for i in range(sgd_model.steps+1)]

plt.plot(steps, cost_curve)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("cost curve")
plt.show()


# In[ ]:




