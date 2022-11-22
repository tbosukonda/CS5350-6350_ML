#!/usr/bin/env python
# coding: utf-8

# In[11]:


#importing the libraries
import pandas as pd
import numpy as np
import scipy.optimize
from math import exp


# In[12]:


#loading the train datset
train_data = pd.read_csv("train.csv")
#train_data.head()


# In[13]:


#separating dependent and independent features 
#converting dataframe to numpy array 
x_train = np.array(train_data.iloc[:,:4])
y_train = train_data.iloc[:,4]


# In[14]:


#converting target [0,1] feature values to [-1, 1] values
y_train = np.array(y_train.replace(0,-1))


# In[15]:


#loading test dataset
test_data = pd.read_csv("test.csv")
#test_data.head()


# In[16]:


#separating dependent and independent features 
#converting dataframe to numpy array 
x_test = np.array(test_data.iloc[:,:4])
y_test = test_data.iloc[:,4]


# In[17]:


#converting target [0,1] feature values to [-1, 1] values
y_test = np.array(y_test.replace(0,-1))


# In[20]:


def gaussian_kernel(x, y, gamma):
    # Gaussian kernel values is given as : K_rbf(x,y) = exp((-||x-y||^2)/c)
    
    num = pow(np.linalg.norm(x-y, ord=2),2)
    return exp(-num / gamma)

class Dual_SVM:
    def __init__(self, x, y, C, kernel = "dot", gamma=None):
        self.w_asterisk = np.ndarray([])
        self.b_asterisk = 0.0
        self.support = []
        
                
    def func(self, d, gamma, kernel):
        #predicting using respective kernel value
        if kernel == "dot":
            return np.sign(np.dot(self.w_asterisk, d) + self.b_asterisk)
        
        if kernel == "gaussian":
            return np.sign(gaussian_kernel(self.w_asterisk, d, gamma) + self.b_asterisk)

    #predictions for noraml kernel    
    def predict_dot(self, x, kernel = "dot", gamma = None):
        r = [] #list that stores the predicted values of the train dataset
        for x_i in x:
            r.append(self.func(x_i, gamma, kernel))
        return np.array(r) #returning the predictions as an array
        
    #predictions for gaussian kernel
    def predict_gaussian(self, x, kernel = "gaussian", gamma = None):
        r = [] #list that stores the predicted values of the train dataset
        for x_i in x:
            r.append(self.func(x_i, gamma, kernel))
        return np.array(r) #returning the predictions as an array
    
    #training the dataset
    def train(self, x, y, C, kernel = "dot", gamma = None):

        #defining the constraints corresponding to the 3 cases
        constraints = ({'type': 'ineq',
                        'fun': lambda a: a
                       },
                       {
                         'type': 'ineq',
                        'fun': lambda a: C - a  
                       },
                      {
                         'type': 'eq',
                        'fun': lambda a: np.dot(a, y)  
                       })
        
        temp = len(x) #length of the dataset
        res = scipy.optimize.minimize(self.inner_loop, x0=np.zeros(shape=(temp,)), args=(x, y), method='SLSQP', constraints=constraints, tol=0.01)

        self.w_asterisk = np.zeros_like(x[0]) #intilizing weight with all zeros
        for i in range(len(x)):
            self.w_asterisk += res['x'][i]*y[i]*x[i]

        self.b_asterisk = 0 #initializing bias with 0
        if kernel == "dot":
            for j in range(len(x)):
                self.b_asterisk += y[j] - np.dot(self.w_asterisk, x[j])
        
        if kernel == "gaussian":
            for j in range(len(x)):
                self.b_asterisk += y[j] - gaussian_kernel(self.w_asterisk, x[j], gamma)
        
        self.b_asterisk /= len(x)

        th = 1e-5 #defining the threshold
        for i, a in enumerate(res['x']):
            if a > th:
                self.support.append(x[i])
                
    def inner_loop(self, a, x, y, kernel = "dot"):
        y_res = y * np.ones((len(y), len(y)))
        a_res = a * np.ones((len(a), len(a)))

        if kernel == "dot":
            x_values = np.matmul(x, x.T)
            
        if kernel == "gaussian":
            temp_val_1 = np.matmul(pow(x, 2), np.ones_like(x.T))
            temp_val_2 = np.matmul(np.ones_like(x), pow(x.T, 2))
            x_values = temp_val_1 - 2*np.matmul(x,x.T) + temp_val_2 
            x_values = np.exp(-( x_values / gamma))

        values = (y_res * y_res.T) * (a_res * a_res.T) * x_values
        return 0.5 * np.sum(values) - np.sum(a)


# In[21]:


#basic dual SVM implementation for different C values
C_vals = [100/873, 500/873, 700/873]

for c in C_vals:
    d_svm = Dual_SVM(x_train, y_train, c)
    print("For C = ", c)
    d_svm.train(x_train, y_train, c)
    print("learned weights: ", d_svm.w_asterisk)
    print("learned bias: ", d_svm.b_asterisk)
    print("training accuracy: ", np.mean(y_train == d_svm.predict_dot(x_train)))
    print("testing accuracy: ", np.mean(y_test == d_svm.predict_dot(x_test)))


# In[22]:


# dual SVM implementation with gaussian kernel for different C and gamma values
C_vals = [100/873, 500/873, 700/873]
gammas = [0.1, 0.5, 1, 5, 100]
for c in C_vals:
    for gamma in gammas:
        print("For C = ", c)
        print("For gamma = ", gamma)
        d_svm.train(x_train, y_train, c, kernel='gaussian', gamma=gamma)
        print("learned weights: ", d_svm.w_asterisk)
        print("learned bias: ", d_svm.b_asterisk)
        print("training accuracy: ", np.mean(y_train == d_svm.predict_gaussian(x_train, kernel='gaussian', gamma=gamma)))
        print("testing accuracy: ", np.mean(y_test == d_svm.predict_gaussian(x_test, kernel='gaussian', gamma=gamma)))



# In[23]:


#finding the number of support vectors for different gamma intervals when C = 500/873
C_vals = 500/873
gammas = [0.1, 0.5, 1, 5, 100]
sv = []
for gamma in gammas:
    print("For C = ", c)
    print("For gamma = ", gamma)
    d_svm.train(x_train, y_train, c, kernel='gaussian', gamma=gamma)
    print("learned weights: ", d_svm.w_asterisk)
    print("learned bias: ", d_svm.b_asterisk)
    print("number of support vectors: ", len(d_svm.support))
    sv.append(d_svm.support)
    print("training accuracy: ", np.mean(y_train == d_svm.predict_gaussian(x_train, kernel='gaussian', gamma=gamma)))
    print("testing accuracy: ", np.mean(y_test == d_svm.predict_gaussian(x_test, kernel='gaussian', gamma=gamma)))



# In[24]:


for i in range(4):
    count = 0
    for v in np.array(sv[i]):
        if v in np.array(sv[i+1]):
            count += 1
    print("overlap from gamma = ", gammas[i], " to ", gammas[i+1], " : ", count)

