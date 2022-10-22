#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


train_df = pd.read_csv(r"concrete/train.csv", names = ["Cement", "Slag", "Fly_ash", "Water", "SP",
                                                       "Coarse_Aggr", "Fine_Aggr", "Output"])


# In[4]:


#train_df.head()


# In[12]:


#train_df.info()


# In[5]:


test_df = pd.read_csv(r"concrete/test.csv", names = ["Cement", "Slag", "Fly_ash", "Water", "SP",
                                                       "Coarse_Aggr", "Fine_Aggr", "Output"])


# In[6]:


x_train = train_df.iloc[:,:7]
#x_train.head()


# In[7]:


y_train = train_df["Output"]
#y_train.head()


# In[13]:


#coverting the dataframe into an ndarray
X = x_train.to_numpy()
Y = y_train.to_numpy()


# In[26]:


#X^T.X
A = np.dot(X.transpose(),X)


# In[27]:


#X^T.Y
B = np.dot(X.transpose(), Y)


# In[28]:


#(X^T.X)^-1
A_inverse = np.linalg.inv(A)


# In[29]:


#((X^T.X)^-1).(X^T.Y)
opt_w = np.dot(A_inverse, B)


# In[30]:


print("The optial weight vector is : ", opt_w)


# In[ ]:




