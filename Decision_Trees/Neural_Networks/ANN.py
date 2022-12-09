#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


train_df = pd.read_csv("train.csv") 
#train_df.head()
test_df = pd.read_csv("test.csv") 
test_df.head()


# In[3]:


train_df = np.array(train_df)
test_df = np.array(test_df)


# In[11]:


import warnings
warnings.filterwarnings("ignore")
class ANN:
    def __init__(self, width, x_train, y_train, question):

        self.width = width
        self.x = x_train 
        self.y = y_train 
        
        #defining a list of 1's (bias values) of the length of the train datset
        bias = np.transpose(np.array([[1] * x_train.shape[0]])) 
        #adding the bias term to all the rows in the training data
        self.x = np.concatenate((bias, self.x), axis = 1)
        #initialising weights randomly
        if question == 1:
            self.weights = initialize_weights(width, x_train.shape[1]+1)
        else:
            self.weights = initialize_zero_weights(width, x_train.shape[1]+1)
        self.nodes = init_nodes_arr(width, x_train.shape[1]+1)
        
    def sgd(self, gamma, d, epochs):
        
        indices = np.arange(0, len(self.x))
        
        for t in range(epochs):
            #shuffling the data
            np.random.shuffle(indices)
            for i in indices:
                x_i = self.x[i]
                y_i = self.y[i]
                
                #calculating the learning rate
                temp = (gamma/d)*t
                lr = gamma / (1.0 + temp)
                
                gradient = self.backpropagation(x_i, y_i)
                update_weights(self.weights, lr, gradient)
                
    def backpropagation(self, x_i, y):
        cache = [np.zeros(self.width), np.zeros(self.width), np.zeros(1)] #corresponding to the hidden layer 1, hidden layer 2, and output layer
        gradient = [np.zeros((self.width, len(x_i))), np.zeros((self.width, self.width)), np.zeros((1, self.width))] #initialize_gradient(self.width, len(x_i))
        
        #computing cache of output layer
        prediction = self.predict(x_i)
        cache[2][0] = cross_entropy_deriv(y, (1.0 / (1.0 + np.exp(-prediction))) * deriv_sigmoid_activation_function(prediction))
        #computing gradient values
        for n in range(self.width):
            gradient[2][0][n] = cache[2][0] * self.nodes[2][n]

        # back propogate 2nd hidden layer followed by 1st hidden layer to update the cache and gradient values
        self.update_layer(cache, gradient, 2, self.width, self.width)
        self.update_layer(cache, gradient, 1, len(x_i), self.width)

        return gradient
    
    def update_layer(self, cache, gradient, layer, nodes_prev_layer, nodes_curr_layer):

        #Iterating over all nodes except bias
        for node in range(1, nodes_prev_layer):
            temp = deriv_sigmoid_activation_function(self.nodes[layer][node])
            sigmoid_deriv = deriv_sigmoid_activation_function(self.nodes[layer][node])

            #outgoing edges from node
            edge_out = self.weights[layer][:, node]
            deriv = np.dot(cache[layer], edge_out)
            cache[layer - 1][node] = deriv * sigmoid_deriv

            #incoming edge to node
            for edge in range(nodes_prev_layer):
                temp = deriv * sigmoid_deriv
                gradient[layer - 1][node][edge] = temp * self.nodes[layer - 1][edge]
                
                
    def predict(self, x_i):
        self.nodes[0] = x_i
        self.predict_layer(1, len(x_i)) #forward propagation for hidden layer 1
        self.predict_layer(2, self.width) #forward propagation for hiddem layer 2
        
        return np.dot(self.weights[2][0], self.nodes[2])

    def predict_layer(self, layer_idx, num_nodes):
        
        for node_idx in range(1, num_nodes):
            
            #y_j = sum over all nodes of x_i^j-1 * w_i^j-1,j
            wt = self.weights[layer_idx - 1][node_idx]
            node_val = self.nodes[layer_idx - 1]
            total_sum = np.dot(wt, node_val) #wt.dot(node_val)
            
            #applying activation function over the total sum
            self.nodes[layer_idx][node_idx] = (1.0 / (1.0 + np.exp(-total_sum))) 


# In[12]:


def initialize_weights(width, d):
    weights = []
    weights.append(np.random.rand(width, d)) #layer 1
    weights.append(np.random.rand(width, width)) #layer 2
    weights.append(np.random.rand(1, width)) #output

    return weights

def initialize_zero_weights(width, d):
    weights = []
    weights.append(np.zeros((width, d))) #layer 1
    weights.append(np.zeros((width, width))) #layer 2
    weights.append(np.zeros((1, width))) #output

    return weights

def init_nodes_arr(width, d):
    nodes = []
    nodes.append(np.zeros(d)) # input layer
    nodes.append(np.zeros(width)) # hidden layer 1
    nodes.append(np.zeros(width)) # hidden layer 2
    nodes[1][0] = 1 # hidden layer 2 bias
    nodes[2][0] = 1 # hidden layer 2 bias

    return nodes


def deriv_sigmoid_activation_function(z):
    return z * (1.0 - z)


def update_weights(weights, lr, gradient):
    for layer in range(len(weights)):
        weights[layer] = weights[layer] - lr * gradient[layer]

def cross_entropy_deriv(y, y_target):
    if y==1:
        return y_target-1
    return y_target

def calc_error(NN, test_df):
    X = test_df[:, : test_df.shape[1]-1]
    y = test_df[:, test_df.shape[1]-1]
    size_df = len(test_df)
    
    #defining a list of 1's (bias values) of the length of the train datset
    bias = np.transpose(np.array([[1] * test_df.shape[0]]))
    #adding the bias term to all the rows in the training data
    X = np.concatenate((bias, X), axis = 1)
    
    #counter for number of misclassifications
    cnt = 0

    for i in range(size_df):
        predict = NN.predict(X[i])
        prob = (1.0 / (1.0 + np.exp(-predict)))
        
        if prob<0.5:
            pred = 0
        else:
            pred = 1
            
        if y[i] != pred:
            cnt += 1

    return cnt / size_df


# In[13]:


X = train_df[:, : train_df.shape[1]-1] #independent features of train dataset
y = train_df[:, train_df.shape[1]-1]   #dependent features of train dataset
widths = [5, 10, 25, 50, 100]
for w in widths:
    NN = ANN(w, X, y, 1)
    NN.sgd(0.1, 0.01, 10) #parameters: gamma, d, epochs
    print("Width: ", w)
    print("Training Error: ", calc_error(NN, train_df))
    print("Test Error: ", calc_error(NN, test_df))


# In[14]:


for w in widths:
    NN_b = ANN(w, X, y, 2)
    NN_b.sgd(0.1, 0.01, 10) #parameters: gamma, d, epochs
    print("Width: ", w)
    print("Training Error: ", calc_error(NN_b, train_df))
    print("Test Error: ", calc_error(NN_b, test_df))


# In[ ]:




