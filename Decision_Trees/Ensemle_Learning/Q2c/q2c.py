#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import multiprocessing as mp
header = ['age', 'job', 'marital', 'education','default', 'balance', 'housing', 'loan', 'contact', 
            'day', 'month', 'duration', 'campaign', 'pdays','previous', 'poutcome', 'y']

#finding the entropy of the the attribute
def entropy(df):
    entropy = 0
    labels = []
    for i in range(len(df)):
        labels.append(df[i]["y"])
    counts = dict(zip(labels, [labels.count(i) for i in labels]))
    if 1 not in counts.keys():
        counts[1] = 0
    if 0 not in counts.keys():
        counts[0] = 0
        
    p_count = counts[1]/len(df)
    n_count = counts[0]/len(df)
    
    #computing logarithmic value
    if p_count == 0:
        p_log = 0
    else:
        p_log = np.log2(p_count)
    
    if n_count == 0:
        n_log = 0
    else:
        n_log = np.log2(n_count)
    
    #computing entropy
    entropy = entropy + (((-1)* p_count * p_log) + ((-1)* n_count * n_log))
    return entropy


#finding information gain
def info_gain(model, left, right, current_uncertainty):    
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - (p * model(left)) - ((1 - p) * model(right))

#finding the unique values an attribute can take
def unique_vals(df, col):
    labels = []
    for i in range(len(df)):
        labels.append(df[i][col])
    return set(labels)

#finding the count of the unique values of the label attribute
def label_counts(df):
    labels = []
    for i in range(len(df)):
        labels.append(df[i]["y"])
    counts = dict(zip(labels, [labels.count(i) for i in labels]))
    if 1 not in counts.keys():
        counts[1] = 0
    if 0 not in counts.keys():
        counts[0] = 0
    return counts

#defining a condition that is used to split the dataset
class Condition:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):# Compare the feature value in an example to the conidition
        val = example[self.column]
        return val == self.value

    def __repr__(self): #print the condition
        condition = "=="
        return "Is %s %s %s?" % (self.column, condition, str(self.value))
    
#split the dataset by checking each row and appending it to either a true set or false set
def partition(df, question):
    true_rows, false_rows = [], []
    for i in range(len(df)):
        row = df[i]
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
            
    return true_rows, false_rows    

#Finding the feature or value of the feature that best splits the dataset
def find_best_split(model, rows):
    best_gain = 0  # maintaining best information gain
    best_question = None  # maintaining the condition of the corresponding information gain
    current_uncertainty = model(rows)
    features = list(rows[0].keys())
    n_features = len(features)
    
    for col in features[0:n_features-1]:  # for each feature
        values = unique_vals(rows, col)# unique values in the column          

        for val in values:  # for each value

            question = Condition(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)
            
            # Skip this split if it doesn't divide the dataset
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(model, true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

#defining the leaf node by setting it as the max value of all the values the attribute of a feature can take
class Leaf:
    def __init__(self, df):
        self.predictions = self.max_leaf(df)
        
    def max_leaf(self, df):
        res = label_counts(df)
        max_value = max(res, key = res.get)
        return {max_value:res[max_value]}
    
#defining the decision node that splits the node
class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        
#constructing the tree
def build_tree(model, rows):
    
    if len(rows)>0:
        #finding the attribute that best splits the datset and finding its information gain
        gain, question = find_best_split(model, rows)

        #leaf node case
        if gain == 0:
            return Leaf(rows)

        #partition the datset 
        true_rows, false_rows = partition(rows, question)

        # Recursively build the true branch.
        true_branch = build_tree(model, true_rows)

        # Recursively build the false branch.
        false_branch = build_tree(model, false_rows)

    #returning the decision node
    return Decision_Node(question, true_branch, false_branch)

#method to make prediction for test dataset
def classify(row, node):

    # leaf node case
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)
    
def print_tree(node, spacing=""):

    # Leaf node case
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")
    


# In[3]:


train = pd.read_csv(r'train.csv', names = ['age', 'job', 'marital', 'education',
                                                                            'default', 'balance', 'housing', 'loan', 'contact', 
                                                                            'day', 'month', 'duration', 'campaign', 'pdays',
                                                                            'previous', 'poutcome', 'y'])
#train.head()


# In[4]:


test = pd.read_csv(r'test.csv', names = ['age', 'job', 'marital', 'education',
                                                                            'default', 'balance', 'housing', 'loan', 'contact', 
                                                                            'day', 'month', 'duration', 'campaign', 'pdays',
                                                                            'previous', 'poutcome', 'y'])
#test.head()


# In[5]:


#to find the median of the attribute/column
def threshold(df, col):
    l = len(df.iloc[:,col])
    if l%2 == 0:
        median = (df.iloc[l//2,col] + df.iloc[l//2-1,col])/2 
    else:
        median = df.iloc[l//2,0]
    return median


# In[6]:


#column with numerical values: age, balance, day, duration, campaign, pdays, previous
#corresponding column numbers: 0, 5, 9, 11, 12, 13, 14
age_threshold = threshold(train, 0)
balance_threshold = threshold(train, 5)
day_threshold = threshold(train, 9)
duration_threshold = threshold(train, 11)
campaign_threshold = threshold(train, 12)
pdays_threshold = threshold(train, 13)
previous_threshold = threshold(train, 14)


# In[7]:


#converting numerical values into binary categories
#if value>threshold, then 1, else 0
def convert(df, col, threshold):
    for i in range(len(df.iloc[:,col])):
        if df.iloc[i,col]>threshold:
            df.iloc[i,col] = 1
        else:
            df.iloc[i,col] = 0


# In[8]:


#converting train dataset
convert(train, 0, age_threshold)
convert(train, 5, balance_threshold)
convert(train, 9, day_threshold)
convert(train, 11, duration_threshold)
convert(train, 12, campaign_threshold)
convert(train, 13, pdays_threshold)
convert(train, 14, previous_threshold)


# In[9]:


#converting test dataset
convert(test, 0, age_threshold)
convert(test, 5, balance_threshold)
convert(test, 9, day_threshold)
convert(test, 11, duration_threshold)
convert(test, 12, campaign_threshold)
convert(test, 13, pdays_threshold)
convert(test, 14, previous_threshold)


# In[10]:


train["y"] = train["y"].replace(to_replace = "yes", value = 1)
train["y"] = train["y"].replace(to_replace = "no", value = 0)
x_train_df = train.iloc[:,:16]
y_train_df = pd.DataFrame(train.iloc[:,16], columns = ["y"])
x_train = x_train_df.to_dict('records')
y_train = y_train_df.to_dict('records')


# In[11]:


test["y"] = test["y"].replace(to_replace = "yes", value = 1)
test["y"] = test["y"].replace(to_replace = "no", value = 0)
x_test_df = test.iloc[:,:16]
y_test_df = pd.DataFrame(test.iloc[:,16], columns = ["y"])
x_test = x_test_df.to_dict('records')
y_test = y_test_df.to_dict('records')


# In[12]:


print("Converting numerical values in TRAIN DATA into binary categories")
train.head()


# In[13]:


print("Converting numerical values in TEST DATA into binary categories")
test.head()


# In[14]:


train_df = train.to_dict('records')
test_df = test.to_dict('records')


# In[46]:


class Bagging:
    
    def __init__(self,num_of_bagged=5):
        # Initialised with number of bagged trees
        self.num_of_bagged=num_of_bagged
        self.models = []
        
    def fit(self,df, data, num_samples): 
        # to store the models
        self.models=[]
        indexs=np.random.choice(len(data),size = num_samples)
        sample_subspace = []
        for i in indexs:
            sample_subspace.append(data[i])
        tree = build_tree(entropy, sample_subspace)
        return tree
            
            
    def make_trees(self, df, n_trees, sample_space_size, num_workers = 4):
        
        mult_data = [df] * n_trees
        mult_samp = [sample_space_size] * n_trees

        with mp.Pool(num_workers) as pool:
            self.models = pool.starmap(self.fit, zip(df, mult_data, mult_samp))
            
    def getFirstTree(self):
        return self.models[0]
            
    def predict(self,X):
        pred = np.zeros(len(X))
        # predicting with each stored models
        for model in self.models:
            y_pred = []
            
            for i in range(len(X)):
                predicted = list(classify(X[i], model))
                y_pred.append(predicted[0])
            pred = pred + y_pred
            
        norm = self.num_of_bagged * np.ones(len(X))
        res = np.round(pred/norm) # Model averaging
        
        return list(map(int, res))
    
    def b_var_predict(self, X, models):
        pred = np.zeros(len(X))
        # predicting with each stored models
        for model in models:
            y_pred = []
            for i in range(len(X)):
                predicted = list(classify(X[i], model))
                y_pred.append(predicted[0])
            pred = pred + y_pred
        norm = 500 * np.ones(len(X))
        res = np.round(pred/norm) # Model averaging
        return list(map(int, res))
    
    def acc(self,y_true,y_pred):
        count = 0
        for i in range(len(y_pred)):
            if y_true[i] == y_pred[i]:
                count = count+1
        return count/len(y_pred)


# In[81]:


import random
bagged_trees = []
first_tree = []
train_acc = []
test_acc = []


for i in range(1,100):
    indexs = []
    data = []
    print("iteration : ", i)
    idx = random.randint(1,len(train_df)-1)
    for j in range(1000):
        while idx in indexs:
            idx = random.randint(1,len(train_df)-1)
        indexs.append(idx) 
    for k in indexs:
        data.append(train_df[k])
    
    #data is the new dataset
    print("Constructing 500 trees")
    bagging_model = Bagging(500) #500 trees
    bagging_model.make_trees(data, 500, 1000) # dataset, num. of trees, sample_space
    bagged_trees.append(bagging_model)
    first_tree.append(bagging_model.getFirstTree())
    


# In[82]:


single_tree_bias, bagged_bias, single_tree_variance, bagged_variance = [],[],[],[]
single_y_pred = bagging_model.b_var_predict(x_test, first_tree)
gtl = list(y_test_df['y'])
single_tree_bias.append((gtl - np.mean(single_y_pred)) ** 2)
single_tree_variance.append(np.std(single_y_pred) ** 2)


# In[83]:


print(np.mean(single_tree_bias))
print(np.mean(single_tree_variance))


# In[84]:


bagged_y_pred = bagging_model.predict(x_test)
bagged_bias.append((gtl - np.mean(bagged_y_pred)) ** 2)


# In[85]:


bagged_variance.append(np.std(bagged_y_pred) ** 2)
print(np.mean(bagged_bias))
print(np.mean(bagged_variance))


# In[86]:


print("GSE of single tree: ", np.mean(single_tree_bias) + np.mean(single_tree_variance))
print("GSE of bagged trees: ", np.mean(bagged_bias) + np.mean(bagged_variance))


# In[ ]:




