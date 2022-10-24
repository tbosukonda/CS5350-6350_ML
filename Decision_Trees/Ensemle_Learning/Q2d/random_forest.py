#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import multiprocessing as mp


# In[2]:


train = pd.read_csv(r'train.csv', names = ['age', 'job', 'marital', 'education',
                                                                            'default', 'balance', 'housing', 'loan', 'contact', 
                                                                            'day', 'month', 'duration', 'campaign', 'pdays',
                                                                            'previous', 'poutcome', 'y'])
#train.head()


# In[3]:


test = pd.read_csv(r'test.csv', names = ['age', 'job', 'marital', 'education',
                                                                            'default', 'balance', 'housing', 'loan', 'contact', 
                                                                            'day', 'month', 'duration', 'campaign', 'pdays',
                                                                            'previous', 'poutcome', 'y'])
#test.head()


# In[4]:


#to find the median of the attribute/column
def threshold(df, col):
    l = len(df.iloc[:,col])
    if l%2 == 0:
        median = (df.iloc[l//2,col] + df.iloc[l//2-1,col])/2 
    else:
        median = df.iloc[l//2,0]
    return median


# In[5]:


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


# In[9]:


#converting train dataset
convert(train, 0, age_threshold)
convert(train, 5, balance_threshold)
convert(train, 9, day_threshold)
convert(train, 11, duration_threshold)
convert(train, 12, campaign_threshold)
convert(train, 13, pdays_threshold)
convert(train, 14, previous_threshold)


# In[8]:


#converting test dataset
convert(test, 0, age_threshold)
convert(test, 5, balance_threshold)
convert(test, 9, day_threshold)
convert(test, 11, duration_threshold)
convert(test, 12, campaign_threshold)
convert(test, 13, pdays_threshold)
convert(test, 14, previous_threshold)


# In[10]:


#converting intomatrix form
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
    


# In[ ]:


def get_column_name(idx):
    column_idx = {1:'age', 2:'job', 3:'marital', 4:'education',5:'default', 6:'balance', 7:'housing', 8:'loan',
                  9:'contact', 10:'day', 11:'month', 12:'duration', 13:'campaign', 14:'pdays',15:'previous', 
                  16:'poutcome', 17:'y'}
    column_idx = dict(column_idx)
    return column_idx[idx]


# In[63]:


import random
class RandomForest:
    
    def __init__(self,num_of_bagged=5, num_of_features = 2):
        # Initialised with number of bagged trees
        self.num_of_bagged=num_of_bagged
        self.num_of_features = num_of_features
        self.models = []
        
    def fit(self,df, data, num_samples): 
        indexs=np.random.choice(len(data),size = num_samples)
        f_indexs = []
        idx = random.randint(1,15)
        for i in range(self.num_of_features):
            while idx in f_indexs:
                idx = random.randint(1,15)
            f_indexs.append(idx)    
             
        f_indexs = np.append(f_indexs, 17)
        
        sample_subspace = []
        for i in indexs:
            sample = []
            col = []
            for j in f_indexs:
                col_j = get_column_name(j)
                col.append(col_j)
                sample.append(data[i][col_j])
            sample_subspace.append(dict(zip(col, sample)))
        tree = build_tree(entropy, sample_subspace)
        return tree
            
    def make_trees(self, df, n_trees, sample_space_size, num_workers = 4):
        
        mult_data = [df] * n_trees
        mult_samp = [sample_space_size] * n_trees

        with mp.Pool(num_workers) as pool:
            self.models = pool.starmap(self.fit, zip(df, mult_data, mult_samp))
            
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
    
    def acc(self,y_true,y_pred):
        count = 0
        for i in range(len(y_pred)):
            if y_true[i] == y_pred[i]:
                count = count+1
        return count/len(y_pred)


# In[70]:


train_acc_2 = []
test_acc_2 = []
train_acc_4 = []
test_acc_4 = []
train_acc_6 = []
test_acc_6 = []

x = [1, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
for i in x:
    for j in [2, 4, 6]:
        print("Random Forest for iteration : ", i, "with feature subset equal to : ", j)
        rf_model = RandomForest(i, j)
        rf_model.make_trees(train_df, i, 1000)
        print("---> making predictions")
        train_y_pred = rf_model.predict(x_train)
        test_y_pred = rf_model.predict(x_test)
        
        if j==2:
            train_acc_2.append(rf_model.acc(list(y_train_df["y"]), train_y_pred))
            test_acc_2.append(rf_model.acc(list(y_test_df["y"]), test_y_pred))
            
        elif j==4:
            train_acc_4.append(rf_model.acc(list(y_train_df["y"]), train_y_pred))
            test_acc_4.append(rf_model.acc(list(y_test_df["y"]), test_y_pred))
        
        elif j==6:
            train_acc_6.append(rf_model.acc(list(y_train_df["y"]), train_y_pred))
            test_acc_6.append(rf_model.acc(list(y_test_df["y"]), test_y_pred))


# In[71]:


import matplotlib.pyplot as plt
y_axis_train2 = [1-i for i in train_acc_2]
y_axis_train4 = [1-i for i in train_acc_4]
y_axis_train6 = [1-i for i in train_acc_6]
y_axis_test2 = [1-i for i in test_acc_2]
y_axis_test4 = [1-i for i in test_acc_4]
y_axis_test6 = [1-i for i in test_acc_6]


# In[74]:


plt.plot(x, y_axis_train2, color = 'g', label='train')
plt.plot(x, y_axis_test2, color = 'r', label='test')

plt.xlabel("No. of trees")
plt.ylabel("Error with feature size = 2")
plt.title("Variation of train and test error with number of trees")
plt.show()


# In[81]:


plt.plot(x, y_axis_train4, color = 'g', label='train')
plt.plot(x, y_axis_test4, color = 'r', label='test')

plt.xlabel("No. of trees")
plt.ylabel("Error with feature size = 4")
plt.title("Variation of train and test error with number of trees")
plt.show()


# In[82]:


plt.plot(x, y_axis_train6, color = 'g', label='train')
plt.plot(x, y_axis_test6, color = 'r', label='test')

plt.xlabel("No. of trees")
plt.ylabel("Error with feature size = 2")
plt.title("Variation of train and test error with number of trees")
plt.show()


# In[ ]:




