#!/usr/bin/env python
# coding: utf-8

# Defining the ID3 algorithm that computes the information gain using entropy, gini index, and majority error

# In[1]:


#finding the entropy of the the attribute
def entropy(df):
    entropy = 0
    counts = label_counts(df)
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(df))
        entropy = entropy + (-1) * prob_of_lbl * np.log2(prob_of_lbl)
    return entropy


# In[2]:


#finding the gini index of the the attribute
def gini(df):
    counts = label_counts(df)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(df))
        impurity -= prob_of_lbl**2
    return impurity


# In[3]:


#finding the majority error of the attribute
def ME(df):
    ME = 0
    me_vals = []
    counts = label_counts(df)
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(df))
        me_vals.append(prob_of_lbl)
    return min(me_vals)


# In[4]:


#finding information gain
def info_gain(model, left, right, current_uncertainty):    
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * model(left) - (1 - p) * model(right)


# Constructing decision tree using above metrics

# In[5]:


import pandas as pd #for handling dataframes
import numpy as np #for numerical operations


# In[6]:


#importing train dataset and setting the column names
df_train = pd.read_csv(r'train.csv', 
                 names = ["buying","maint","doors","persons","lug_boot","safety","label"])
#df_train.head()


# In[7]:


#checking the description of all the attributes
#df_train.info()


# In[8]:


#defining the various columns of the dataset
header = ["buying","maint","doors","persons","lug_boot","safety","label"]


# In[9]:


#finding the unique values an attribute can take
def unique_vals(df, col):
    return set(df.iloc[:,col])

#example
#print("Finding the unique values the attribute door can take : ",unique_vals(df_train, 2))


# In[11]:


#finding the count of the unique values of the label attribute
def label_counts(df):
    counts = df.iloc[:,6].value_counts()
    return counts.to_dict()

#res = label_counts(df_train)
#print(res)


# In[12]:


#defining a condition that is used to split the dataset
class Condition:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):# Compare the feature value in an example to the conidition
        val = example[self.column]
        return val == self.value

    def __repr__(self): #to display the condition
        condition = "=="
        return "Is %s %s %s?" % (header[self.column], condition, str(self.value))


# In[13]:


#split the dataset by checking each row and appending it to either a true set or false set
def partition(df, question):
    true_rows, false_rows = [], []
    for i in range(len(df)):
        row = df.iloc[i, :]
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    
    #converting the rows of list type into DataFrame type
    true_rows = pd.DataFrame(true_rows, columns = ["buying","maint","doors","persons","lug_boot","safety","label"])
    false_rows = pd.DataFrame(false_rows, columns = ["buying","maint","doors","persons","lug_boot","safety","label"])
    return true_rows, false_rows    
    


# In[14]:


#Finding the feature or value of the feature that best splits the dataset
def find_best_split(model, rows):
    #print("in find_best_split : ",type(rows))
    best_gain = 0  # maintaining best information gain
    best_question = None  # maintaining the condition of the corresponding information gain
    current_uncertainty = model(rows)
    n_features = len(rows.columns) - 1  # number of columns

    for col in range(n_features):  # for each feature
        values = unique_vals(rows, col)# unique values in the column          

        for val in values:  # for each value

            question = Condition(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)
            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(model, true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


# In[16]:


#defining the leaf node by setting it as the max value of all the values the attribute of a feature can take
class Leaf:
    def __init__(self, df):
        self.predictions = self.max_leaf(df)
        
    def max_leaf(self, df):
        res = label_counts(df)
        max_value = max(res, key = res.get)
        return {max_value:res[max_value]}


# In[17]:


#defining the decision node that splits the node
class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


# In[18]:


#constructing the tree
def build_tree(model, rows, max_depth, c_depth):
    
    #finding the attribute that best splits the datset and finding its information gain
    gain, question = find_best_split(model, rows)

    #leaf node case
    if gain == 0 or c_depth>=max_depth:
        return Leaf(rows)
    
    #partition the datset 
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(model, true_rows, max_depth, c_depth+1)

    # Recursively build the false branch.
    false_branch = build_tree(model, false_rows, max_depth, c_depth+1)

    #returning the decision node
    return Decision_Node(question, true_branch, false_branch)


# In[19]:


def print_tree(node, spacing=""):

    # Leaf node case
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # calling recursively on left part until leaf node is obtained
    print (spacing + '-> True:')
    print_tree(node.true_branch, spacing + "  ")

    # calling recursively on left part until leaf node is obtained
    print (spacing + '-> False:')
    print_tree(node.false_branch, spacing + "  ")


# In[20]:


#method to make prediction for test dataset
def classify(row, node):

    # leaf node case
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


# In[21]:


print("constructing decision tree using gini index with max depth as 6")
my_tree_1 = build_tree(gini, df_train, 6, 0) 


# In[22]:


#print_tree(my_tree_1)


# In[23]:


df_test = pd.read_csv(r'test.csv', 
                 names = ["buying","maint","doors","persons","lug_boot","safety","label"])


# In[24]:


#make predictions and find error on test data
def calc_error(pred_tree):
    cnt = 0
    for i in range(len(df_test)):
        actual = df_test.iloc[i,6] #label value
        predicted = list(classify(df_test.iloc[i,:], pred_tree)) #predicted label value
        if actual != predicted[0]:
            cnt = cnt+1
    error = (cnt/len(df_test))*100
    return error
    


# In[25]:


error1 = calc_error(my_tree_1) #error for gini index with depth 6


# In[26]:


print("constructing decision tree using gini index with max depth as 3")
my_tree_2 = build_tree(gini, df_train, 3, 0) 


# In[27]:


#print_tree(my_tree_2)


# In[28]:


error2 = calc_error(my_tree_2) #error for gini index with depth 3


# In[29]:


print("constructing decision tree using entropy with max depth as 3")
my_tree_3 = build_tree(entropy, df_train, 3, 0) 
#print_tree(my_tree_3)


# In[30]:


error3 = calc_error(my_tree_3) #error for entropy with depth as 3


# In[31]:


print("constructing decision tree using entropy with max depth as 6")
my_tree_4 = build_tree(entropy, df_train, 6, 0) 
#print_tree(my_tree_4)


# In[32]:


error4 = calc_error(my_tree_4) #error for entropy with depth as 6


# In[33]:


print("Constructing decision tree using Majority Error with max depth as 3")
my_tree_5 = build_tree(ME, df_train, 3, 0) 
#print_tree(my_tree_5)


# In[34]:


error5 = calc_error(my_tree_5) #error ME with max depth as 3


# In[35]:


print("constructing decision tree using Majority Error with max depth as 6")
my_tree_6 = build_tree(ME, df_train, 6, 0) 
#print_tree(my_tree_6)


# In[36]:


error6 = calc_error(my_tree_6) #error ME with max depth as 6


# In[38]:


#train data prediction and error calculation
def train_calc_error(pred_tree):
    cnt = 0
    for i in range(len(df_train)):
        actual = df_train.iloc[i,6] #label value
        predicted = list(classify(df_train.iloc[i,:], pred_tree)) #predicted label value
        if actual != predicted[0]:
            cnt = cnt+1
    error = (cnt/len(df_train))*100
    return error
    


# In[39]:


error7 = train_calc_error(my_tree_1) #train error for gini index with depth 6
error8 = train_calc_error(my_tree_2) #train error for gini index with depth 3
error9 = train_calc_error(my_tree_3) #train error for entropy with depth 3
error10 = train_calc_error(my_tree_4) #train error for entropy with depth 6
error11 = train_calc_error(my_tree_5) #train error for ME with depth 3
error12 = train_calc_error(my_tree_6) #train error for ME with depth 6


# In[40]:


print("Train data error")
print("\t\tdepth-3\t\t\t\t depth-6")
print("gini index\t", error8, "\t\t", error7)
print("entropy\t\t", error9, "\t\t", error10)
print("ME\t\t", error11, "\t\t", error12)


# In[37]:


print("Test data error")
print("\t\tdepth-3\t\t\t\t depth-6")
print("gini index\t", error2, "\t\t", error1)
print("entropy\t\t", error3, "\t\t", error4)
print("ME\t\t", error5, "\t\t", error6)

