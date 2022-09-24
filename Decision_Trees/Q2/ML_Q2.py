#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#loading train dataset
train_df = pd.read_csv(r'train.csv', names = ['age', 'job', 'marital', 'education',
                                                                            'default', 'balance', 'housing', 'loan', 'contact', 
                                                                            'day', 'month', 'duration', 'campaign', 'pdays',
                                                                            'previous', 'poutcome', 'y'])
#train_df.head()


# In[3]:


#loading test datset
test_df = pd.read_csv(r'test.csv', names = ['age', 'job', 'marital', 'education',
                                                                            'default', 'balance', 'housing', 'loan', 'contact', 
                                                                            'day', 'month', 'duration', 'campaign', 'pdays',
                                                                            'previous', 'poutcome', 'y'])
#test_df.head()


# In[5]:


#description of dataset
#train_df.info()


# In[6]:


#finding the entropy of the the attribute
def entropy(df):
    entropy = 0
    counts = label_counts(df)
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(df))
        entropy = entropy + (-1) * prob_of_lbl * np.log2(prob_of_lbl)
    return entropy


# In[7]:


#finding the gini index of the the attribute
def gini(df):
    counts = label_counts(df)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(df))
        impurity -= prob_of_lbl**2
    return impurity


# In[8]:


#finding the majority error of the attribute
def ME(df):
    ME = 0
    me_vals = []
    counts = label_counts(df)
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(df))
        me_vals.append(prob_of_lbl)
    return min(me_vals)


# In[9]:


#finding information gain
def info_gain(model, left, right, current_uncertainty):    
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * model(left) - (1 - p) * model(right)


# In[10]:


header = ['age', 'job', 'marital', 'education','default', 'balance', 'housing', 'loan', 'contact', 
            'day', 'month', 'duration', 'campaign', 'pdays','previous', 'poutcome', 'y']


# Converting the numberical columns into binary categories by using threshold. Threshold is calculated by finding the median of the column. If the column values are greater than threshold, then value = 1, else value = 0

# In[11]:


#to find the median of the attribute/column
def threshold(df, col):
    l = len(df.iloc[:,col])
    if l%2 == 0:
        median = (train_df.iloc[l//2,col] + train_df.iloc[l//2-1,col])/2 
    else:
        median = train_df.iloc[l//2,0]
    return median


# In[12]:


#column with numerical values: age, balance, day, duration, campaign, pdays, previous
#corresponding column numbers: 0, 5, 9, 11, 12, 13, 14
age_threshold = threshold(train_df, 0)
balance_threshold = threshold(train_df, 5)
day_threshold = threshold(train_df, 9)
duration_threshold = threshold(train_df, 11)
campaign_threshold = threshold(train_df, 12)
pdays_threshold = threshold(train_df, 13)
previous_threshold = threshold(train_df, 14)


# In[13]:


"""print("age threshold : ", age_threshold)
print("balance threshold : ", balance_threshold)
print("day threshold : ", day_threshold)
print("duration threshold : ", duration_threshold)
print("campaign_threshold : ", campaign_threshold)
print("pdays threshold : ", pdays_threshold)
print("previous threshold : ", previous_threshold)"""


# In[14]:


#converting numerical values into binary categories
#if value>threshold, then 1, else 0
def convert(df, col, threshold):
    for i in range(len(df.iloc[:,col])):
        if df.iloc[i,col]>threshold:
            df.iloc[i,col] = 1
        else:
            df.iloc[i,col] = 0


# In[15]:


#converting train dataset
convert(train_df, 0, age_threshold)
convert(train_df, 5, balance_threshold)
convert(train_df, 9, day_threshold)
convert(train_df, 11, duration_threshold)
convert(train_df, 12, campaign_threshold)
convert(train_df, 13, pdays_threshold)
convert(train_df, 14, previous_threshold)


# In[16]:


print("Converting numberical values in TRAIN DATA into binary categories")
train_df.head()


# In[17]:


#converting test dataset
convert(test_df, 0, age_threshold)
convert(test_df, 5, balance_threshold)
convert(test_df, 9, day_threshold)
convert(test_df, 11, duration_threshold)
convert(test_df, 12, campaign_threshold)
convert(test_df, 13, pdays_threshold)
convert(test_df, 14, previous_threshold)


# In[18]:


print("Converting numberical values in TEST DATA into binary categories")
test_df.head()


# In[19]:


#finding the unique values an attribute can take
def unique_vals(df, col):
    return set(df.iloc[:,col])


# In[20]:


#finding the count of the unique values of the label attribute
def label_counts(df):
    counts = df.iloc[:,16].value_counts()
    return counts.to_dict()
#res = label_counts(train_df)
#print(res)


# In[21]:


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
        return "Is %s %s %s?" % (header[self.column], condition, str(self.value))


# In[22]:


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
    true_rows = pd.DataFrame(true_rows, columns = ['age', 'job', 'marital', 'education','default', 'balance', 'housing', 'loan', 'contact', 
            'day', 'month', 'duration', 'campaign', 'pdays','previous', 'poutcome', 'y'])
    false_rows = pd.DataFrame(false_rows, columns = ['age', 'job', 'marital', 'education','default', 'balance', 'housing', 'loan', 'contact', 
            'day', 'month', 'duration', 'campaign', 'pdays','previous', 'poutcome', 'y'])
    return true_rows, false_rows    
    


# In[23]:


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


# In[24]:


#defining the leaf node by setting it as the max value of all the values the attribute of a feature can take
class Leaf:
    def __init__(self, df):
        self.predictions = self.max_leaf(df)
        
    def max_leaf(self, df):
        res = label_counts(df)
        max_value = max(res, key = res.get)
        return {max_value:res[max_value]}


# In[25]:


#defining the decision node that splits the node
class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


# In[26]:


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


# In[27]:


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


# In[28]:


#method to make prediction for test dataset
def classify(row, node):

    # leaf node case
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


# In[29]:


print("THE BLOCKS TAKE AROUND 20 MINUTES TO EXECUTE")
print("Constructing decision tree using gini index with depth 3")
my_tree_1 = build_tree(gini, train_df, 3, 0) #decision tree using gini index with max depth as 3
#print_tree(my_tree_1)


# In[30]:


def calc_error(pred_tree):
    cnt = 0
    for i in range(len(test_df)):
        actual = test_df.iloc[i,16]
        predicted = list(classify(test_df.iloc[i,:], pred_tree))
        if actual != predicted[0]:
            cnt = cnt+1
    error = (cnt/len(test_df))*100
    return error
    


# In[31]:


#print("Error using gini index with depth 3: ",calc_error(my_tree_1))
error1 = calc_error(my_tree_1)


# In[33]:


print("Constructing decision tree using gini index with depth 10")
my_tree_2 = build_tree(gini, train_df, 10, 0) #decision tree using gini index with max depth as 10
#print_tree(my_tree_2)


# In[34]:


error2 = calc_error(my_tree_2)


# In[35]:


print("Constructing decision tree using entropy with depth 10")
my_tree_3 = build_tree(entropy, train_df, 10, 0) #decision tree using entropy with max depth as 10
#print_tree(my_tree_3)


# In[36]:


error3 = calc_error(my_tree_3)


# In[37]:


print("Constructing decision tree using entropy with depth 3")
my_tree_4 = build_tree(entropy, train_df, 3, 0) #decision tree using entropy index with max depth as 3
#print_tree(my_tree_4)


# In[38]:


error4 = calc_error(my_tree_4)


# In[39]:


print("Constructing decision tree using majority error with depth 10")
my_tree_5 = build_tree(ME, train_df, 10, 0) #decision tree using ME with max depth as 10
#print_tree(my_tree_5)


# In[40]:


error5 = calc_error(my_tree_5)


# In[41]:


print("Constructing decision tree using majority error with depth 3")
my_tree_6 = build_tree(ME, train_df, 3, 0) #decision tree using ME with max depth as 3
#print_tree(my_tree_6)


# In[42]:


error6 = calc_error(my_tree_6)


# In[46]:


#make predictions
def train_calc_error(pred_tree):
    cnt = 0
    for i in range(len(train_df)):
        actual = train_df.iloc[i,6] #label value
        predicted = list(classify(train_df.iloc[i,:], pred_tree)) #predicted label value
        if actual != predicted[0]:
            cnt = cnt+1
    error = (cnt/len(train_df))*100
    return error
    


# In[49]:


error7 = train_calc_error(my_tree_1) #3
error8 = train_calc_error(my_tree_2) #10
error9 = train_calc_error(my_tree_3) #10
error10 = train_calc_error(my_tree_4) #3
error11 = train_calc_error(my_tree_5) #10
error12 = train_calc_error(my_tree_6) #3


# In[54]:


print("Train data error")
print("\t\tdepth-3\t\t\t depth-10")
print("gini index\t", error7, "\t\t\t", error8)
print("entropy\t\t", error10, "\t\t\t", error9)
print("ME\t\t", error12, "\t", error11)


# In[45]:


print("Test data error")
print("\t\tdepth-3\t\t\t depth-6")
print("gini index\t", error1, "\t\t\t", error2)
print("entropy\t\t", error3, "\t\t\t", error4)
print("ME\t\t", error5, "\t", error6)


# Replacing "unknown" with the majority of other values in the same attribute 

# In[55]:


def general_label_counts(df, col):
    counts = df.iloc[:,col].value_counts()
    return counts.to_dict()

def max_val(df, col):
        res = general_label_counts(df, col)
        del res['unknown']
        max_value = max(res, key = res.get)
        return max_value


# In[56]:


print("replacing unknown values with median")
job_replace = max_val(train_df, 1)
train_df['job'] = train_df['job'].replace(['unknown'], job_replace)


# In[57]:


edu_replace = max_val(train_df, 3)
train_df['education'] = train_df['education'].replace(['unknown'], edu_replace)


# In[58]:


cont_replace = max_val(train_df, 8)
train_df['contact'] = train_df['contact'].replace(['unknown'], cont_replace)


# In[59]:


pout_replace = max_val(train_df, 15)
train_df['poutcome'] = train_df['poutcome'].replace(['unknown'], cont_replace)


# In[60]:


#replacing the unknown variables in test set
test_df['job'] = test_df['job'].replace(['unknown'], job_replace)
test_df['education'] = test_df['education'].replace(['unknown'], edu_replace)
test_df['contact'] = test_df['contact'].replace(['unknown'], cont_replace)
test_df['poutcome'] = test_df['poutcome'].replace(['unknown'], cont_replace)


# In[61]:


print("Constructing decision tree using gini index with depth 3")
my_tree_7 = build_tree(gini, train_df, 3, 0) #decision tree using gini index with max depth as 3
#print_tree(my_tree_7)


# In[62]:


error7 = calc_error(my_tree_7)


# In[63]:


print("Constructing decision tree using entropy with depth 3")
my_tree_8 = build_tree(entropy, train_df, 3, 0) #decision tree using entropy with max depth as 3
#print_tree(my_tree_8)


# In[64]:


error8 = calc_error(my_tree_8)


# In[65]:


print("Constructing decision tree using majority error with depth 3")
my_tree_9 = build_tree(ME, train_df, 3, 0) #decision tree using ME with max depth as 3
#print_tree(my_tree_9)


# In[66]:


error9 = calc_error(my_tree_9)


# In[67]:


print("Constructing decision tree using gini index with depth 10")
my_tree_10 = build_tree(gini, train_df, 10, 0) #decision tree using gini index with max depth as 10
#print_tree(my_tree_10)


# In[68]:


error10 = calc_error(my_tree_10)


# In[69]:


print("Constructing decision tree using entropy with depth 10")
my_tree_11 = build_tree(entropy, train_df, 10, 0) #decision tree using entropy with max depth as 10
#print_tree(my_tree_11)


# In[70]:


error11 = calc_error(my_tree_11)


# In[71]:


print("Constructing decision tree using majority error with depth 10")
my_tree_12 = build_tree(ME, train_df, 10, 0) #decision tree using ME with max depth as 3
#print_tree(my_tree_12)


# In[72]:


error12 = calc_error(my_tree_12)


# In[75]:


error13 = train_calc_error(my_tree_7)
error14 = train_calc_error(my_tree_8)
error15 = train_calc_error(my_tree_9) 
error16 = train_calc_error(my_tree_10) 
error17 = train_calc_error(my_tree_11) 
error18 = train_calc_error(my_tree_12) 


# In[79]:


print("Train data error")
print("\t\tdepth-3\t\t depth-10")
print("gini index\t", error13, "\t\t\t", error16)
print("entropy\t\t", error14, "\t\t\t", error17)
print("ME\t\t", error15, "\t", error18)


# In[74]:


print("Test data error")
print("\t\tdepth-3\t\t\t depth-6")
print("gini index\t", error7, "\t\t\t", error10)
print("entropy\t\t", error8, "\t\t\t", error11)
print("ME\t\t", error9, "\t\t\t", error12)


# In[ ]:




