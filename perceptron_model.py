#!/usr/bin/env python
# coding: utf-8

# # Objective-To determine seven different types of beans like CALI,SIRA,BOMBAY based on their physical appearancelike:
# ### Area,Perimeter,MajorAxisLength,MinorAxisLength,Roundness

# * Area - (A), The area of a bean zone and the number of pixels within its boundaries.
# * Perimeter - (P), Bean circumference is defined as the length of its border.
# * MajorAxisLength - (L), The distance between the ends of the longest line that can be drawn from a bean.
# * MinorAxisLength - (l), The longest line that can be drawn from the bean while standing perpendicular to the main axis.
# * Roundness - (R), Calculated with the following formula: (4piA)/(P^2)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron


# In[2]:


data = pd.read_csv('final_file.csv')


# In[3]:


data.head()


# In[4]:


data['Class'].unique()


# In[5]:


data.isnull().sum()


# In[ ]:





# In[6]:


def split_dataset(dataset, c1 : np.ndarray, c2 : np.ndarray, feature1, feature2, class_train_size):
    data = dataset[[feature1, feature2, 'Class']]
    data = data[(dataset['Class'] == c1) | (dataset['Class'] == c2)]
    
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()

    # adding an amount of [class_train_size] from each class to the train set, and an amount of [50 - class_train_size] to the test set
    for c in [c1 , c2]:
        class_data = data[data['Class'] == c] 
        train_set = pd.concat([train_set, class_data.iloc[:class_train_size]])
        test_set = pd.concat([test_set, class_data.iloc[class_train_size:]])

    # encoding
    train_set.replace({c1 : -1,c2 : 1},inplace=True)
    test_set.replace({c1 : -1,c2 : 1},inplace= True)
    
    #shuffling
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    test_set = test_set.sample(frac=1).reset_index(drop=True)
    
    return train_set, test_set


# In[ ]:





# In[7]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'Perimeter'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[8]:


train_set


# In[9]:


test_set


# In[10]:


# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='s', label='Versicolor')
# plt.xlabel('Sepal length [cm]')
# plt.ylabel('Petal length [cm]')
# plt.legend(loc='upper left')
# plt.show()


# In[11]:


train_set.shape


# In[12]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[13]:


X


# In[14]:


y


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[16]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[17]:


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # initialize weights and bias
        # we want weights to have the length of the number of features because we have w for each feature
        self.weights = np.random.random(n_features)
        self.bias = 0
        
        max_label = np.max(y) # Finds the maximum label in the 'y' array (original labels)
        y_ = np.array([1 if i == max_label else -1 for i in y]) # Transforms labels in 'y' to binary
        
        for iter in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)
                
                # Perceptron update rule
                update = self.learning_rate * (y_predicted - y_[idx])
                
                self.weights -= update * x_i
                self.bias -= update
        
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted
    
    def activation_function(self, x):
        return np.where(x>=0, 1, -1)


# In[18]:


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# In[19]:


model = Perceptron()
model.fit(X_train, y_train)


# In[20]:


model.weights


# In[21]:


model.bias


# In[22]:


predictions = model.predict(X_test)


# In[23]:


print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[24]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[25]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming your data (X, y_train, y_test) and trained model (model) are available

# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[26]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'MajorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[27]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[29]:


model = Perceptron()
model.fit(X_train, y_train)


# In[30]:


predictions = model.predict(X_test)


# In[31]:


print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[32]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[33]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming your data (X, y_train, y_test) and trained model (model) are available

# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[34]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'MinorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[35]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[37]:


model = Perceptron()
model.fit(X_train, y_train)


# In[38]:


predictions = model.predict(X_test)


# In[39]:


print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[40]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[41]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[42]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[43]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[45]:


model = Perceptron()
model.fit(X_train, y_train)


# In[46]:


predictions = model.predict(X_test)


# In[47]:


print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[48]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[49]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[50]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Perimeter'  # Replace with desired feature
feature2 = 'MajorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[51]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[53]:


model = Perceptron()
model.fit(X_train, y_train)


# In[54]:


predictions = model.predict(X_test)


# In[55]:


print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[56]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[57]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[58]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Perimeter'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[59]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[61]:


model = Perceptron()
model.fit(X_train, y_train)


# In[62]:


predictions = model.predict(X_test)


# In[63]:


print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[64]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[65]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[66]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Perimeter'  # Replace with desired feature
feature2 = 'MinorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[67]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[68]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[69]:


model = Perceptron()
model.fit(X_train, y_train)


# In[70]:


predictions = model.predict(X_test)


# In[71]:


print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[72]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[73]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[74]:


data.columns


# In[75]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'MajorAxisLength'  # Replace with desired feature
feature2 = 'MinorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[76]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[77]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[78]:


model = Perceptron()
model.fit(X_train, y_train)


# In[79]:


predictions = model.predict(X_test)


# In[80]:


print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[81]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[82]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[83]:


data.columns


# In[84]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'MajorAxisLength'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[85]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[86]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[87]:


model = Perceptron()
model.fit(X_train, y_train)


# In[88]:


predictions = model.predict(X_test)


# In[89]:


print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[90]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[91]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[92]:


data.columns


# In[93]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'MinorAxisLength'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[94]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[95]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[96]:


model = Perceptron()
model.fit(X_train, y_train)


# In[97]:


predictions = model.predict(X_test)


# In[98]:


print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[99]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[100]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[101]:


data.columns


# In[102]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'MajorAxisLength'  # Replace with desired feature
feature2 = 'MinorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[103]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[104]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[105]:


data.columns


# In[106]:


c1 = 'BOMBAY'
c2 = 'SIRA'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'Perimeter'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[107]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[108]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[109]:


c1 = 'BOMBAY'
c2 = 'SIRA'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[110]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[111]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[112]:


c1 = 'BOMBAY'
c2 = 'SIRA'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[113]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[114]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[115]:


c1 = 'BOMBAY'
c2 = 'SIRA'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[116]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[117]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[118]:


c1 = 'BOMBAY'
c2 = 'SIRA'

feature1 = 'Perimeter'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[119]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[120]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[121]:


c1 = 'BOMBAY'
c2 = 'SIRA'

feature1 = 'Perimeter'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[122]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[123]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[124]:


c1 = 'BOMBAY'
c2 = 'SIRA'

feature1 = 'Perimeter'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[125]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[126]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[127]:


c1 = 'BOMBAY'
c2 = 'SIRA'

feature1 = 'MajorAxisLength'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[128]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[129]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[130]:


c1 = 'BOMBAY'
c2 = 'SIRA'

feature1 = 'MinorAxisLength'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[131]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[132]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[133]:


data.columns


# In[134]:


c1 = 'CALI'
c2 = 'SIRA'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'Perimeter'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[135]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[136]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[137]:


c1 = 'CALI'
c2 = 'SIRA'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'MajorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[138]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[139]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[140]:


c1 = 'CALI'
c2 = 'SIRA'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'MinorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[141]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[142]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[143]:


c1 = 'CALI'
c2 = 'SIRA'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[144]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[145]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[146]:


c1 = 'CALI'
c2 = 'SIRA'

feature1 = 'Perimeter'  # Replace with desired feature
feature2 = 'MinorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[147]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[148]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[149]:


c1 = 'CALI'
c2 = 'SIRA'

feature1 = 'Perimeter'  # Replace with desired feature
feature2 = 'MinorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[150]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[151]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[152]:


c1 = 'CALI'
c2 = 'SIRA'

feature1 = 'Perimeter'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[153]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[154]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[155]:


data.columns


# In[156]:


c1 = 'CALI'
c2 = 'SIRA'

feature1 = 'MajorAxisLength'  # Replace with desired feature
feature2 = 'MinorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[157]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[158]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[159]:


c1 = 'CALI'
c2 = 'SIRA'

feature1 = 'MajorAxisLength'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[160]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[161]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[162]:


c1 = 'CALI'
c2 = 'SIRA'

feature1 = 'MinorAxisLength'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)

# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
#call the perceptron model and fit the data
model = Perceptron()
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[163]:


import matplotlib.pyplot as plt

# Predict labels for the test set
predictions = model.predict(X_test)

# Create a scatter plot for training data
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:, 1], color='blue', label='BOMBAY - Train')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='red', label='CALI - Train')

# Create a scatter plot for testing data, make them a bit grayish
plt.scatter(X_test[y_test==-1][:, 0], X_test[y_test==-1][:, 1], color='lightblue', label='BOMBAY - Test')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], color='pink', label='CALI - Test')

# Create a scatter plot for predictions on the test data
plt.scatter(X_test[predictions==-1][:, 0], X_test[predictions==-1][:, 1], color='lightblue', marker='x', label='Predicted BOMBAY')
plt.scatter(X_test[predictions==1][:, 0], X_test[predictions==1][:, 1], color='pink', marker='x', label='Predicted CALI')

# Add labels and legend
plt.xlabel('bean area')
plt.ylabel('bean parameter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[164]:


# Extract weights and bias from the trained model
w1, w2 = model.weights  # Extract weights without bias
b=model.bias
# Define x-axis range (adjust based on your data)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y-values for the line equation
y_values = (-w1 / w2) * x_range - (b / w2)

# Create the plot
plt.figure()

# Plot the decision boundary line
plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')

# Plot training points (assuming appropriate colors and markers)
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')

# Plot testing points (assuming appropriate colors and markers)
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')

# Set labels and legend
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# Show the plot
plt.show()

