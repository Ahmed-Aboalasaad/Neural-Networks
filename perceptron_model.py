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
# from sklearn.linear_model import Perceptron


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


def confusion_matrix(actual, predicted):

  # Check if the input arrays have the same size
  if len(actual) != len(predicted):
    raise ValueError("Number of actual and predicted labels must be equal.")

  # Extract the unique classes (assuming they are integers)
  classes = np.unique(actual)

  # Initialize the confusion matrix with zeros
  confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)

  # Iterate through each data point
  for i in range(len(actual)):
    actual_class = actual[i]
    predicted_class = predicted[i]

    # Find the indices of the actual and predicted classes in the confusion matrix
    actual_index = np.where(classes == actual_class)[0][0]
    predicted_index = np.where(classes == predicted_class)[0][0]

    # Increment the corresponding cell in the confusion matrix
    confusion_matrix[actual_index, predicted_index] += 1

  return confusion_matrix


# In[20]:


model = Perceptron()
model.fit(X_train, y_train)


# In[21]:


model.weights


# In[22]:


model.bias


# In[23]:


predictions = model.predict(X_test)


# In[ ]:





# In[24]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[25]:


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


# In[26]:


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


# In[27]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'MajorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[28]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[30]:


model = Perceptron()
model.fit(X_train, y_train)


# In[31]:


predictions = model.predict(X_test)


# In[32]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[33]:


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


# In[34]:


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


# In[35]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'MinorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[36]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[38]:


model = Perceptron()
model.fit(X_train, y_train)


# In[39]:


predictions = model.predict(X_test)


# In[40]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[41]:


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


# In[42]:


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


# In[43]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[44]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[46]:


model = Perceptron()
model.fit(X_train, y_train)


# In[47]:


predictions = model.predict(X_test)


# In[48]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[49]:


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


# In[50]:


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


# In[51]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Perimeter'  # Replace with desired feature
feature2 = 'MajorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[52]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[54]:


model = Perceptron()
model.fit(X_train, y_train)


# In[55]:


predictions = model.predict(X_test)


# In[56]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[57]:


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


# In[58]:


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





# In[59]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Perimeter'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[60]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[61]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[62]:


model = Perceptron()
model.fit(X_train, y_train)


# In[63]:


predictions = model.predict(X_test)


# In[64]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[65]:


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


# In[66]:


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





# In[67]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Perimeter'  # Replace with desired feature
feature2 = 'MinorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[68]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[69]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[70]:


model = Perceptron()
model.fit(X_train, y_train)


# In[71]:


predictions = model.predict(X_test)


# In[72]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[73]:


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


# In[74]:


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


# In[75]:


data.columns


# In[76]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'MajorAxisLength'  # Replace with desired feature
feature2 = 'MinorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[77]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[78]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[79]:


model = Perceptron()
model.fit(X_train, y_train)


# In[80]:


predictions = model.predict(X_test)


# In[81]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[82]:


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


# In[83]:


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


# In[84]:


data.columns


# In[85]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'MajorAxisLength'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[86]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[88]:


model = Perceptron()
model.fit(X_train, y_train)


# In[89]:


predictions = model.predict(X_test)


# In[90]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[91]:


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


# In[92]:


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


# In[93]:


data.columns


# In[94]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'MinorAxisLength'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[95]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[96]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[97]:


model = Perceptron()
model.fit(X_train, y_train)


# In[98]:


predictions = model.predict(X_test)


# In[99]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[100]:


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


# In[101]:


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


# In[102]:


data.columns


# In[103]:


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
print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[104]:


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


# In[105]:


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


# In[106]:


data.columns


# In[107]:


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
print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[108]:


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


# In[109]:


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


# In[110]:


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
print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[111]:


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


# In[112]:


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


# In[113]:


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
print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[114]:


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


# In[115]:


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


# In[116]:


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
print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[117]:


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


# In[118]:


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


# In[119]:


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
print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[120]:


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


# In[121]:


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


# In[122]:


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
print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[123]:


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


# In[124]:


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


# In[125]:


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
print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[126]:


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


# In[127]:


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


# In[128]:


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
print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[129]:


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


# In[130]:


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


# In[131]:


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
print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[132]:


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


# In[133]:


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





# In[134]:


data.columns


# In[135]:


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
print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[136]:


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


# In[137]:


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


# In[138]:


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
print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[139]:


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


# In[140]:


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





# In[141]:


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
print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[142]:


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


# In[143]:


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





# In[144]:


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
print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[145]:


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


# In[146]:


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





# In[147]:


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
print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[148]:


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


# In[149]:


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





# In[150]:


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
print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[151]:


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


# In[152]:


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





# In[153]:


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

print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[154]:


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


# In[155]:


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


# In[156]:


data.columns


# In[157]:


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
print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[158]:


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


# In[159]:


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





# In[160]:


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

print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[161]:


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


# In[162]:


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





# In[163]:


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

print(confusion_matrix(y_test, predictions))
print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[164]:


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


# In[165]:


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

