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


# In[7]:


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000, bias_flag=True):
        self.learning_rate = learning_rate 
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.bias_flag=bias_flag
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
                if self.bias_flag:
                    self.bias -= update
        
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted
    
    def activation_function(self, x):
        return np.where(x>=0, 1, -1)


# In[8]:


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# In[9]:


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


# In[10]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'Perimeter'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[11]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[13]:


model = Perceptron()
model.fit(X_train, y_train)


# In[14]:


model.weights


# In[15]:


model.bias


# In[16]:


predictions = model.predict(X_test)


# In[ ]:





# In[17]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[18]:


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
plt.ylabel('bean perimeter')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[19]:


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


# In[20]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'MajorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[21]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[23]:


model = Perceptron()
model.fit(X_train, y_train)


# In[24]:


predictions = model.predict(X_test)


# In[25]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[26]:


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
plt.ylabel('bean major axis length')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[27]:


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
plt.ylabel('bean major axes length')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[28]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'MinorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[29]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[31]:


model = Perceptron()
model.fit(X_train, y_train)


# In[32]:


predictions = model.predict(X_test)


# In[33]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[34]:


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
plt.ylabel('bean minor axis lenght')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[35]:


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
plt.ylabel('bean minor axis lenght')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[36]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Area'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[37]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[39]:


model = Perceptron()
model.fit(X_train, y_train)


# In[40]:


predictions = model.predict(X_test)


# In[41]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[42]:


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
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[43]:


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
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[44]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Perimeter'  # Replace with desired feature
feature2 = 'MajorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[45]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[47]:


model = Perceptron()
model.fit(X_train, y_train)


# In[48]:


predictions = model.predict(X_test)


# In[49]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[50]:


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
plt.xlabel('bean perimeter')
plt.ylabel('bean MajorAxisLength')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[51]:


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
plt.xlabel('bean perimeter')
plt.ylabel('bean MajorAxisLength')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[52]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Perimeter'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[53]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[55]:


model = Perceptron()
model.fit(X_train, y_train)


# In[56]:


predictions = model.predict(X_test)


# In[57]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[58]:


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
plt.xlabel('bean perimeter')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[59]:


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
plt.xlabel('bean perimeter')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[60]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'Perimeter'  # Replace with desired feature
feature2 = 'MinorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[61]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[63]:


model = Perceptron()
model.fit(X_train, y_train)


# In[64]:


predictions = model.predict(X_test)


# In[65]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[66]:


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
plt.xlabel('bean perimeter')
plt.ylabel('bean MinorAxisLength')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[67]:


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
plt.xlabel('bean perimeter')
plt.ylabel('bean MinorAxisLength')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[68]:


data.columns


# In[69]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'MajorAxisLength'  # Replace with desired feature
feature2 = 'MinorAxisLength'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[70]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[72]:


model = Perceptron()
model.fit(X_train, y_train)


# In[73]:


predictions = model.predict(X_test)


# In[74]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[75]:


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
plt.xlabel('bean MajorAxisLength')
plt.ylabel('bean MinorAxisLength')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[76]:


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
plt.xlabel('bean MajorAxisLength')
plt.ylabel('bean MinorAxisLength')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[77]:


data.columns


# In[78]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'MajorAxisLength'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[79]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[81]:


model = Perceptron()
model.fit(X_train, y_train)


# In[82]:


predictions = model.predict(X_test)


# In[83]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[84]:


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
plt.xlabel('bean MajorAxisLenght')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[85]:


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
plt.xlabel('bean MajorAxisLenght')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[86]:


data.columns


# In[87]:


c1 = 'BOMBAY'
c2 = 'CALI'

feature1 = 'MinorAxisLength'  # Replace with desired feature
feature2 = 'roundnes'  # Replace with desired feature

class_train_size = 30

# Split the data using the provided function
train_set, test_set = split_dataset(data, c1, c2, feature1, feature2, class_train_size)


# In[88]:


# Select only 2 features for visualization purpose
X = train_set.iloc[:,[0,1]].values
y = train_set.iloc[:, -1].values


# In[89]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[90]:


model = Perceptron()
model.fit(X_train, y_train)


# In[91]:


predictions = model.predict(X_test)


# In[92]:


print(confusion_matrix(y_test, predictions))

print(f'Perceptron classification accuracy: {accuracy(y_test, predictions)}')


# In[93]:


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
plt.xlabel('bean MinorAxisLenght')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[94]:


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
plt.xlabel('bean MinorAxisLenght')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[95]:


data.columns


# In[96]:


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


# In[97]:


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


# In[98]:


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


# In[99]:


c1 = 'BOMBAY'
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
plt.ylabel('bean MajorAxisLenght')
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
plt.xlabel('bean area')
plt.ylabel('bean MajorAxisLenght')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[102]:


c1 = 'BOMBAY'
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
plt.ylabel('bean MinorAxisLenght')
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
plt.xlabel('bean area')
plt.ylabel('bean MinorAxisLenght')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[105]:


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


# In[106]:


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
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[107]:


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
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[108]:


c1 = 'BOMBAY'
c2 = 'SIRA'

feature1 = 'Perimeter'  # Replace with desired feature
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


# In[109]:


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
plt.xlabel('bean perimeter')
plt.ylabel('bean MajorAxisLength')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[110]:


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
plt.xlabel('bean perimeter')
plt.ylabel('bean MajorAxisLength')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[111]:


c1 = 'BOMBAY'
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


# In[112]:


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
plt.xlabel('bean perimeter')
plt.ylabel('bean MinorAxisLength')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[113]:


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
plt.xlabel('bean perimeter')
plt.ylabel('bean MinorAxisLength')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[114]:


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


# In[115]:


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
plt.xlabel('bean Perimeter')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[116]:


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
plt.xlabel('bean Perimeter')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[117]:


c1 = 'BOMBAY'
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


# In[118]:


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
plt.xlabel('bean MajorAxisLength')
plt.ylabel('bean MinorAxisLength')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[119]:


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
plt.xlabel('bean MajorAxisLength')
plt.ylabel('bean MinorAxisLength')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[120]:


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


# In[121]:


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
plt.xlabel('bean MajorAxisLength')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[122]:


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
plt.xlabel('bean MajorAxisLength')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[123]:


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


# In[124]:


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
plt.xlabel('bean MinorAxisLength')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[125]:


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
plt.xlabel('bean MinorAxisLength')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[126]:


data.columns


# In[127]:


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
plt.xlabel('bean Area')
plt.ylabel('bean major axis lenght')
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
plt.xlabel('bean Area')
plt.ylabel('bean major axis lenght')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[133]:


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


# In[134]:


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
plt.xlabel('bean Area')
plt.ylabel('bean minor axis lenght')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[135]:


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
plt.xlabel('bean Area')
plt.ylabel('bean minor axis lenght')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[136]:


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


# In[137]:


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
plt.xlabel('bean Area')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[138]:


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
plt.xlabel('bean Area')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[139]:


c1 = 'CALI'
c2 = 'SIRA'

feature1 = 'Perimeter'  # Replace with desired feature
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


# In[140]:


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
plt.xlabel('bean Perimeter')
plt.ylabel('bean MajorAxisLenght')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[141]:


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
plt.xlabel('bean Perimeter')
plt.ylabel('bean MajorAxisLenght')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[142]:


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


# In[143]:


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
plt.xlabel('bean Perimeter')
plt.ylabel('bean MinorAxisLenght')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[144]:


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
plt.xlabel('bean Perimeter')
plt.ylabel('bean MinorAxisLenght')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[145]:


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


# In[146]:


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
plt.xlabel('bean Perimeter')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[147]:


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
plt.xlabel('bean Perimeter')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[148]:


data.columns


# In[149]:


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
plt.xlabel('bean MajorAxisLength')
plt.ylabel('bean MinorAxisLength')
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
plt.xlabel('bean MajorAxisLength')
plt.ylabel('bean MinorAxisLength')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[152]:


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
plt.xlabel('bean MajorAxisLength')
plt.ylabel('bean roundnes')
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
plt.xlabel('bean MajorAxisLength')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[ ]:





# In[155]:


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


# In[156]:


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
plt.xlabel('bean MinorAxisLength')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[157]:


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
plt.xlabel('bean MinorAxisLength')
plt.ylabel('bean roundnes')
plt.legend(loc='upper left')

# Show the plot
plt.show()

