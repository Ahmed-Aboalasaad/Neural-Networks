import numpy as np

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


def mean_squared_error(y_true, y_pred):
    return np.sum((y_true - y_pred)**2) / len(y_true)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy