import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog as fd
import MultiLayerPerceptron
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import MultiLayerPerceptron
add_bias = False
file_path = "C:\\Users\\p c\\Desktop\\University\\NN\\Labs\\Neural-Networks\\MLP\\data.csv"

def get_file_path():
    global file_path
    file_path = fd.askopenfilename()
    file_path_entry.delete(0, tk.END)
    file_path_entry.insert(0, file_path)

## print any variable that you want to trace in this function
def debug():
    print(file_path)
    print(var.get())
    print(input_size)
    print(output_size)
    print(number_of_layers)
    print(neurons_per_layer)
    print(activation_function)
    print(epochs)
    print(learning_rate)
    print(train_to_all_ratio)
    print(bias)
    
def convert(predictions) :
    classes = []
    for prediction in predictions:
        max_value_index = prediction.tolist().index(max(prediction))
        classes.append(max_value_index)
    return np.array([classes])

def convert2(actual):
    classes = []
    for act in actual:
        max_value_index = act.tolist().index(max(act))
        classes.append(max_value_index)
    return np.array([classes])

accuracy = 0
def Train_MLP():
    text_variable.set('training')
    global input_size, output_size, neurons_per_layer, activation_function, bias, number_of_layers
    global epochs, learning_rate, train_to_all_ratio

    # Neural Network's Architecture [trainable parameters]
    input_size = int(input_layer_entry.get())
    output_size = int(output_layer_entry.get())
    neurons_per_layer = [int(neuron) for neuron in neurons_per_layer_entry.get().split(' ')]
    number_of_layers = len(neurons_per_layer)
    activation_function = activation_function_name.get()
    bias = var.get()
    

    #Hyperparameters
    epochs = int(epochs_entry.get())
    learning_rate = float(learning_rate_entry.get())
    train_to_all_ratio = float(train_to_all_ratio_entry.get())



    data = pd.read_csv(file_path)
    X_ = data[['Area',	'Perimeter', 'MajorAxisLength',	'MinorAxisLength',	'roundnes']]
    y_ = data[['Class_BOMBAY',	'Class_CALI',	'Class_SIRA']]
    size = data.shape[0]
    train_size = int(train_to_all_ratio * size)
    X_train = X_.iloc[0 : train_size].values
    y_train = y_.iloc[0 : train_size].values
    X_test = X_.iloc[train_size: ].values
    y_test = y_.iloc[train_size: ].values


    MLP = MultiLayerPerceptron.MultiLayerPerceptron(input_size, output_size, number_of_layers, neurons_per_layer, learning_rate, activation_function, epochs, bias, 42)

    # XOR
    MLP.fit(X_train, y_train)
    text_variable.set('done')

    predictions = MLP.predict(X=np.array(X_test))
    global p
    global a
    p  = convert(predictions)
    a  = convert2(y_test)
    global cm 
    cm = confusion_matrix(a.reshape(-1, 1), p.reshape(-1, 1))
    global accuracy
    accuracy = accuracy_score(a.reshape(-1, 1), p.reshape(-1, 1))
    accuracy_display.insert(0, accuracy)
    print(MLP.weights)
    # accuracy_display.set
    # recall = recall_score(a.reshape(-1, 1), p.reshape(-1, 1 ), average='macro')
    # precision = precision_score(a.reshape(-1, 1), p.reshape(-1, 1),  average='macro')

def cm_():
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='coolwarm', fmt='g', cbar=False, linewidths=0.8)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix', fontsize=20)
    plt.show()

def show_metrics():
    # precision_display.insert(0, f"{precision}")
    # recall_display.insert(0, f"{recall}")
    accuracy_display.insert(0, f"{accuracy}")

## important constants
WIN_WIDTH, WIN_HEIGHT = 550, 500
WIDGETS_LEFT_ALIGNMENT1 = 140
LABELS_LEFT_ALIGNMENT1 = 20
WIDGETS_LEFT_ALIGNMENT2 = 410
LABELS_LEFT_ALIGNMENT2 = 310
WIDGETS_LEFT_ALIGNMENT3 = 300
LABELS_LEFT_ALIGNMENT3 = 230
depth = 100
chunk = 40

# initializing main window
window = tk.Tk()
window.maxsize(WIN_WIDTH, WIN_HEIGHT)
window.minsize(WIN_WIDTH, WIN_HEIGHT)
window.geometry(f'{WIN_WIDTH}x{WIN_HEIGHT}')

# file path
file_path_entry = tk.Entry(window, width=52)
file_path_entry.place(x=WIDGETS_LEFT_ALIGNMENT1-60 , y=10)
file_path_label = tk.Label(window, text='Data:')
file_path_label.place(x = LABELS_LEFT_ALIGNMENT1, y=10)
file_path_browse_button = tk.Button(window, command= get_file_path, text='Browse', width=10)
file_path_browse_button.place(x=400, y=5)
file_path_entry.insert(0, file_path)

# temp debugging button
temp_button = tk.Button(window, command= debug, text='Debug', width=7)
temp_button.place(x=490, y=5)

## labels
ANN_architecture_label = tk.Label(window, text='Architecture',  font=("Helvetica", 12))
ANN_architecture_label.place(x=WIDGETS_LEFT_ALIGNMENT1-60, y=60)
Hyperparameters_label = tk.Label(window, text='Hyperparameters',  font=("Helvetica", 12))
Hyperparameters_label.place(x=WIDGETS_LEFT_ALIGNMENT1+190, y=60)

# input layer entry
input_layer_entry = tk.Entry(window, width = 7)
input_layer_entry.place(x=WIDGETS_LEFT_ALIGNMENT1 , y=depth)
input_layer_label = tk.Label(window, text='Input Layer Size')
input_layer_label.place(x=LABELS_LEFT_ALIGNMENT1, y=depth)

# output layer entry
output_layer_entry = tk.Entry(window, width = 7)
output_layer_entry.place(x=WIDGETS_LEFT_ALIGNMENT1 , y=depth + chunk)
output_layer_label = tk.Label(window, text='Output Layer Size')
output_layer_label.place(x=LABELS_LEFT_ALIGNMENT1, y=depth + chunk)

# Neurons per layer entry
neurons_per_layer_entry = tk.Entry(window, width = 15)
neurons_per_layer_entry.place(x=WIDGETS_LEFT_ALIGNMENT1 , y=depth + chunk*2)
output_layer_label = tk.Label(window, text='Neurons per Layer')
output_layer_label.place(x=LABELS_LEFT_ALIGNMENT1 , y=depth + chunk*2)

# Activation function
activation_function_name = tk.StringVar()
activation_function_combo_box = ttk.Combobox(window, width=12, values=['Tanh', 'Sigmoid'], textvariable=activation_function_name, )
activation_function_combo_box.place(x=WIDGETS_LEFT_ALIGNMENT1 , y=depth + chunk*3)
activation_function_combo_box.current(0)
activation_function_label = tk.Label(window, text='Activation Function')
activation_function_label.place(x=LABELS_LEFT_ALIGNMENT1 , y=depth + chunk*3)

# Bias checkbox
var = tk.BooleanVar() # this var holds the current value of the check_box
bias_check_box = tk.Checkbutton(window , variable=var)
bias_check_box.place(x= WIDGETS_LEFT_ALIGNMENT1 , y =depth+chunk*4)
bias_label = tk.Label(window, text='add bias')
bias_label.place(x=LABELS_LEFT_ALIGNMENT1 , y=depth+chunk*4)

# Epochs entry
epochs_entry = tk.Entry(window, width = 10)
epochs_entry.place(x=WIDGETS_LEFT_ALIGNMENT2 , y=depth)
epochs_label = tk.Label(window, text='epochs')
epochs_label.place(x=LABELS_LEFT_ALIGNMENT2 , y=depth)

# Learning rate entry
learning_rate_entry = tk.Entry(window, width = 10)
learning_rate_entry.place(x=WIDGETS_LEFT_ALIGNMENT2 , y=depth+chunk)
learning_rate_label = tk.Label(window, text='learning rate')
learning_rate_label.place(x=LABELS_LEFT_ALIGNMENT2 , y=depth+chunk)

# train/all ratio entry
train_to_all_ratio_entry = tk.Entry(window, width = 10)
train_to_all_ratio_entry.place(x=WIDGETS_LEFT_ALIGNMENT2 , y=depth+chunk*2)
train_to_all_ratio_label = tk.Label(window, text='train/all ratio')
train_to_all_ratio_label.place(x=LABELS_LEFT_ALIGNMENT2 , y=depth+chunk*2)

# train button
train_button = tk.Button(window, command= Train_MLP, text='Train')
train_button.place(x=WIDGETS_LEFT_ALIGNMENT1-50, y=320, height=25, width=105)
text_variable = tk.StringVar()
text_variable.set('off')
train_status_label = tk.Label(window, textvariable=text_variable )
train_status_label.place(x=WIDGETS_LEFT_ALIGNMENT1-40 , y=350)

# test button
train_button = tk.Button(window, command= show_metrics, text='Test')
train_button.place(x=WIDGETS_LEFT_ALIGNMENT1-50, y=370, height=25, width=105)

# confusion matrix
confusion_matrix_button = tk.Button(window, command= cm_, text='Confusion Matrix')
confusion_matrix_button.place(x=WIDGETS_LEFT_ALIGNMENT1-50, y=400, height=25, width=105)

## Metrics
# accuracy
accuracy_display = tk.Entry(window, width = 15)
accuracy_display.place(x=WIDGETS_LEFT_ALIGNMENT3 , y=320)   # 330
accuracy_label = tk.Label(window, text='Accuracy')
accuracy_label.place(x=LABELS_LEFT_ALIGNMENT3 , y=320)

# precision
precision_display = tk.Entry(window, width = 15, )
precision_display.place(x=WIDGETS_LEFT_ALIGNMENT3 , y=360) #375
precision_label = tk.Label(window, text='Precision')
precision_label.place(x=LABELS_LEFT_ALIGNMENT3 , y=360)

# accuracy
recall_display = tk.Entry(window, width = 15, )
recall_display.place(x=WIDGETS_LEFT_ALIGNMENT3 , y=405) #420
recall_label = tk.Label(window, text='Recall')
recall_label.place(x=LABELS_LEFT_ALIGNMENT3 , y=405)

tk.mainloop()
