import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog as fd
import re
import MultiLayerPerceptron
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
add_bias = False
file_path = "C:\\Users\\p c\\Desktop\\University\\NN\\Labs\\Neural-Networks\\MLP\\data.csv"

def get_file_path():
    global file_path
    file_path = fd.askopenfilename()
    txt_file_path.delete(0, tk.END)
    txt_file_path.insert(0, file_path)

## print any variable that you want to trace in this function
def debug():
    print(file_path)
    print(addBias.get())
    print(input_size)
    print(output_size)
    print(number_of_layers)
    print(neurons_per_layer)
    print(activation_function)
    print(epochs)
    print(learning_rate)
    print(training_ratio)
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
    global input_size, output_size, neurons_per_layer, activation_function, bias, number_of_layers
    global epochs, learning_rate, training_ratio

    # Neural Network's Architecture [trainable parameters]
    input_size = int(txt_input_layer.get())
    output_size = int(txt_output_layer.get())
    neurons_per_layer = re.findall(r'\d+', txt_neurons_per_layer.get())
    neurons_per_layer = [int(number) for number in neurons_per_layer]
    hidden_layers_number = len(neurons_per_layer)
    activation_function = combo_activation_function.get()
    bias = addBias.get()
    
    # Hyperparameters
    epochs = int(txt_epochs.get())
    learning_rate = float(txt_learning_rate.get())
    training_ratio = float(txt_training_ratio.get())

    # Data Split
    data = pd.read_csv(file_path)
    # X = data[['Area',	'Perimeter', 'MajorAxisLength',	'MinorAxisLength',	'roundnes']]
    # y = data[['Class_BOMBAY',	'Class_CALI',	'Class_SIRA']]
    # size = data.shape[0]
    # train_size = int(training_ratio * size)
    # x_train = X.iloc[0 : train_size].values
    # y_train = y.iloc[0 : train_size].values
    # x_test = X.iloc[train_size: ].values
    # y_test = y.iloc[train_size: ].values

    X = data.iloc[:, :input_size]
    Y = data.iloc[:, -output_size:]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=training_ratio, random_state=42)

    # Fitting
    MLP = MultiLayerPerceptron.MultiLayerPerceptron(input_size, output_size, hidden_layers_number, neurons_per_layer, learning_rate, activation_function, epochs, bias, 42)
    MLP.fit(x_train, y_train)

    # Metrics
    predictions = MLP.predict(X=np.array(x_test))
    global p
    global a
    global cm 
    global accuracy
    p  = convert(predictions)
    a  = convert2(y_test)
    cm = confusion_matrix(a.reshape(-1, 1), p.reshape(-1, 1))

    accuracy = accuracy_score(a.reshape(-1, 1), p.reshape(-1, 1))
    txt_accuracy.insert(0,str(accuracy))
    recall = recall_score(a.reshape(-1, 1), p.reshape(-1, 1 ), average='macro')
    txt_recall.insert(0, str(recall))
    precision = precision_score(a.reshape(-1, 1), p.reshape(-1, 1),  average='macro')
    txt_precision.insert(0, str(precision))
    print(f'\nWeights after training:\n{MLP.weights}\n')

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
    txt_accuracy.insert(0, f"{accuracy}")

## Constants for the layout
WIN_WIDTH, WIN_HEIGHT = 550, 500
WIDGETS_LEFT_ALIGNMENT1 = 140
LABELS_LEFT_ALIGNMENT1 = 20
WIDGETS_LEFT_ALIGNMENT2 = 410
LABELS_LEFT_ALIGNMENT2 = 310
WIDGETS_LEFT_ALIGNMENT3 = 350
LABELS_LEFT_ALIGNMENT3 = 285
depth = 100
chunk = 40

# initializing main window
window = tk.Tk()
window.maxsize(WIN_WIDTH, WIN_HEIGHT)
window.minsize(WIN_WIDTH, WIN_HEIGHT)
window.geometry(f'{WIN_WIDTH}x{WIN_HEIGHT}')

# file path
txt_file_path = tk.Entry(window, width=52)
txt_file_path.place(x=WIDGETS_LEFT_ALIGNMENT1-60 , y=10)
label_file_path = tk.Label(window, text='Data:')
label_file_path.place(x = LABELS_LEFT_ALIGNMENT1, y=10)
btn_browse = tk.Button(window, command= get_file_path, text='Browse', width=10)
btn_browse.place(x=400, y=5)
txt_file_path.insert(0, file_path)

# Debugging button
btn_debug = tk.Button(window, command=debug, text='Debug', width=7)
btn_debug.place(x=490, y=5)

## labels
label_architecture = tk.Label(window, text='Architecture',  font=("Helvetica", 12))
label_architecture.place(x=WIDGETS_LEFT_ALIGNMENT1-60, y=60)
label_hyperparameters = tk.Label(window, text='Hyperparameters',  font=("Helvetica", 12))
label_hyperparameters.place(x=WIDGETS_LEFT_ALIGNMENT1+190, y=60)
label_metrics = tk.Label(window, text='Metrics', font=("Helvetica", 12))
label_metrics.place(x=WIDGETS_LEFT_ALIGNMENT1+190, y=275)

# input layer
txt_input_layer = tk.Entry(window, width = 7)
txt_input_layer.place(x=WIDGETS_LEFT_ALIGNMENT1 , y=depth)
label_input_layer = tk.Label(window, text='Input Layer Size')
label_input_layer.place(x=LABELS_LEFT_ALIGNMENT1, y=depth)
txt_input_layer.insert(0, '5')

# output layer
txt_output_layer = tk.Entry(window, width = 7)
txt_output_layer.place(x=WIDGETS_LEFT_ALIGNMENT1 , y=depth + chunk)
label_output_layer = tk.Label(window, text='Output Layer Size')
label_output_layer.place(x=LABELS_LEFT_ALIGNMENT1, y=depth + chunk)
txt_output_layer.insert(0, '3')

# Neurons per layer
txt_neurons_per_layer = tk.Entry(window, width = 15)
txt_neurons_per_layer.place(x=WIDGETS_LEFT_ALIGNMENT1 , y=depth + chunk*2)
label_neurons_per_layer = tk.Label(window, text='Neurons per Layer')
label_neurons_per_layer.place(x=LABELS_LEFT_ALIGNMENT1 , y=depth + chunk*2)

# Activation function
activation_function = tk.StringVar()
combo_activation_function = ttk.Combobox(window, width=12, values=['Tanh', 'Sigmoid'], textvariable=activation_function)
combo_activation_function.place(x=WIDGETS_LEFT_ALIGNMENT1 , y=depth + chunk*3)
combo_activation_function.current(0)
label_activation_function = tk.Label(window, text='Activation Function')
label_activation_function.place(x=LABELS_LEFT_ALIGNMENT1 , y=depth + chunk*3)

# Bias checkbox
addBias = tk.BooleanVar()
check_bias = tk.Checkbutton(window , variable=addBias)
check_bias.place(x= WIDGETS_LEFT_ALIGNMENT1 , y =depth+chunk*4)
label_bias = tk.Label(window, text='add bias')
label_bias.place(x=LABELS_LEFT_ALIGNMENT1 , y=depth+chunk*4)

# Epochs entry
txt_epochs = tk.Entry(window, width = 10)
txt_epochs.place(x=WIDGETS_LEFT_ALIGNMENT2 , y=depth)
label_epochs = tk.Label(window, text='epochs')
label_epochs.place(x=LABELS_LEFT_ALIGNMENT2 , y=depth)

# Learning rate entry
txt_learning_rate = tk.Entry(window, width = 10)
txt_learning_rate.place(x=WIDGETS_LEFT_ALIGNMENT2 , y=depth+chunk)
label_learning_rate = tk.Label(window, text='learning rate')
label_learning_rate.place(x=LABELS_LEFT_ALIGNMENT2 , y=depth+chunk)

# train/all ratio entry
txt_training_ratio = tk.Entry(window, width = 10)
txt_training_ratio.place(x=WIDGETS_LEFT_ALIGNMENT2 , y=depth+chunk*2)
label_training_ratio = tk.Label(window, text='train/all ratio')
label_training_ratio.place(x=LABELS_LEFT_ALIGNMENT2 , y=depth+chunk*2)

# train button
btn_train = tk.Button(window, command= Train_MLP, text='Train')
btn_train.place(x=WIDGETS_LEFT_ALIGNMENT1-50, y=325, height=25, width=105)

# test button
btn_test = tk.Button(window, command=show_metrics, text='Test')
btn_test.place(x=WIDGETS_LEFT_ALIGNMENT1-50, y=365, height=25, width=105)

# confusion matrix
btn_confusion_matrix = tk.Button(window, command= cm_, text='Confusion Matrix')
btn_confusion_matrix.place(x=WIDGETS_LEFT_ALIGNMENT1-50, y=405, height=25, width=105)

## Metrics
# accuracy
txt_accuracy = tk.Entry(window, width = 15)
txt_accuracy.place(x=WIDGETS_LEFT_ALIGNMENT3 , y=320)   # 330
label_accuracy = tk.Label(window, text='Accuracy')
label_accuracy.place(x=LABELS_LEFT_ALIGNMENT3 , y=320)

# precision
txt_precision = tk.Entry(window, width = 15, )
txt_precision.place(x=WIDGETS_LEFT_ALIGNMENT3 , y=360) #375
label_precision = tk.Label(window, text='Precision')
label_precision.place(x=LABELS_LEFT_ALIGNMENT3 , y=360)

# accuracy
txt_recall = tk.Entry(window, width = 15, )
txt_recall.place(x=WIDGETS_LEFT_ALIGNMENT3 , y=405) #420
label_recall = tk.Label(window, text='Recall')
label_recall.place(x=LABELS_LEFT_ALIGNMENT3 , y=405)

tk.mainloop()
