import pandas as pd
import sklearn as sk
import numpy as np
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import splitter
import SLP
from Adaline import *
import metrics
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("final_file.csv")


# __ GUI Functionalitites __ #
def train_when_clicked():
    global w1, w2, b, accuracy, EPSILON, C_F, X_train, X_test, y_train, y_test
    feat1 = cmb_feature1.get()
    feat2 = cmb_feature2.get()
    class1 = cmb_class1.get()
    class2 = cmb_class2.get()
    alpha = float(txt_learning_rate.get())
    bias = var.get()
    model = radio_button_state.get()
    epochs = None
    EPSILON = None

    if model == 1: ## perceptron
        epochs = int(txt_epochs.get())
    else:          ## adaline
        EPSILON = float(txt_MSE_threshold.get())

    X_train, X_test, y_train, y_test = splitter.split_dataset(data, class1, class2, feat1, feat2, 30)

    if model == 1:
        _model = SLP.Perceptron(learning_rate= alpha, epochs= epochs, bias_flag= bias)
        _model.fit(X_train, y_train)
        w1, w2, b = _model.weights[0], _model.weights[1], _model.bias
        predictions = _model.predict(X_test).reshape(40, 1)
        C_F = metrics.confusion_matrix(y_test, predictions)
        accuracy = metrics.accuracy(y_test, predictions)

    if model == 2:
        _model = Adaline(learning_rate= alpha, EPSILON= EPSILON, include_bias= bias)
        _model.fit(X_train, y_train)
        w1, w2, b = _model.get_weights()
        print(_model.get_weights())


def show_evaluation():
    metric.delete(0, END)
    if radio_button_state.get() == 1: # SLP metric
        metric.insert(0, str(accuracy))
    else: # Adaline metric
        pass
        # metrics.mean_squared_error(y_test, )   # Mahmoud should calculate the metric here

def show_confusion_matrix():
    plt.figure(figsize=(8, 6))
    sns.heatmap(C_F, annot=True, cmap='coolwarm', fmt='g', cbar=False, linewidths=0.8)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix', fontsize=20)
    plt.show()


def test():
    x1 = float(txt_testVal1.get())
    x2 = float(txt_testVal2.get())
    selectclass1 = cmb_class1.get()
    selectclass2 = cmb_class2.get()
    net = (x1 * w1) + (x2 * w2) + b

    if net >= 0:
        txt_result.delete(0, END)
        txt_result.insert(0, selectclass1)
    else:
        txt_result.delete(0 , END)
        txt_result.insert(0 , selectclass2)


def decision_boundary(X_train, X_test, y_train, y_test, w1, w2, b):
    min_ = min(X_train[:, 0].min(), X_test[:, 0].min())
    max_ = max(X_train[:, 0].max(), X_test[:, 0].max())

    x_range = np.linspace(min_, max_,  100)
    y_values = (-w1 / w2) * x_range - (b / w2)
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])
    plt.figure()
    plt.plot(x_range, y_values, color='black', linewidth=2, label='Decision Boundary')
    plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='BOMBAY (Train)')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='CALI (Train)')
    plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='lightblue', label='BOMBAY (Test)')
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='pink', label='CALI (Test)')
    plt.xlabel('Feature 1')  
    plt.ylabel('Feature 2')  
    plt.legend(loc='upper left')
    plt.show()


# __ GUI Form __ #

# Initialize the form
form = Tk()
form.geometry("600x700+500+150")
form.resizable(False , False)
form.title('Dry Beans')
form.config(background='lightblue')
form.iconbitmap('./icon.ico')
background_color = 'lightblue'
feature1_name = StringVar()
feature2_name = StringVar()

### Selecting Features:
# Feature 1 Label
label_feature1 = Label(form , text='feature 1' , fg='black' , bg=background_color)
label_feature1.place(x= 20 , y= 30)

# Feature 1 Text Box
cmb_feature1 = ttk.Combobox(form , value=('Area' , 'Perimeter' , 'MajorAxisLength' , 'MinorAxisLength' , 'roundnes') ,textvariable=feature1_name)
cmb_feature1.place(x = 80 , y= 30)

# Feature 2 Label
label_class2 = Label(form , text= 'feature 2' , fg='black' , bg= background_color)
label_class2.place(x= 300 , y= 30)

# Feature 2 Text Box
cmb_feature2 = ttk.Combobox(form , value=('Area' , 'Perimeter' , 'MajorAxisLength' , 'MinorAxisLength' , 'roundnes') , textvariable= feature2_name)
cmb_feature2.place(x = 370 , y= 30)


### Selecting Classes:
# Class1 Label
label_class1 = Label(form , text= 'Class 1' , fg='black' , bg= background_color)
label_class1.place(x= 20 , y= 80)

# Class1 Combe Box
cmb_class1 = ttk.Combobox(form , value=('BOMBAY' , 'CALI' , 'SIRA') )
cmb_class1.place(x = 80 , y= 80)

# Class2 Label
label_class2 = Label(form , text= 'Class 2' , fg='black' , bg= background_color)
label_class2.place(x= 300 , y= 80)

# Class2 Combe Box
cmb_class2 = ttk.Combobox(form , value=('BOMBAY' , 'CALI' , 'SIRA'))
cmb_class2.place(x = 370 , y= 80)


### Input: learning rate, #epochs, and MSE_threshold
# Learning Rate Label
label_learning_rate = Label(form , text= 'Learning rate' , fg='black' , bg= background_color)
label_learning_rate.place(x= 20 , y= 130)

# Learning Rate Text Box
txt_learning_rate = Entry(form , justify= 'center' , width= 10)
txt_learning_rate.place(x= 120 , y= 130)

# #Epochs Label
label_epochs = Label(form , text= '#Epochs' , fg='black' , bg= background_color)
label_epochs.place(x= 300 , y= 130)

# #Epochs Text Box
txt_epochs = Entry(form , justify= 'center' , width= 10)
txt_epochs.place(x= 370 , y= 130)

# MSE Threshold Label
label_MSE_threshold = Label(form , text= 'MSE threshold' , fg='black' , bg= background_color)
label_MSE_threshold.place(x= 20 , y= 180)

# MSE Threshold Text Box
txt_MSE_threshold = Entry(form , justify= 'center' , width= 15)
txt_MSE_threshold.place(x= 120 , y= 180)


### Add Bias?
# Bias Label
var = BooleanVar()
label_bias = Label(form , text= 'Biss' , fg='black' , bg= background_color)
label_bias.place(x= 40 , y= 230)

# Bias Check Box
check_bias = Checkbutton(form , bg='lightblue', variable=var)
check_bias.place(x= 80 , y =230)


### Choosing a Model
radio_button_state = IntVar()

# Preceptron Label
label_SLP = Label(form , text= 'SLP' , fg='black' , bg= background_color)
label_SLP.place(x= 170 , y= 260)

# Perceptron Radio Button
radio_SLP = Radiobutton(form , value=1 ,bg= background_color, anchor='w' , variable=radio_button_state)
radio_SLP.place(x= 200 , y=260)

# Adaline Label
label_adaline = Label(form , text= 'Adaline' , fg='black' , bg= background_color)
label_adaline.place(x= 330 , y= 260)

# Adaline Radio Button
radio_adaline = Radiobutton(form , value=2 ,bg= background_color, anchor='w' , variable=radio_button_state)
radio_adaline.place(x= 380 , y= 260)


### Trainging Button
btn_train = Button(form , text= 'Train' , fg= 'black' , bg= 'white' , width= 20, command=train_when_clicked)
btn_train.place(x= 200 , y= 320)


### Testing 
# Test Value 1 Label
label_testVal1 = Label(form, textvariable= feature1_name , fg='black' , bg= background_color )
label_testVal1.place(x= 20 , y= 400)

# Test Value 1 Text Box
txt_testVal1 = Entry(form , justify= 'center' , width= 15)
txt_testVal1.place(x= 140 , y= 400)

# Test Value 2 Label
label_testVal2 = Label(form, textvariable= feature2_name , fg='black' , bg= background_color )
label_testVal2.place(x= 300 , y= 400)

# Test Value 2 Text Box
txt_testVal2 = Entry(form , justify= 'center' , width= 15)
txt_testVal2.place(x= 420 , y= 400)

# Test button
btn_test = Button(form , text= 'Test' , fg= 'black' , bg= 'white' , width= 20, command=test)
btn_test.place(x= 110 , y= 450)

# Result Label
label_result = Label(form , text= 'Classified as:' , fg= 'black' , bg= background_color)
label_result.place(x= 300 , y= 455)

# Result Text Box (only for showing)
txt_result = Entry(form , justify= 'center' , width= 15)
txt_result.place(x= 380 , y= 455)


### Metrics:
# Show Decision Boundary Button
btn_decision_Boundary = Button(form, text='Show Decision Boundary', fg='black', bg='white', width=30, command=lambda: decision_boundary(X_train, X_test, y_train, y_test, w1, w2, b))
btn_decision_Boundary.place(x = 70, y = 630)

# Show Confusion Matrix Button
btn_CF = Button(form, text='Confusion Matrix', fg='black', bg='white', width=30, command= show_confusion_matrix)
btn_CF.place(x = 300, y = 630)

# Metric Label
label_MSE_accuracy = Label(form , text= 'MSE/Accuracy' , fg='black' , bg= background_color)
label_MSE_accuracy.place(x= 150 , y= 580)

# Metric Text Box (only for showing)
metric = Entry(form , justify= 'center' , width= 15)
metric.place(x= 240 , y= 580)

# Evaluation Button
btn_evalute = Button(form, text='evaluate', fg='black', bg='white', width=15, command=show_evaluation)
btn_evalute.place(x = 350, y = 580)

form.mainloop()
