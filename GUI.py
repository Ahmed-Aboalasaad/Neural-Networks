import pandas as pd
import sklearn as sk
import numpy as np
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import extra_functions
import SLP
from Adaline import *
import metrics
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("final_file.csv")


####
def train_when_clicked():
    global w1, w2, b, accuracy, mse, C_F
    feat1 = CmbFt1.get()
    feat2 = CmbFt2.get()
    class1 = Cmbcls1.get()
    class2 = Cmbcls2.get()
    alpha = float(TxtLr.get())
    bias = var.get()
    model = radio_button_state.get()
    epochs = None
    EPSILON = None
    
    if model == 1: ## perceptron
        epochs = int(Txtep.get())
    else:          ## adaline
        EPSILON = float(TxtMse.get())

    X_train, X_test, y_train, y_test = extra_functions.split_dataset(data, class1, class2, feat1, feat2, 30)
    
    if model == 1:
        _model = SLP.Perceptron(learning_rate= alpha, epochs= epochs, bias_flag= bias)
        _model.fit(X_train, y_train)
        w1, w2, b = _model.weights[0], _model.weights[1], _model.bias
        predictions = _model.predict(X_test).reshape(40, 1)
        C_F = metrics.confusion_matrix(y_test, predictions)
        accuracy = metrics.accuracy(y_test, predictions)

    if model == 2:  ## ignore until Adaline is fixed
        pass
        # _model = Adaline(learning_rate= alpha, EPSILON= EPSILON, include_bias= bias)
        # _model.fit(X_train, y_train)
        # w1, w2, b = _model.get_weights()
        # print(_model.get_weights())



def show_evaluation():
    MSE_OR_ACCURACY.delete(0, END)
    MSE_OR_ACCURACY.insert(0, str(accuracy))

def show_confusion_matrix():
    plt.figure(figsize=(8, 6))
    sns.heatmap(C_F, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

###


#form
m = Tk()
m.geometry("600x700+500+150")
m.resizable(False , False)
m.title('Dry Beans')
m.config(background='lightblue')
#m.iconbitmap('D:\\university\Year 3\\Semester 2\\Neural Network\\Section\\Lab 3\\assignment\\icon.ico')


strbg = 'lightblue'
SelectItem1 = StringVar()
SelectItem2 = StringVar()


#features

#label feature 1
ft1 = Label(m , text= 'feature 1' , fg='black' , bg= strbg)
ft1.place(x= 20 , y= 30)

#combo Box feature 1
CmbFt1 = ttk.Combobox(m , value=('Area' , 'Perimeter' , 'MajorAxisLength' , 'MinorAxisLength' , 'roundnes') ,textvariable= SelectItem1 )
CmbFt1.place(x = 80 , y= 30)

#label feature 2
ft2 = Label(m , text= 'feature 2' , fg='black' , bg= strbg)
ft2.place(x= 300 , y= 30)

#combo Box feature 2
CmbFt2 = ttk.Combobox(m , value=('Area' , 'Perimeter' , 'MajorAxisLength' , 'MinorAxisLength' , 'roundnes') , textvariable= SelectItem2)
CmbFt2.place(x = 370 , y= 30)



#classes

#label class 1
cls1 = Label(m , text= 'Class 1' , fg='black' , bg= strbg)
cls1.place(x= 20 , y= 80)

#combo Box class 1
Cmbcls1 = ttk.Combobox(m , value=('BOMBAY' , 'CALI' , 'SIRA') )
Cmbcls1.place(x = 80 , y= 80)

#label class 2
ft2 = Label(m , text= 'Class 2' , fg='black' , bg= strbg)
ft2.place(x= 300 , y= 80)

#combo Box class 2
Cmbcls2 = ttk.Combobox(m , value=('BOMBAY' , 'CALI' , 'SIRA'))
Cmbcls2.place(x = 370 , y= 80)



#learning rate , epoch and MSE threshold

#label learning rate
lr = Label(m , text= 'Learning rate' , fg='black' , bg= strbg)
lr.place(x= 20 , y= 130)

#Text box learning rate
TxtLr = Entry(m , justify= 'center' , width= 10)
TxtLr.place(x= 120 , y= 130)

#label Epoch
ep = Label(m , text= 'Epoch' , fg='black' , bg= strbg)
ep.place(x= 300 , y= 130)

#Text box Epoch
Txtep = Entry(m , justify= 'center' , width= 10)
Txtep.place(x= 370 , y= 130)

#label MSE threshold
Mse = Label(m , text= 'MSE threshold' , fg='black' , bg= strbg)
Mse.place(x= 20 , y= 180)

#Text box MSE threshold
TxtMse = Entry(m , justify= 'center' , width= 15)
TxtMse.place(x= 120 , y= 180)


#Bias

#label Bias
var = BooleanVar()
bias = Label(m , text= 'Bais' , fg='black' , bg= strbg)
bias.place(x= 40 , y= 230)

#Check Box Bias 
chbBias = Checkbutton(m , bg='lightblue', variable=var)
chbBias.place(x= 80 , y =230)


#preceptron and Adaline

radio_button_state = IntVar()

#label Preceptron
pre = Label(m , text= 'Preceptron' , fg='black' , bg= strbg)
pre.place(x= 170 , y= 260)

#radio button preceptron
RadPre = Radiobutton(m , value=1 ,bg= strbg, anchor='w' , variable=radio_button_state)
RadPre.place(x= 240 , y=260)

#label Adaline
pre = Label(m , text= 'Adaline' , fg='black' , bg= strbg)
pre.place(x= 330 , y= 260)

#radio button Adaline
RadAda = Radiobutton(m , value=2 ,bg= strbg, anchor='w' , variable=radio_button_state)
RadAda.place(x= 380 , y= 260)

#button Training
trnBtn = Button(m , text= 'Training' , fg= 'black' , bg= 'white' , width= 20, command=train_when_clicked)
trnBtn.place(x= 200 , y= 320)



#Input 2 Features

#label input feature 1
InFt1 = Label(m, textvariable= SelectItem1 , fg='black' , bg= strbg )
InFt1.place(x= 20 , y= 400)

#Textbox input feature 1
Txtft1 = Entry(m , justify= 'center' , width= 15)
Txtft1.place(x= 140 , y= 400)

#label input feature 2
InFt2 = Label(m, textvariable= SelectItem2 , fg='black' , bg= strbg )
InFt2.place(x= 300 , y= 400)

#Textbox input feature 2
Txtft2 = Entry(m , justify= 'center' , width= 15)
Txtft2.place(x= 420 , y= 400)

#Test button
testBtn = Button(m , text= 'Test' , fg= 'black' , bg= 'white' , width= 20)
testBtn.place(x= 200 , y= 450)


#class result

#label class result
res = Label(m , text= 'Class result' , fg= 'black' , bg= strbg)
res.place(x= 40 , y= 500)

#Textbox class result
Txtclsres = Entry(m , justify= 'center' , width= 15 , state= 'readonly')
Txtclsres.place(x= 120 , y= 500)



Decision_Boundary = Button(m, text='Show Decision Boundary', fg='black', bg='white', width=30)
Decision_Boundary.place(x = 70, y = 630)


CF = Button(m, text='Show Confusion Matrix', fg='black', bg='white', width=30, command= show_confusion_matrix)
CF.place(x = 300, y = 630)


#label Adaline
_label = Label(m , text= 'MSE/Accuracy' , fg='black' , bg= strbg)
_label.place(x= 150 , y= 580)
#Text box MSE threshold
MSE_OR_ACCURACY = Entry(m , justify= 'center' , width= 15)
MSE_OR_ACCURACY.place(x= 240 , y= 580)

ACC = Button(m, text='acc/mse', fg='black', bg='white', width=15, command=show_evaluation)
ACC.place(x = 350, y = 580)

m.mainloop()

