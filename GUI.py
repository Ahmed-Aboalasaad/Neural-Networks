import pandas as pd
import sklearn as sk
import numpy as np
from tkinter import *
from tkinter import messagebox
from tkinter import ttk

#form
m = Tk()
m.geometry("550x550+500+150")
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
CmbFt1 = ttk.Combobox(m , value=('Area' , 'Perimeter' , 'MajorAxisLenght' , 'MinorAxisLenght' , 'Roundness') ,textvariable= SelectItem1 )
CmbFt1.place(x = 80 , y= 30)

#label feature 2
ft2 = Label(m , text= 'feature 2' , fg='black' , bg= strbg)
ft2.place(x= 300 , y= 30)

#combo Box feature 2
CmbFt2 = ttk.Combobox(m , value=('Area' , 'Perimeter' , 'MajorAxisLenght' , 'MinorAxisLenght' , 'Roundness') , textvariable= SelectItem2)
CmbFt2.place(x = 370 , y= 30)



#classes

#label class 1
cls1 = Label(m , text= 'Class 1' , fg='black' , bg= strbg)
cls1.place(x= 20 , y= 80)

#combo Box class 1
Cmbcls1 = ttk.Combobox(m , value=('Bombay' , 'Cali' , 'Sira') )
Cmbcls1.place(x = 80 , y= 80)

#label class 2
ft2 = Label(m , text= 'Class 2' , fg='black' , bg= strbg)
ft2.place(x= 300 , y= 80)

#combo Box class 2
Cmbcls2 = ttk.Combobox(m , value=('Bombay' , 'Cali' , 'Sira'))
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
bias = Label(m , text= 'Bais' , fg='black' , bg= strbg)
bias.place(x= 40 , y= 230)

#Check Box Bias 
chbBias = Checkbutton(m , bg='lightblue')
chbBias.place(x= 80 , y =230)


#preceptron and Adaline

#label Preceptron
pre = Label(m , text= 'Preceptron' , fg='black' , bg= strbg)
pre.place(x= 170 , y= 260)

#radio button preceptron
RadPre = Radiobutton(m , value=1 ,bg= strbg )
RadPre.place(x= 240 , y=260)

#label Adaline
pre = Label(m , text= 'Adaline' , fg='black' , bg= strbg)
pre.place(x= 330 , y= 260)

#radio button Adaline
RadPre = Radiobutton(m , value=2 ,bg= strbg )
RadPre.place(x= 380 , y= 260)

#button Training
trnBtn = Button(m , text= 'Training' , fg= 'black' , bg= 'white' , width= 20)
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

m.mainloop()