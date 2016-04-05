#Investigate overfitting and model selection Problem with sin((2pi)x) graph
#2016.03.15
#Seung Hyun Yoon

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.mlab as mlab
import numpy as np
import math
import random

#Init Training set
trsetx = []
trsety = []
#Root mean square for training set
rms_trainning = []
#Root mean square for test set
rms_test = []
#Value of w_star
wStar_training = 0
wStar_test = 0aaaaaaa
#Temp value for calculate RMS
coefficient_temp = []
polynomial_temp = []

#Number of Training data
numOfTrset = int(input("Training Data size: "))

#Order of polynomial Function
order = int(input("Order of polynomial Function: "))

#Generate x cordinate value uniformly
trsetx = np.linspace(0, 1, numOfTrset)

#Gererate noise(having gaussian distribution)
noise = np.random.normal(0, 0.3, numOfTrset)

#Making training Set
for i in range (0, numOfTrset):
	#Generate training data
	#x_tr = round(random.random(),4) #Random number generation for x cordinate
	trdata = round(np.sin((2*np.pi)*trsetx[i]) + noise[i],4)
	#Insert training data into list
	#trsetx.append(x_tr)
	trsety.append(trdata)

#Calculate RMS for Training set and Test set
for i in range (0, order+1):
	coefficient_temp = np.polyfit(trsetx, trsety, i)
	polynomial_temp = np.poly1d(coefficient_temp)
	for j in range (0, numOfTrset):
		wStar_training += math.pow((polynomial_temp(trsetx[j])-trsety[j]),2)
	#Sum RMS 1000 times for calculating Erms in test set
	for k in range (0, 1000):
		x_test = round(random.random(),4)
		wStar_test += math.pow((polynomial_temp(x_test)-np.sin((2*np.pi)*x_test)),2)
	rms_trainning.append(math.sqrt(wStar_training/numOfTrset))
	rms_test.append(math.sqrt(wStar_test/1000))
	wStar_training = 0
	wStar_test = 0

#Find optimal graph for test set
optimal_order = rms_test.index(min(rms_test))
print("-----------------------------------")
print("The Optimal polynomial order: ", optimal_order)
print("-----------------------------------")

#For visualize curve fitting order of first setting value
coefficients = np.polyfit(trsetx, trsety, order)
polynomial = np.poly1d(coefficients)
xs = np.arange(0, 1.01, 0.01)
ys = polynomial(xs)

#For visualize optimal curve fitting graph
optimal_coefficients = np.polyfit(trsetx, trsety, optimal_order)
optimal_polynomial = np.poly1d(optimal_coefficients)
xs2 = np.arange(0, 1.01, 0.01)
ys2 = optimal_polynomial(xs2)

#First window
plt.figure(1)
#Original sin2pi graph
t=np.linspace(0, 1, 100)
y1=np.sin(2*np.pi*t)
plt.plot(t,y1)
#Training Datas
plt.plot(trsetx,trsety,'rs')
#Curve fitting graph
plt.plot(xs,ys)
Title_patch1 = mpatches.Patch(color='green', label= str(order) + 'th Order Curve Fitting Graph')
plt.legend(handles=[Title_patch1])

#Second window
plt.figure(2)
#Original sin2pi graph
t=np.linspace(0, 1, 100)
y1=np.sin(2*np.pi*t)
plt.plot(t,y1)
#Training Datas
plt.plot(trsetx,trsety,'rs')
#Curve fitting graph
plt.plot(xs2,ys2)
Title_patch2 = mpatches.Patch(color='green', label=str(optimal_order) + 'th Optimal Curve Fitting Graph')
plt.legend(handles=[Title_patch2])

#Third window
plt.figure(3)
#Draw RMS graph
k=np.linspace(0, order, order+1)
plt.plot(k,rms_trainning,'b')
plt.plot(k,rms_test,'r')
plt.ylabel('E_rms')
plt.xlabel('Orders (M)')
red_patch = mpatches.Patch(color='red', label='Test Set')
blue_patch = mpatches.Patch(color='blue', label='Training Set')
plt.legend(handles=[red_patch, blue_patch])

#Draw all graphs
plt.show()

