# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 00:23:21 2020

@author: DELLRJ5999
"""

"""All the lib which are needed for the project"""
import anfis
import mfDerivs
import membershipfunction
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import skfuzzy
import pandas as pd

""".csv file which has the dataset to work upon"""
dataset=pd.read_csv('lung_cancer_examples.csv')

"""making a copy of dataset"""
datase=dataset[:]
"""checking for top 5 rows of dataset"""
dataset.head()

"""checking all the unique values in col of dataset"""
datase["Result"].unique()
datase["Alkhol"].unique()
datase["Smokes"].unique()
datase["AreaQ"].unique()
datase["Age"].unique()

"""extracting the result col as target"""
target=dataset.pop('Result')

"""droping the useless cols"""
dataset=dataset.drop("Name",axis=1)
dataset=dataset.drop("Surname",axis=1)

#dataset=dataset.loc[:,["Smokes","Alkhol"]]  

"""checking shape of the target and dataset"""
target.shape
dataset.shape

"""ploting a 3d grapical representation of cols in dataset and 
checking for result values in 1's for white and 0's for black."""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.asarray(datase["Alkhol"])
y = np.asarray(datase["Smokes"])
z = np.asarray(datase["AreaQ"])
c = np.asarray(datase["Result"])
img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
fig.colorbar(img)
plt.show()

"""preparing membership function """
mf = [[['gaussmf',{'mean':0.,'sigma':1.}],['gaussmf',{'mean':-1.,'sigma':2.}],['gaussmf',{'mean':-4.,'sigma':10.}],['gaussmf',{'mean':-7.,'sigma':7.}]],
            [['gaussmf',{'mean':1.,'sigma':2.}],['gaussmf',{'mean':2.,'sigma':3.}],['gaussmf',{'mean':-2.,'sigma':10.}],['gaussmf',{'mean':-10.5,'sigma':5.}]],
            [['gaussmf',{'mean':0.,'sigma':1.}],['gaussmf',{'mean':-1.,'sigma':2.}],['gaussmf',{'mean':-4.,'sigma':10.}],['gaussmf',{'mean':-7.,'sigma':7.}]],
            [['gaussmf',{'mean':1.,'sigma':2.}],['gaussmf',{'mean':2.,'sigma':3.}],['gaussmf',{'mean':-2.,'sigma':10.}],['gaussmf',{'mean':-10.5,'sigma':5.}]]]


mfc=membershipfunction.MemFuncs(mf)

"""training of the model"""
anf=anfis.ANFIS(dataset, target, mfc)

out=anf.trainHybridJangOffLine(epochs=10)

"""graphical plot of errors"""
anf.plotErrors()

"""actual vs trained graph"""
anf.plotResults()


tt=np.asarray(target)

"""cleaning of out values"""
ll=[]
for i in out:
    if(i<0.5):
        ll.append(0)
    else:
        ll.append(1)
        
ll1=np.asarray(ll)

"""confusion matrix"""
from sklearn.metrics import confusion_matrix
cc=confusion_matrix(ll1,tt)

