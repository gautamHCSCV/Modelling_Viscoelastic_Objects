import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os
from collections import defaultdict
import bisect
import warnings
warnings.filterwarnings('ignore')


X = pd.read_csv('dataset1/inp_z_final.csv', names = ['{}'.format(i) for i in range(5)])
X_test = pd.read_csv('dataset1/inp_z_final_test.csv', names = ['{}'.format(i) for i in range(5)])
Y = pd.read_csv('dataset1/out_gk_final.csv', names = ['{}'.format(i) for i in range(2)])
Y_test = pd.read_csv('dataset1/out_gk_final_test.csv', names = ['{}'.format(i) for i in range(2)])
X.head()

from scipy import *
from scipy.linalg import norm, pinv
import math
from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt

# y = Wx + P
class RBF:
     
    def __init__(self, indim, outdim, center_inds):
        self.indim = indim
        self.center_inds = center_inds
        self.outdim = outdim
        self.numCenters = len(self.center_inds)
        self.centers = [np.random.uniform(-1, 1, indim) for i in range(self.numCenters)]
        self.W = np.random.random((self.numCenters, self.outdim))
        self.P = LinearRegression()
        self.lr = LinearRegression()
         
    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return norm(c-d)**1.5
     
    def _calcAct(self, X):
        # calculate activations of RBFs
        G = np.zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G
     
    def train(self, X, Y):
        """ X: matrix of dimensions n x indim 
            y: column vector of dimension n x 1 """
         
        # choose random center vectors from training set
        self.P.fit(X,Y)
        rnd_idx = self.center_inds
        self.centers = [X[i,:] for i in rnd_idx]
         
        # print("center", self.centers)
        # calculate activations of RBFs
        G = self._calcAct(X)
        # print(G) 
        # calculate output weights (pseudoinverse)
        self.W = np.dot(pinv(G), Y)
        val = np.zeros((len(X),2))
        val[:,0] = np.dot(G, self.W)
        val[:,1] = self.P.predict(X)
        # print(val.shape)
        self.lr.fit(val,Y)
        
         
    def test(self, X):
        """ X: matrix of dimensions n x indim """
         
        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        Y2 = self.P.predict(X)
        val = np.zeros((len(X),2))
        val[:,0] = Y
        val[:,1] = Y2
        return self.lr.predict(val)
    
def FBL(value):
    start,end = bisect.bisect_left(Y.iloc[:,0],value), bisect.bisect_right(Y.iloc[:,0],value)
    x = np.array(X.iloc[start:end,1:5])
    y = np.array(Y.iloc[start:end,1])
    
    centers_ind = [1]
    got = defaultdict(lambda : 1)
    got[1]=0
    if 1 == 1:
        rbf = RBF(4, 1,centers_ind)
        rbf.train(x, y)

        y1 = rbf.test(x)
        maxi,ind = 0,0
        for i in range(len(y)):
            if i not in centers_ind and norm(y[i]-y1[i])/(1+norm(y[i]))>maxi:
                maxi = norm(y[i]-y1[i])/(1+norm(y[i]))
                ind = i
        p = 0.7
        error = mean_squared_error(y,y1)**p
        while error<0.15 and p>0:
            p -= 0.02
            error = mean_squared_error(y,y1)**p
        while error>0.15 and len(centers_ind)<len(y)//5:
            centers_ind.append(ind)
            for j in range(ind-3,ind+3):
                got[j]=0
            rbf = RBF(4, 1, centers_ind)
            rbf.train(x,y)

            y1 = rbf.test(x)

            maxi,ind = 0,0
            for i in range(len(y)):
                if got[i] and norm(y[i]-y1[i])/(1+norm(y[i]))>maxi:
                    maxi = norm(y[i]-y1[i])/(1+norm(y[i]))
                    ind = i
            error = mean_squared_error(y,y1)**p
            #print(error,ind)
        #print(centers_ind)
        return centers_ind,start
    
    
Final_indices = []
important_interactions = [2, 79, 91, 86, 82, 100, 11, 103, 7, 23, 26, 3, 19, 95, 117, 107, 112, 35, 13, 6]

for i in important_interactions:
    points, val = FBL(i)
    print(len(points), end = ' ')
    for j in points:
        Final_indices.append(j+val)
        
        
class RBF1:
     
    def __init__(self, indim,outdim, centers_ind):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = len(centers_ind)
        self.centers_ind = centers_ind
        self.centers = [np.random.uniform(-1, 1, indim) for i in range(self.numCenters)]
        self.beta = 8
        self.W = np.random.random((self.numCenters, self.outdim))
         
    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return norm(c-d)**3
     
    def _calcAct(self, X):
        # calculate activations of RBFs
        G = np.zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G
     
    def train(self, X, Y):
        """ X: matrix of dimensions n x indim 
            y: column vector of dimension n x 1 """
         
        # choose random center vectors from training set
        rnd_idx = self.centers_ind
        self.centers = [X[i,:] for i in rnd_idx]
         
        #print("center", self.centers)
        # calculate activations of RBFs
        G = self._calcAct(X)
        #print(G)
         
        # calculate output weights (pseudoinverse)
        self.W = np.dot(pinv(G), Y)
         
    def test(self, X):
        """ X: matrix of dimensions n x indim """
         
        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y
    
    
    
x = np.array(X.iloc[:,1:5])
y = np.array(Y.iloc[:,1])

rbf = RBF1(4, 1,Final_indices)
rbf.train(x, y)

x_test = np.array(X_test.iloc[:,1:5])
y_test = np.array(Y_test.iloc[:,1])
y1 = rbf.test(x_test)
print('\nRoot Mean Squared error: ', mean_squared_error(y_test,y1)**0.5)

import matplotlib.pyplot as plt
plt.figure(figsize=(20,5))
plt.plot(list(range(6000)),y1[:6000], label = 'FBL')
plt.plot(list(range(6000)),y_test[:6000], label = 'Actual')
plt.legend()
plt.xlabel('Time (in ms)')
plt.ylabel('Force')
plt.title('RBF based on FBL')
plt.savefig('fbl1.png')
plt.savefig('fbl1.pdf')
plt.show()


