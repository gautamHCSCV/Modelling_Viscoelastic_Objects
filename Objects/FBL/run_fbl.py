import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os
from datetime import datetime
from collections import defaultdict
import bisect
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt
from math import sqrt
from catboost import CatBoostRegressor


for x in range(1,10):
    print("\n\nObject{}".format(x))
    path = 'FBL/Object{}/Deform/'.format(x)


    X1_train = pd.read_csv(path+'afbl_x1_d_train_{}.csv'.format(x))
    X2_train = pd.read_csv(path+'afbl_x2_d_train_{}.csv'.format(x))
    X1_test = pd.read_csv(path+'afbl_x1_d_test_{}.csv'.format(x))
    X2_test = pd.read_csv(path+'afbl_x2_d_test_{}.csv'.format(x))

    Y_train = pd.read_csv(path+'afbl_f_d_train_{}.csv'.format(x))
    Y_test = pd.read_csv(path+'afbl_f_d_test_{}.csv'.format(x))

    print(X1_train.shape, Y_train.shape, X2_test.shape)

    x_train = np.array(pd.concat((X1_train,X2_train),axis=1))
    x_test = np.array(pd.concat((X1_test,X2_test),axis=1))
    y_train,y_test = np.array(Y_train),np.array(Y_test)
    print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

    from scipy import *
    from scipy.linalg import norm, pinv
    import math

    class RBF:

        def __init__(self, indim, numCenters, outdim):
            self.indim = indim
            self.outdim = outdim
            self.numCenters = numCenters
            self.centers = [np.random.uniform(-1, 1, indim) for i in range(numCenters)]
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
            rnd_idx = np.random.permutation(X.shape[0])[:self.numCenters]
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


    t1 = datetime.now()

    rbf = RBF(2,100, 1)
    rbf.train(x_train, y_train)
    t2 = datetime.now()

    y1 = rbf.test(x_test)
    print('time taken by FBL:', t2-t1)
    print('Root Mean Squared error: ', mean_squared_error(y_test,y1)**0.5)
    
    statement1 = 'time taken by FBL: {} \nRoot Mean Squared error: {}\n'.format(t2-t1, mean_squared_error(y_test,y1)**0.5)
    

    cat = CatBoostRegressor()
    t1 = datetime.now()
    cat.fit(x_train,y_train)
    t2 = datetime.now()

    y2 = cat.predict(x_test)
    print('time taken by Catboost on FBL data:', t2-t1)
    print('Root Mean Squared error: ', mean_squared_error(y_test,y2)**0.5)
    
    statement2 = 'time taken by catboost: {} \nRoot Mean Squared error: {}\n'.format(t2-t1, mean_squared_error(y_test,y2)**0.5)
    
    file1 = open("FBL_Results.txt","a")
    file1.write("\n Object {} \n".format(x))
    file1.write(statement1)
    file1.write(statement2)
    file1.close()

