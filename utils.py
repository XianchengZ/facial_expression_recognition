# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def getData(balance_ones=True):
    df = pd.read_csv("fer2013.csv")
    df['pixels']  = df['pixels'].apply(lambda x: [int(p) for p in x.split()])
    
    X = np.array(df['pixels'].tolist())
    Y = np.array(df['emotion'])
    
    # get the row number and column number from X.shape
    row, col = X.shape
    train_test_split = int(row / 3 * 2) # get the first 2/3 to be train
    # shuffle and split
    X,Y = shuffle(X, Y)
    Xtrain, Ytrain = X[:train_test_split], Y[:train_test_split]
    Xtest, Ytest = X[train_test_split:], Y[train_test_split:]
    
    return Xtrain, Ytrain, Xtest, Ytest

def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)

def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


def cost2(T, Y):
    N = len(T)
    return -np.log(Y[np.arange(N), T]).mean()

def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def error_rate(targets, predictions):
    return np.mean(targets != predictions)