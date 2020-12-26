import matplotlib.pyplot as plt
import random
import numpy as np
import math as m
import random

colors = ['b','g','r','c','m','y','indigo','lime','aqua','peru']

def readfile(fn):
    f = open(fn,'rt')
    H = []
    Y = []
    Yc = []
    for line in f:
        parts = line.split(',')
        pn = int(parts[0])

        tmp = [1.0]
        for i in range(512):
            t = float(parts[1+i])
            tmp.append(t)
        H.append(tmp)
        Y.append(pn)   
        Yc.append(colors[pn])
    f.close()
    
    return np.array(H),Y,Yc

H,Y,Yc = readfile('feats_train.txt')
H_t,Y_t,Yc_t = readfile('feats_test.txt')

Ht = np.transpose(H)
HtH = np.matmul(Ht,H)
iHtH = np.linalg.inv(HtH)
print(iHtH)
iHtHHt = np.matmul(iHtH,Ht)
print(iHtHHt.shape)

maxcor = 0
while True:
    trans = [-1 for i in range(10)]

    for i in range(10):
        r = random.randrange(10)
        while trans[r] != -1:
            r = random.randrange(10)
        trans[r] = i
    print(trans)

    YY = []
    for i in range(len(Y)):
        w = trans[Y[i]]
        tmp = [w]
        YY.append(tmp)

    YY = np.array(YY)
    x = np.matmul(iHtHHt,YY)

    Ys = np.matmul(H,x)
    plt.subplot(211)
    plt.scatter(Ys[:,0], np.random.rand(50000,1), s=0.2, c=Yc)
    
    Ys_t = np.matmul(H_t,x)
    plt.subplot(212)
    plt.scatter(Ys_t[:,0], np.random.rand(10000,1), s=0.2, c=Yc_t)
    
    plt.show()
