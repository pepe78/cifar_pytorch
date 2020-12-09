import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np
import math as m
import random

def getBetterP():
    p = np.random.rand(10,3)*1.0-0.5
    while True:
        dp = np.zeros((10,3))
        for i in range(10):
            tmp = 2.0 * (p[i,0] * p[i,0] + p[i,1] * p[i,1] + p[i,2] * p[i,2] - 1.0)
            dp[i,0] += tmp * 2.0 * p[i,0]
            dp[i,1] += tmp * 2.0 * p[i,1]
            dp[i,2] += tmp * 2.0 * p[i,2]
            
        for i in range(10):
            for j in range(i+1,10):
                tmp = ((p[i,0]-p[j,0])**2) + ((p[i,1]-p[j,1])**2) + ((p[i,2]-p[j,2])**2)
                
                dp[i,0] +=  -1.0 / (tmp**2) * 2.0 * (p[i,0]-p[j,0])
                dp[i,1] +=  -1.0 / (tmp**2) * 2.0 * (p[i,1]-p[j,1])
                dp[i,2] +=  -1.0 / (tmp**2) * 2.0 * (p[i,2]-p[j,2])
                
                dp[j,0] +=  1.0 / (tmp**2) * 2.0 * (p[i,0]-p[j,0])
                dp[j,1] +=  1.0 / (tmp**2) * 2.0 * (p[i,1]-p[j,1])
                dp[j,2] +=  1.0 / (tmp**2) * 2.0 * (p[i,2]-p[j,2])

        #print('dp',dp)
        p -= 0.0001 * dp
        #print('p',p)
        dp = np.absolute(dp)
        if dp.sum() < 0.1:
            break

    return p

p = np.random.rand(10,3)*1.0 - 0.5
p = getBetterP()
print(p)

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
        tmp = [p[w,0], p[w,1], p[w,2]]
        YY.append(tmp)

    YY = np.array(YY)
    x = np.matmul(iHtHHt,YY)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    Ys = np.matmul(H,x)
    ax.scatter(Ys[:,0], Ys[:,1], Ys[:,2], c=Yc, s=0.5, marker='o')
    plt.show()
        
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    Ys_t = np.matmul(H_t,x)
    ax.scatter(Ys_t[:,0], Ys_t[:,1], Ys_t[:,2], c=Yc_t, s=0.5, marker='o')
    plt.show()
    

