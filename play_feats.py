import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

        tmp = []
        for i in range(512):
            t = float(parts[1+i])
            tmp.append(t)
        H.append(tmp)
        Y.append(pn)   
        Yc.append(colors[pn])
    f.close()
    
    return np.array(H),Y,Yc

H,Y,Yc = readfile('feats_train.txt')
#H_t,Y_t,Yc_t = readfile('feats_test.txt')

for i in range(512):
    plt.scatter(H[:,i], Y[:], s=0.5, marker='o')
    plt.show()
