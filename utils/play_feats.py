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
        for i in range(len(parts)-1):
            t = float(parts[1+i])
            tmp.append(t)
        H.append(tmp)
        Y.append(pn)   
        Yc.append(colors[pn])
    f.close()
    
    return np.array(H),Y,Yc

def getcounts(vals, Y, minval, maxval, numclusters):
    counts = [[0.0 for _ in range(numclusters)] for _ in range(10)]
    for l in range(len(vals)):
        num = int((vals[l] - minval) * (numclusters + 0.0) / (maxval-minval))
        counts[Y[l]][num] += 1.0
    
    counts = np.array(counts)
    for l in range(10):
        counts[l,:] *= 1.0/max(counts[l,:])
        
    return counts

H,Y,Yc = readfile('feats_train.txt')
H_t,Y_t,Yc_t = readfile('feats_test.txt')

#H,Y,Yc = readfile('probs_train.txt')
#H_t,Y_t,Yc_t = readfile('probs_test.txt')

for i in range(len(H[0])):
    vals_train = H[:,i]
    vals_test = H_t[:,i]
    
    minval = min(min(vals_train),min(vals_test))
    maxval = max(max(vals_train),max(vals_test))
    
    maxval += 0.001 * (maxval-minval)
    
    numclusters = 300
    counts_train = getcounts(vals_train, Y, minval, maxval, numclusters)
    counts_test = getcounts(vals_test, Y_t, minval, maxval, numclusters)    
    
    for l in range(10):
        plt.plot([minval + (i + 0.0) *(maxval-minval) / (numclusters + 0.0) for i in range(numclusters)], counts_train[l,:] + l, 'b', alpha = 0.8)
        plt.plot([minval + (i + 0.0) *(maxval-minval) / (numclusters + 0.0) for i in range(numclusters)], counts_test[l,:] + l, 'r', alpha = 0.8)
    plt.show()
