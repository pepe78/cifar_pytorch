import matplotlib.pyplot as plt
import sys
import numpy as np
import math as m
import os

def addsubgraph(figshape, which, t, q1, q2, lab):
	plt.subplot2grid(figshape,which)
	plt.plot(t,q2,'r', label='test')
	plt.plot(t,q1,'b', label='train')
	plt.xlabel('epoch')
	plt.ylabel(lab)
	plt.grid(color='k', linestyle=':', linewidth=1)
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
		   
def display_results(a1, a2, e1, e2, waitForGraph=False, tr_H = None, tr_Y = None, te_H = None, te_Y = None, epoch = 0, dirname = './'):
    figure = plt.figure(num='Progress of training', figsize = (15,12) if tr_H is None else (25,12))
    plt.clf()

    figshape = (2,2) if tr_H is None else (2,4)
    t = range(len(a1))
    addsubgraph(figshape, (0,0),t,a1,a2,'accuracy')
    addsubgraph(figshape, (1,0),t,e1,e2,'error')
    addsubgraph(figshape, (0,1),t[len(t)//2:],a1[len(t)//2:],a2[len(t)//2:],'accuracy')
    addsubgraph(figshape, (1,1),t[len(t)//2:],e1[len(t)//2:],e2[len(t)//2:],'error')
    
    if tr_H is not None:
        display_clusters(tr_H, tr_Y, te_H, te_Y, figshape, epoch)
    
    if waitForGraph:
        plt.show()
    else: 
        plt.draw()
        if not os.path.isdir(dirname + 'tmp'):
            os.mkdir(dirname + 'tmp')
        
        plt.savefig(f"{dirname}tmp/epoch_{epoch:04d}.png")
        plt.pause(0.01)

def plotProbs(H, Y, minval, maxval):
    dif = []

    numclusters = 300
    counts_cor =[0 for _ in range(numclusters)]
    counts_inc =[0 for _ in range(numclusters)]
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            num = int((H[i,j] - minval) * (numclusters + 0.0) / (maxval-minval))
            if Y[i] == j:
                counts_cor[num] += 1.0
            else:
                counts_inc[num] += 1.0 / (H.shape[1] - 1.0)
                dif.append(H[i,Y[i]] - H[i,j])
                
    plt.plot([minval + (i + 0.0) *(maxval-minval) / (numclusters + 0.0) for i in range(numclusters)], counts_cor, 'b')
    plt.plot([minval + (i + 0.0) *(maxval-minval) / (numclusters + 0.0) for i in range(numclusters)], counts_inc, 'r')
    
    minval2 = min(dif)
    maxval2 = max(dif)
    maxval2 += 0.001 * (maxval2-minval2)

    counts_dif =[0 for _ in range(numclusters)]
    for i in range(len(dif)):
        num = int((dif[i] - minval2) * (numclusters + 0.0) / (maxval2-minval2))
        counts_dif[num] += 1.0 / (H.shape[1] - 1.0)
        
    plt.plot([minval2 + (i + 0.0) *(maxval2-minval2) / (numclusters + 0.0) for i in range(numclusters)], counts_dif, 'g')

def display_clusters(H,Y,H_t,Y_t, figshape, epoch):
    colors = ['b','g','r','c','m','y','indigo','lime','aqua','peru']

    minval = min(H.min(),H_t.min())
    maxval = max(H.max(),H_t.max())
    
    maxval += 0.001 * (maxval-minval)

    plt.subplot2grid(figshape,(0,3))
    plotProbs(H, Y, minval, maxval)
    plt.subplot2grid(figshape,(1,3))
    plotProbs(H_t, Y_t, minval, maxval)

    H = np.concatenate((H,np.ones((H.shape[0],1))),axis=1)
    H_t = np.concatenate((H_t,np.ones((H_t.shape[0],1))),axis=1)
    
    Yc = []
    for i in range(len(Y)):
        Yc.append(colors[Y[i]])

    Yc_t = []
    for i in range(len(Y_t)):
        Yc_t.append(colors[Y_t[i]])

    
    Ht = np.transpose(H)
    HtH = np.matmul(Ht,H)
    iHtH = np.linalg.inv(HtH)
    iHtHHt = np.matmul(iHtH,Ht)

    maxcor = 0
    trans = [i for i in range(10)]

    YY = []
    for i in range(len(Y)):
        w = trans[Y[i]]
        tmp = [m.cos(w * 2.0 * m.pi / 10.0),m.sin(w * 2.0 * m.pi / 10.0)]
        YY.append(tmp)

    YY = np.array(YY)
    x = np.matmul(iHtHHt,YY)

    Ys = np.matmul(H,x)
    plt.subplot2grid(figshape,(0,2))
    plt.scatter(Ys[:,0], Ys[:,1], s=0.2, c=Yc, alpha=0.3)
    
    Ys_t = np.matmul(H_t,x)
    plt.subplot2grid(figshape,(1,2))
    plt.scatter(Ys_t[:,0], Ys_t[:,1], s=0.2, c=Yc_t, alpha=0.3)

if __name__ == "__main__":
    filename = 'debug.txt'
    
    e1=[]
    a1=[]
    e2=[]
    a2=[]
    with open(filename, 'rt') as f:
        while True:
            content = f.readline()
            if len(content) == 0:
                break
            field = content.replace('\n','').split(',')

            if len(field)!=5:
                break

            e1.append(float(field[1]))
            a1.append(float(field[2]))
            e2.append(float(field[3]))
            a2.append(float(field[4]))

    print('Max train accuracy:', max(a1))
    print('Max test accuracy:', max(a2))
    print('Min train J:', min(e1))
    print('Min test J:', min(e2))

    display_results(a1,a2,e1,e2, True)
