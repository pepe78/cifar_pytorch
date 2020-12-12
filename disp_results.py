import matplotlib.pyplot as plt
import sys
import numpy as np
import math as m

def addsubgraph(figshape, which, t, q1, q2, lab):
	plt.subplot2grid(figshape,which)
	plt.plot(t,q2,'r', label='test')
	plt.plot(t,q1,'b', label='train')
	plt.xlabel('epoch')
	plt.ylabel(lab)
	plt.grid(color='k', linestyle=':', linewidth=1)
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
		   
def display_results(a1, a2, e1, e2, waitForGraph=False, tr_H = None, tr_Y = None, te_H = None, te_Y = None, epoch = 0):
    figure = plt.figure(num='Progress of training', figsize = (15,12) if tr_H is None else (25,12))
    plt.clf()

    figshape = (2,2) if tr_H is None else (2,5)
    t = range(len(a1))
    addsubgraph(figshape, (0,0),t,a1,a2,'accuracy')
    addsubgraph(figshape, (1,0),t,e1,e2,'error')
    addsubgraph(figshape, (0,1),t[len(t)//2:],a1[len(t)//2:],a2[len(t)//2:],'accuracy')
    addsubgraph(figshape, (1,1),t[len(t)//2:],e1[len(t)//2:],e2[len(t)//2:],'error')
    
    if tr_H is not None:
        display_clusters(tr_H, tr_Y, te_H, te_Y, figshape)
    
    if waitForGraph:
        plt.show()
    else: 
        plt.draw()
        #epoch += 100
        #plt.savefig(f"epoch_{epoch}.png")
        plt.pause(0.01)

def display_clusters(H,Y,H_t,Y_t, figshape):
    colors = ['b','g','r','c','m','y','indigo','lime','aqua','peru']
    
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
    plt.scatter(Ys[:,0], Ys[:,1], s=0.2, c=Yc)
    
    Ys_t = np.matmul(H_t,x)
    plt.subplot2grid(figshape,(1,2))
    plt.scatter(Ys_t[:,0], Ys_t[:,1], s=0.2, c=Yc_t)

    YY = []
    for i in range(len(Y)):
        w = trans[Y[i]]
        tmp = [w]
        YY.append(tmp)

    YY = np.array(YY)
    x = np.matmul(iHtHHt,YY)

    Ys = np.matmul(H,x)
    plt.subplot2grid(figshape,(0,3))
    plt.scatter(Ys[:,0], np.random.rand(Ys.shape[0],1), s=0.2, c=Yc)
    
    Ys_t = np.matmul(H_t,x)
    plt.subplot2grid(figshape,(1,3))
    plt.scatter(Ys_t[:,0], np.random.rand(Ys_t.shape[0],1), s=0.2, c=Yc_t)

    pp=[[ 1.11254018,  0.26655949, -0.47088501],
        [ 0.1410844,   1.16623218, -0.39172308],
        [ 0.97738296, -0.37034427,  0.66988191],
        [ 0.20715083, -0.42163416, -1.14502125],
        [-0.79205995,  0.38139223, -0.87649402],
        [ 0.35668311,  0.79060029,  0.88263829],
        [ 0.2871481,  -1.20243241, -0.07202189],
        [-0.9863972,  -0.73336238, -0.13864365],
        [-0.35293657, -0.4870234,   1.07998481],
        [-0.96423092,  0.60948048,  0.47677822]]

    YY = []
    for i in range(len(Y)):
        w = trans[Y[i]]
        YY.append(pp[w])

    YY = np.array(YY)
    x = np.matmul(iHtHHt,YY)
    
    ax = plt.subplot2grid(figshape,(0,4), projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    
    Ys = np.matmul(H,x)
    ax.scatter(Ys[:,0], Ys[:,1], Ys[:,2], c=Yc, s=0.5, marker='o')
    
    ax = plt.subplot2grid(figshape,(1,4), projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    
    Ys_t = np.matmul(H_t,x)
    ax.scatter(Ys_t[:,0], Ys_t[:,1], Ys_t[:,2], c=Yc_t, s=0.5, marker='o')
        
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
