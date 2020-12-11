import matplotlib.pyplot as plt
import sys
import numpy as np
import math as m

def addsubgraph(which, t, q1, q2, lab):
	plt.subplot(which)
	plt.plot(t,q2,'r', label='test')
	plt.plot(t,q1,'b', label='train')
	plt.xlabel('epoch')
	plt.ylabel(lab)
	plt.grid(color='k', linestyle=':', linewidth=1)
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
		   
def display_results(a1, a2, e1, e2, waitForGraph=False, tr_H = None, tr_Y = None, te_H = None, te_Y = None, epoch = 0):
    plt.figure(num='Progress of training', figsize = (15,12) if tr_H is None else (25,12))
    plt.clf()

    t = range(len(a1))
    addsubgraph(221 if tr_H is None else 241,t,a1,a2,'accuracy')
    addsubgraph(223 if tr_H is None else 245,t,e1,e2,'error')
    addsubgraph(222 if tr_H is None else 242,t[len(t)//2:],a1[len(t)//2:],a2[len(t)//2:],'accuracy')
    addsubgraph(224 if tr_H is None else 246,t[len(t)//2:],e1[len(t)//2:],e2[len(t)//2:],'error')
    
    if tr_H is not None:
        display_clusters(tr_H, tr_Y, te_H, te_Y)
    
    if waitForGraph:
        plt.show()
    else: 
        plt.draw()
        #epoch += 100
        #plt.savefig(f"epoch_{epoch}.png")
        plt.pause(0.01)

def display_clusters(H,Y,H_t,Y_t):
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
    plt.subplot(243)
    plt.scatter(Ys[:,0], Ys[:,1], s=0.2, c=Yc)
    
    Ys_t = np.matmul(H_t,x)
    plt.subplot(247)
    plt.scatter(Ys_t[:,0], Ys_t[:,1], s=0.2, c=Yc_t)

    YY = []
    for i in range(len(Y)):
        w = trans[Y[i]]
        tmp = [w]
        YY.append(tmp)

    YY = np.array(YY)
    x = np.matmul(iHtHHt,YY)

    Ys = np.matmul(H,x)
    plt.subplot(244)
    plt.scatter(Ys[:,0], np.random.rand(Ys.shape[0],1), s=0.2, c=Yc)
    
    Ys_t = np.matmul(H_t,x)
    plt.subplot(248)
    plt.scatter(Ys_t[:,0], np.random.rand(Ys_t.shape[0],1), s=0.2, c=Yc_t)
    
        
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
