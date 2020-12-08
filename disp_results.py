import matplotlib.pyplot as plt
import sys

def addsubgraph(which, t, q1, q2, lab):
	plt.subplot(which)
	plt.plot(t,q2,'r', label='test')
	plt.plot(t,q1,'b', label='train')
	plt.xlabel('epoch')
	plt.ylabel(lab)
	plt.grid(color='k', linestyle=':', linewidth=1)
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
		   ncol=2, mode="expand", borderaxespad=0.)
		   
def display_results(a1, a2, e1, e2, waitForGraph=False):
    plt.figure(num='Progress of training', figsize=(15,12))
    plt.clf()

    t = range(len(a1))
    addsubgraph(221,t,a1,a2,'accuracy')
    addsubgraph(223,t,e1,e2,'error')
    addsubgraph(222,t[len(t)//2:],a1[len(t)//2:],a2[len(t)//2:],'accuracy')
    addsubgraph(224,t[len(t)//2:],e1[len(t)//2:],e2[len(t)//2:],'error')
    
    if waitForGraph:
        plt.show()
    else: 
        plt.draw()
        plt.pause(0.01)

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

    display_results(a1,a2,e1,e1, True)
