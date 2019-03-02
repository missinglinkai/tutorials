import numpy as np
from pylab import zeros, arange, subplots, plt, savefig

files = ['-All','-randomCrop','-None'] #'None-resizing'
legend = ['-All','-randomCrop','-None'] #'none, resizing'
colors = ['r','b','g','y','m','c','k','#f49542','#8c5219','#3df744','#f936c2','#8c1919']
its = 500
points = 135 #135
disp_int_used = 10

fig = plt.figure()
ax1 = plt.subplot(111)
ax1.set_xlabel('iteration')
ax1.set_ylabel('train acc (..), val acc (-)')
ax1.set_xlim([0,its])
it_axes = (arange(points) * disp_int_used)

def lpf(arr):

    for i, el in enumerate(arr):
        if i < 5: continue
        if el == 0: arr[i] = arr[i-1]

    for i,el in enumerate(arr):
        window = 10
        if i == 0 or i == len(arr)-1 : continue
        if window > i or len(arr)-i < window:
            window = min(i,len(arr)-i)
        arr[i] = np.mean(arr[i-window:i+window])
    return arr

for i, file in enumerate(files):
    a = np.load('../../datasets/102flowers/training_data/dataAugmentation' + file + '.txt.npz')
    train_acc = a['arr_1'][0:points*2]
    train_acc_adapted = zeros(points)
    val_acc = zeros(points)

    for k in range(0,points):
        train_acc_adapted[k] = train_acc[k*2]
    train_acc_adapted = lpf(train_acc_adapted)
    val_acc[1:] = lpf(a['arr_2'][0:points-1])

    ax1.plot(it_axes, train_acc_adapted, linestyle=':', color=colors[i])
    ax1.plot(it_axes, val_acc, color=colors[i], label=legend[i])

handles, labels = ax1.get_legend_handles_labels()
lgd = ax1.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
fig.savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()