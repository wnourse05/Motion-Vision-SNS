from utilities import gen_gratings, dt
import matplotlib.pyplot as plt
import numpy as np

def grating(dir):
    period = (10*dt)/1000
    freq = 1/period
    stim, y = gen_gratings([3,3], freq, dir, 1)

    stim = stim.to('cpu').numpy()
    num_samples = np.shape(stim)[0]

    plt.figure()
    for i in range(num_samples):
        matrix = np.reshape(stim[i,:], (3,3))
        plt.subplot(2,4,i+1)
        plt.imshow(matrix, cmap='gray', vmin=0, vmax=1)
        # plt.colorbar()
    plt.suptitle(dir)

grating('a')
grating('b')
grating('c')
grating('d')

plt.show()