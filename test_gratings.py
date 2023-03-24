from utilities import gen_gratings, dt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

angles = np.arange(0, 360, 30)
vel = 40
dt = 1000
num_steps = 5
wavelength = 160
shape = (32,32)
fig = plt.figure()
grid = GridSpec(num_steps, len(angles), figure=fig)
for i in range(len(angles)):
    gratings = gen_gratings(wavelength, angles[i], vel, dt, num_steps)
    for j in range(num_steps):
        plt.subplot(grid[j, i])
        if j == 0:
            plt.title('%i'%(angles[i]))
        plt.set_cmap('gray')
        plt.imshow(gratings[j,:].reshape(shape), interpolation='none')
plt.show()