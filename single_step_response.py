import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

from utilities import dt, cutoff_fastest
from motion_vision_networks import gen_single_column

#                   Retina          L1                                  L2                              L3                  Mi1         Mi9             Tm1             Tm9             CT1_On          CT1_Off
cutoffs = np.array([cutoff_fastest, cutoff_fastest/10, cutoff_fastest, cutoff_fastest/5, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest])

model, net = gen_single_column(cutoffs)

t = np.arange(0,50, dt)
inputs = torch.ones([len(t), net.get_num_inputs()])
data = np.zeros([len(t), net.get_num_outputs_actual()])

for i in range(len(t)):
    data[i,:] = model(inputs[i,:])
data = data.transpose()

"""
########################################################################################################################
PLOTTING
"""
group = True

if group:
    plt.figure()
    grid = matplotlib.gridspec.GridSpec(4, 4)

if group:
    plt.subplot(grid[0,0])
else:
    plt.figure()
plt.plot(t, data[:][0])
plt.ylim([-0.1,1.1])
plt.title('Retina')

if group:
    plt.subplot(grid[1,0])
else:
    plt.figure()
plt.plot(t, data[:][1])
plt.ylim([-0.1,1.1])
plt.title('L1')

if group:
    plt.subplot(grid[1,1])
else:
    plt.figure()
plt.plot(t, data[:][2])
plt.ylim([-0.1,1.1])
plt.title('L2')

if group:
    plt.subplot(grid[1,2])
else:
    plt.figure()
plt.plot(t, data[:][3])
plt.ylim([-0.1,1.1])
plt.title('L3')

if group:
    plt.subplot(grid[2,0])
else:
    plt.figure()
plt.plot(t, data[:][4])
plt.ylim([-0.1,1.1])
plt.title('Mi1')

if group:
    plt.subplot(grid[2,1])
else:
    plt.figure()
plt.plot(t, data[:][5])
plt.ylim([-0.1,1.1])
plt.title('Mi9')

if group:
    plt.subplot(grid[2,2])
else:
    plt.figure()
plt.plot(t, data[:][6])
plt.ylim([-0.1,1.1])
plt.title('Tm1')

if group:
    plt.subplot(grid[2,3])
else:
    plt.figure()
plt.plot(t, data[:][7])
plt.ylim([-0.1,1.1])
plt.title('Tm9')

if group:
    plt.subplot(grid[3,0])
else:
    plt.figure()
plt.plot(t, data[:][8])
plt.ylim([-0.1,1.1])
plt.title('CT1_On')

if group:
    plt.subplot(grid[3,1])
else:
    plt.figure()
plt.plot(t, data[:][9])
plt.ylim([-0.1,1.1])
plt.title('CT1_Off')

plt.show()
