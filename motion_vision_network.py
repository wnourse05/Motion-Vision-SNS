import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import sys

from sns_toolbox.networks import Network
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingMatrixConnection, NonSpikingPatternConnection
from sns_toolbox.renderer import render

"""
General Network parameters
"""
dim_desired_max = 32

net = Network('Motion Vision')

R = 1.0

"""
Load the input image
"""

img = cv.imread('/home/will/Pictures/sample_images/cameraman.png')   # load image file

shape_original = img.shape  # dimensions of the original image
dim_long = max(shape_original[0],shape_original[1]) # longest dimension of the original image
ratio = dim_desired_max/dim_long    # scaling ratio of original image
img_resized = cv.resize(img,None,fx=ratio,fy=ratio) # scale original image using ratio

img_color = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # transform the image from BGR to RGB
img_color_resized = cv.cvtColor(img_resized, cv.COLOR_BGR2RGB)  # resize the RGB image
img_gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)  # convert the resized image to grayscale [0-255]

shape = img_gray.shape  # dimensions of the resized grayscale image

img_flat = img_gray.flatten()   # flatten the image into 1 vector for neural processing
flat_size = len(img_flat)   # length of the flattened image vector

img_flat = img_flat*R/255.0 # scale all the intensities from 0-255 to 0-R

plt.figure()
plt.imshow(img_gray,cmap='gray')
plt.axis('off')

"""
Construct the retina
"""

neuron_retina = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=5.0)

net.add_population(neuron_retina, shape, name='Retina')
net.add_input('Retina', size=flat_size, name='Image')
net.add_output('Retina', name='Retina Output')

"""
Construct the lamina
"""

def create_bandpass_lamina(net, params_neuron, params_field, neuron_name):
    g_ab = params_neuron['g_ab']
    g_ac = params_neuron['g_ac']
    g_bd = params_neuron['g_bd']
    g_cd = params_neuron['g_cd']

    g_ab_matrix = np.zeros([shape[0]*shape[1], shape[0]*shape[1]])
    np.fill_diagonal(g_ab_matrix, g_ab)
    g_ac_matrix = np.zeros_like(g_ab_matrix)
    np.fill_diagonal(g_ac_matrix, g_ac)
    g_bd_matrix = np.zeros_like(g_ab_matrix)
    np.fill_diagonal(g_bd_matrix, g_bd)
    g_cd_matrix = np.zeros_like(g_ab_matrix)
    np.fill_diagonal(g_cd_matrix, g_cd)

    del_e_ab = params_neuron['del_e_ab']
    del_e_ac = params_neuron['del_e_ac']
    del_e_bd = params_neuron['del_e_bd']
    del_e_cd = params_neuron['del_e_cd']

    del_e_ab_matrix = np.zeros_like(g_ab_matrix)
    np.fill_diagonal(del_e_ab_matrix, del_e_ab)
    del_e_ac_matrix = np.zeros_like(g_ab_matrix)
    np.fill_diagonal(del_e_ac_matrix, del_e_ac)
    del_e_bd_matrix = np.zeros_like(g_ab_matrix)
    np.fill_diagonal(del_e_bd_matrix, del_e_bd)
    del_e_cd_matrix = np.zeros_like(g_ab_matrix)
    np.fill_diagonal(del_e_cd_matrix, del_e_cd)

    c_fast = params_neuron['c_fast']
    c_slow = params_neuron['c_slow']
    rest = 1.0

    neuron_fast = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=c_fast, resting_potential=rest)
    neuron_slow = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=c_slow, resting_potential=rest)

    net.add_population(neuron_fast, shape, name=neuron_name+'_input')
    net.add_population(neuron_fast, shape, name=neuron_name+'_fast', initial_value=0.0)
    net.add_population(neuron_slow, shape, name=neuron_name+'_slow', initial_value=0.0)
    net.add_population(neuron_fast, shape, name=neuron_name+'_output')

    synapse_ab = NonSpikingMatrixConnection(max_conductance=g_ab_matrix, reversal_potential=del_e_ab_matrix, e_lo=np.zeros_like(g_ab_matrix), e_hi=np.ones_like(g_ab_matrix))
    synapse_ac = NonSpikingMatrixConnection(max_conductance=g_ac_matrix, reversal_potential=del_e_ac_matrix, e_lo=np.zeros_like(g_ab_matrix), e_hi=np.ones_like(g_ab_matrix))
    synapse_bd = NonSpikingMatrixConnection(max_conductance=g_bd_matrix, reversal_potential=del_e_bd_matrix, e_lo=np.zeros_like(g_ab_matrix), e_hi=np.ones_like(g_ab_matrix))
    synapse_cd = NonSpikingMatrixConnection(max_conductance=g_cd_matrix, reversal_potential=del_e_cd_matrix, e_lo=np.zeros_like(g_ab_matrix), e_hi=np.ones_like(g_ab_matrix))

    net.add_connection(synapse_ab, neuron_name+'_input', neuron_name+'_fast')
    net.add_connection(synapse_ac, neuron_name+'_input', neuron_name+'_slow')
    net.add_connection(synapse_bd, neuron_name+'_fast', neuron_name+'_output')
    net.add_connection(synapse_cd, neuron_name+'_slow', neuron_name+'_output')

    net.add_output(neuron_name+'_output', neuron_name+' Output')

    synapse_ret = NonSpikingPatternConnection(max_conductance_kernel=params_field['g'], reversal_potential_kernel=params_field['del_e'], e_lo_kernel=np.zeros_like(params_field['g']), e_hi_kernel=np.ones_like(params_field['g']))
    net.add_connection(synapse_ret, 'Retina', neuron_name+'_input')

def create_lowpass_lamina(net, params_neuron, params_field, neuron_name):
    neuron_type = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=params_neuron['c'], resting_potential=params_neuron['rest'])
    synapse_ret = NonSpikingPatternConnection(max_conductance_kernel=params_field['g'], reversal_potential_kernel=params_field['del_e'], e_lo_kernel=np.zeros_like(params_field['g']), e_hi_kernel=np.ones_like(params_field['g']))

    net.add_population(neuron_type, shape, name=neuron_name)

    net.add_connection(synapse_ret, 'Retina', neuron_name)

    net.add_output(neuron_name, neuron_name+' Output')

params_neuron_L1 = pickle.load(open('Receptive Field Reconstruction/L1_params.p', 'rb'))
params_field_L1 = pickle.load(open('Receptive Field Reconstruction/L1_field_params.p', 'rb'))
params_neuron_L2 = pickle.load(open('Receptive Field Reconstruction/L2_params.p', 'rb'))
params_field_L2 = pickle.load(open('Receptive Field Reconstruction/L2_field_params.p', 'rb'))
params_neuron_L3 = pickle.load(open('Receptive Field Reconstruction/L3_params.p', 'rb'))
params_field_L3 = pickle.load(open('Receptive Field Reconstruction/L3_field_params.p', 'rb'))
params_neuron_L4 = pickle.load(open('Receptive Field Reconstruction/L4_params.p', 'rb'))
params_field_L4 = pickle.load(open('Receptive Field Reconstruction/L4_field_params.p', 'rb'))
params_neuron_L5 = pickle.load(open('Receptive Field Reconstruction/L5_params.p', 'rb'))
params_field_L5 = pickle.load(open('Receptive Field Reconstruction/L5_field_params.p', 'rb'))

create_bandpass_lamina(net, params_neuron_L1, params_field_L1, 'L1')    # L1
create_bandpass_lamina(net, params_neuron_L2, params_field_L2, 'L2')    # L2
create_lowpass_lamina( net, params_neuron_L3, params_field_L3, 'L3')    # L3
create_bandpass_lamina(net, params_neuron_L4, params_field_L4, 'L4')    # L4
create_bandpass_lamina(net, params_neuron_L5, params_field_L5, 'L5')    # L5

render(net)



dt = 1  # calculate the ideal dt
t_max = 5  # run for 15 ms
steps = int(t_max/dt)   # number of steps to simulate

model = net.compile(backend='numpy',dt=dt,debug=False) # compile using the numpy backend

for i in range(steps):
    print('%i / %i steps'%(i+1,steps))

    # plt.figure()    # create a figure for live plotting the retina and lamina states
    # grid = matplotlib.gridspec.GridSpec(2,5)
    # plt.subplot(grid[0,2])
    # plt.title('Retina')
    # plt.axis('off')
    # plt.subplot(grid[1,0])
    # plt.title('L1')
    # plt.axis('off')
    # plt.subplot(grid[1, 1])
    # plt.title('L2')
    # plt.axis('off')
    # plt.subplot(grid[1, 2])
    # plt.title('L3')
    # plt.axis('off')
    # plt.subplot(grid[1, 3])
    # plt.title('L4')
    # plt.axis('off')
    # plt.subplot(grid[1, 4])
    # plt.title('L5')
    # plt.axis('off')

    out = model(img_flat)   # run the network for one dt
    retina = out[:flat_size]    # separate the retina and lamina states
    l1 = out[flat_size:2*flat_size]
    l2 = out[2*flat_size:3*flat_size]
    l3 = out[3*flat_size:4*flat_size]
    l4 = out[4*flat_size:5*flat_size]
    l5 = out[5*flat_size:]

    retina_reshape = np.reshape(retina,shape)   # reshape to from flat to an image
    l1_reshape = np.reshape(l1,shape)
    l2_reshape = np.reshape(l2, shape)
    l3_reshape = np.reshape(l3, shape)
    l4_reshape = np.reshape(l4, shape)
    l5_reshape = np.reshape(l5, shape)

    # plt.subplot(grid[0,2])  # plot the current state
    # plt.imshow(retina_reshape,cmap='gray')
    # plt.subplot(grid[1,0])
    # plt.imshow(l1_reshape, cmap='gray')
    # plt.subplot(grid[1,1])
    # plt.imshow(l2_reshape, cmap='gray')
    # plt.subplot(grid[1, 2])
    # plt.imshow(l3_reshape, cmap='gray')
    # plt.subplot(grid[1,3])
    # plt.imshow(l4_reshape, cmap='gray')
    # plt.subplot(grid[1,4])
    # plt.imshow(l5_reshape, cmap='gray')

plt.figure()
plt.imshow(retina_reshape, cmap='gray')
plt.title('Retina')
plt.axis('off')

plt.figure()
plt.imshow(l1_reshape, cmap='gray')
plt.title('L1')
plt.axis('off')

plt.figure()
plt.imshow(l2_reshape, cmap='gray')
plt.title('L2')
plt.axis('off')

plt.figure()
plt.imshow(l3_reshape, cmap='gray')
plt.title('L3')
plt.axis('off')

plt.figure()
plt.imshow(l4_reshape, cmap='gray')
plt.title('L4')
plt.axis('off')

plt.figure()
plt.imshow(l5_reshape, cmap='gray')
plt.title('L5')
plt.axis('off')

plt.show()
