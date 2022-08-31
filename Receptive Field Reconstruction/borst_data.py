"""
Recreate the spatial receptive fields found in
"The Temporal Tuning of the Drosophila Motion Detectors Is Determined by the Dynamics of Their Input Elements" and
"Dynamic Signal Compression for Robust Motion Vision in Flies"
William Nourse
April 29th, 2022
"""
import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt

# Model data from supplemental material
# [Mi1 Tm3 Mi4 Mi9 Tm1 Tm2 Tm4 Tm9]
title = ['Mi1', 'Tm3', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm4', 'Tm9']
A_rel = np.array([0.022, 0.0, 0.132, 0.063, 0.04, 0.035, 0.054, 0.046])
FWHM_cen = np.array([6.81, 11.91, 6.47, 6.37, 8.12, 7.93, 11.45, 6.92])
FWHM_sur = np.array([28.81, 1.0, 16.14, 23.98, 27.14, 30.52, 34.62, 23.78])
invert = np.array([1, 1, 1, -1, -1, -1, -1, -1])
std_cen = FWHM_cen/(2*np.sqrt(2*np.log(2)))
std_sur = FWHM_sur/(2*np.sqrt(2*np.log(2)))

def calc_2d_point(x, y, A_rel, std_cen, std_sur):
    return np.exp((-(x**2 + y**2))/(2*std_cen**2)) - A_rel*np.exp((-(x**2 + y**2))/(2*std_sur**2))

def calc_1d_point(x, A_rel, std_cen, std_sur):
    return np.exp((-x**2)/(2*std_cen**2)) - A_rel*np.exp((-x**2)/(2*std_sur**2))

res = 1
max_angle = 30
min_angle = -30
axis = np.arange(min_angle, max_angle+res, res)
split = 4

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",["darkblue","blue","white","red","darkred"])
norm = plt.Normalize(-1,1)

plt.figure()
for neuron in range(split):
    plt.subplot(2,4,neuron+1)
    field_2d = np.zeros([len(axis),len(axis)])
    for i in range(len(axis)):
        for j in range(len(axis)):
            field_2d[i,j] = invert[neuron] * calc_2d_point(axis[i], axis[j], A_rel[neuron], std_cen[neuron], std_sur[neuron])
    plt.imshow(field_2d, extent=[axis[0],axis[-1],axis[0],axis[-1]],cmap=cmap,norm=norm)
    plt.colorbar()
    plt.title(title[neuron])
    plt.xlabel('Azimuth (deg)')
    plt.ylabel('Elevation (deg)')

    plt.subplot(2,4,neuron+5)
    field_1d = np.zeros(len(axis))
    for i in range(len(axis)):
        field_1d[i] = invert[neuron] * calc_1d_point(axis[i], A_rel[neuron], std_cen[neuron], std_sur[neuron])

    plt.plot(axis,field_1d)
    plt.xlabel('Angle (deg)')
    plt.ylabel('Response')

plt.figure()
for neuron in range(split,len(title)):
    plt.subplot(2,4,neuron-split+1)
    field_2d = np.zeros([len(axis),len(axis)])
    for i in range(len(axis)):
        for j in range(len(axis)):
            field_2d[i,j] = invert[neuron] * calc_2d_point(axis[i], axis[j], A_rel[neuron], std_cen[neuron], std_sur[neuron])

    plt.imshow(field_2d, extent=[axis[0],axis[-1],axis[0],axis[-1]],cmap=cmap,norm=norm)
    plt.colorbar()
    plt.title(title[neuron])
    plt.xlabel('Azimuth (deg)')
    plt.ylabel('Elevation (deg)')

    plt.subplot(2,4,neuron-split+5)
    field_1d = np.zeros(len(axis))
    for i in range(len(axis)):
        field_1d[i] = invert[neuron] * calc_1d_point(axis[i], A_rel[neuron], std_cen[neuron], std_sur[neuron])

    plt.plot(axis,field_1d)
    plt.xlabel('Angle (deg)')
    plt.ylabel('Response')


plt.show()