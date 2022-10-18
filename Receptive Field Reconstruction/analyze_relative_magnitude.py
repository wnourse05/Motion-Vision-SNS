import numpy as np
import matplotlib.pyplot as plt
import pickle

def calc_1d_point(x, A_rel, std_cen, std_sur):
    return np.exp((-x**2)/(2*std_cen**2)) - A_rel*np.exp((-x**2)/(2*std_sur**2))

def gen_bar_chart(dataset, axis, width, offset):
    num_neurons = len(dataset['title'])
    data = np.zeros([num_neurons, len(axis)])
    normalized_data = np.zeros([num_neurons, len(axis)])
    peak_vals = np.zeros(num_neurons)
    for i in range(num_neurons):
        data[i,:] = np.abs(calc_1d_point(axis, dataset['A_rel'][i], dataset['std_cen'][i], dataset['std_sur'][i]))
        peak_val = np.max(data)
        normalized_data[i,:] = data[i,:] / peak_val
        peak_vals[i] = peak_val
        plt.bar(axis + width * (i+offset), normalized_data[i,:], width, label=dataset['title'][i])
        # print(i+offset)
    return i+offset+1, data, normalized_data, peak_vals


field_data = pickle.load(open('borst_receptive_fields.p', 'rb'))

res = 5
max_angle = 0
min_angle = -20
axis = np.arange(min_angle, max_angle+res, res)

plt.figure()
width = 0.35
offset, _, lamina_norm, _ = gen_bar_chart(field_data['lamina'], axis, width, 0)
offset, _, medulla_on_norm, _ = gen_bar_chart(field_data['medullaOn'], axis, width, offset)
_, _, medulla_off_norm, _ = gen_bar_chart(field_data['medullaOff'], axis, width, offset)
plt.legend()

norms = np.vstack((lamina_norm, medulla_on_norm, medulla_off_norm))
peaks = np.max(norms,axis=0)
std = np.std(norms,axis=0)
var = np.var(norms,axis=0)

plt.figure()
plt.bar(axis, peaks, yerr=var)
print(peaks)

plt.show()
