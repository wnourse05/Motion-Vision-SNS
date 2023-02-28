import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

from utilities import dt, cutoff_fastest, Stimulus, device, gen_gratings
from motion_vision_networks import gen_test_emd
from sns_toolbox.renderer import render

#                   Retina          L1                                  L2                              L3                  Mi1         Mi9             Tm1             Tm9             CT1_On          CT1_Off
cutoffs = np.array([cutoff_fastest, cutoff_fastest/10, cutoff_fastest, cutoff_fastest/5, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest])

def test_emd(model, net, stimulus, y):
    model.reset()
    size = 7*7
    start = 3*7
    # render(net)

    # stim = Stimulus(stimulus, interval)
    num_samples = np.shape(stimulus)[0]
    stim = stimulus

    t = np.linspace(0, num_samples*dt/1000, num=num_samples)
    data = torch.zeros([num_samples, net.get_num_outputs_actual()], device=device)

    for i in range(num_samples):
        data[i,:] = model(stim[i,:])

    data = data.to('cpu')
    data = data.transpose(0,1)
    retina = data[:size, :]
    t4_a = data[size:2*size, :]
    t4_b = data[2*size:3*size, :]
    t4_c = data[3*size:4*size, :]
    t4_d = data[4*size:, :]

    retina_row = retina[start:start+7,:]
    t4_a_row = t4_a[start:start+7,:]
    t4_a_single = t4_a_row[3,:]
    t4_b_row = t4_b[start:start+7,:]
    t4_b_single = t4_b_row[3,:]
    t4_c_row = t4_c[start:start+7,:]
    t4_c_single = t4_c_row[3,:]
    t4_d_row = t4_d[start:start+7,:]
    t4_d_single = t4_d_row[3,:]

    plt.plot(t, t4_a_single, label='T4a')
    plt.plot(t, t4_b_single, label='T4b')
    plt.plot(t, t4_c_single, label='T4c')
    plt.plot(t, t4_d_single, label='T4d')
    plt.legend()

def test_emd_all_neurons(model, net, stim):
    model.reset()
    size = 7*7
    start = 3*7
    # render(net)

    # stim = Stimulus(stimulus, interval)
    num_samples = np.shape(stim)[0]

    t = np.linspace(0, num_samples*dt/1000, num=num_samples)
    data = torch.zeros([num_samples, net.get_num_outputs_actual()], device=device)

    for i in range(num_samples):
        data[i,:] = model(stim[i,:])

    data = data.to('cpu')
    data = data.transpose(0,1)
    center = data[:size, :]
    left = data[size:2*size, :]
    right = data[2*size:3*size, :]
    t4_a = data[3*size:, :]

    center_row = center[start:start+7,:]
    center_single = center_row[3,:]
    left_row = left[start:start+7,:]
    left_single = left_row[2,:]
    right_row = right[start:start+7,:]
    right_single = right_row[4,:]
    t4_a_row = t4_a[start+7:,:]
    t4_a_single = t4_a_row[3,:]

    plt.plot(t, t4_a_single, label='EMD')
    plt.plot(t, left_single, label='Left')
    plt.plot(t, center_single, label='Center')
    plt.plot(t, right_single, label='T4d')
    plt.legend()

def test_all_emd(model, net, freq, num_cycles, neuron):
    print('Frequency: ' + str(freq) + ' Hz')
    shape = [7, 7]
    stim_a, y_a = gen_gratings(shape, freq, 'a', num_cycles)
    stim_b, y_b = gen_gratings(shape, freq, 'b', num_cycles)
    stim_c, y_c = gen_gratings(shape, freq, 'c', num_cycles)
    stim_d, y_d = gen_gratings(shape, freq, 'd', num_cycles)

    plt.figure()
    plt.subplot(2,2,1)
    plt.title('Right -> Left')
    test_emd(model, net, stim_a, y_a)
    plt.subplot(2, 2, 2)
    plt.title('Left -> Right')
    test_emd(model, net, stim_b, y_b)
    plt.subplot(2, 2, 3)
    plt.title('Bottom -> Top')
    test_emd(model, net, stim_c, y_c)
    plt.subplot(2,2,4)
    plt.title('Top -> Bottom')
    test_emd(model, net, stim_d, y_d)
    plt.suptitle(neuron + ': ' + str(freq)+' Hz')


model_t4, net_t4 = gen_test_emd((7,7), output_retina=True, output_t4a=True, output_t4b=True, output_t4c=True, output_t4d=True)
model_t5, net_t5 = gen_test_emd((7,7), output_retina=True, output_t5a=True, output_t5b=True, output_t5c=True, output_t5d=True)

model_t4_all, net_t4_all = gen_test_emd((7,7), output_mi1=True, output_mi9=True, output_ct1on=True, output_t4a=True)
model_t5_all, net_t5_all = gen_test_emd((7,7), output_tm1=True, output_tm9=True, output_ct1off=True, output_t5a=True)

# shape = [7,7]
# freq = 100    # Hz
# stim_a, y_a = gen_gratings(shape, freq, 'a', 10)
# stim_b, y_b = gen_gratings(shape, freq, 'b', 10)
# stim_c, y_c = gen_gratings(shape, freq, 'c', 10)
# stim_d, y_d = gen_gratings(shape, freq, 'd', 10)
# # stim_a = torch.vstack((on_rl, on_rl, on_rl, on_rl, on_rl))
# # stim_b = torch.vstack((on_lr, on_lr, on_lr, on_lr, on_lr))
#
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(y_a.to('cpu').numpy(), label='ref', color='black', linestyle='--')
# plt.plot(stim_a.to('cpu').numpy()[:,0], label='a', color='C0')
# plt.plot(stim_b.to('cpu').numpy()[:,0], label='b', color='C1')
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(y_a.to('cpu').numpy(), label='ref', color='black', linestyle='--')
# plt.plot(stim_c.to('cpu').numpy()[:,0], label='c', color='C2')
# plt.plot(stim_d.to('cpu').numpy()[:,0], label='d', color='C3')
# plt.legend()

# test_all_emd(model_t4, net_t4, 50, 10, 'T4')
# test_all_emd(model_t4, net_t4, 100, 10, 'T4')
# test_all_emd(model_t4, net_t4, 150, 10, 'T4')
#
test_all_emd(model_t5, net_t5, 50, 10, 'T5')
# test_all_emd(model_t5, net_t5, 100, 10, 'T5')
# test_all_emd(model_t5, net_t5, 150, 10, 'T5')

shape = [7,7]
freq = 100    # Hz
stim_a, y_a = gen_gratings(shape, freq, 'a', 10)

# plt.figure()
# plt.title('T4')
# test_emd_all_neurons(model_t4_all, net_t4_all, stim_a)
plt.figure()
plt.title('T5')
test_emd_all_neurons(model_t5_all, net_t5_all, stim_a)

plt.show()