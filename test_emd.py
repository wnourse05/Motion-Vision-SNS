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

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.title('Retina')
    # plt.plot(t, y.to('cpu')[:len(t)], color='black', linestyle='--')
    # for i in range(3):
    #     plt.plot(t, retina_row[i+2,:], color='C'+str(i+2))
    # plt.subplot(2,1,2)
    # plt.title('T4')
    plt.plot(t, t4_a_single, label='T4a')
    plt.plot(t, t4_b_single, label='T4b')
    plt.plot(t, t4_c_single, label='T4c')
    plt.plot(t, t4_d_single, label='T4d')
    plt.legend()

def test_all_emd(model, net, freq, num_cycles):
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
    plt.suptitle(str(freq)+' Hz')


model, net = gen_test_emd((7,7), output_retina=True, output_t4a=True, output_t4b=True, output_t4c=True, output_t4d=True)

shape = [7,7]
freq = 100    # Hz
stim_a, y_a = gen_gratings(shape, freq, 'a', 10)
stim_b, y_b = gen_gratings(shape, freq, 'b', 10)
stim_c, y_c = gen_gratings(shape, freq, 'c', 10)
stim_d, y_d = gen_gratings(shape, freq, 'd', 10)
# stim_a = torch.vstack((on_rl, on_rl, on_rl, on_rl, on_rl))
# stim_b = torch.vstack((on_lr, on_lr, on_lr, on_lr, on_lr))

plt.figure()
plt.subplot(2,1,1)
plt.plot(y_a.to('cpu').numpy(), label='ref', color='black', linestyle='--')
plt.plot(stim_a.to('cpu').numpy()[:,0], label='a', color='C0')
plt.plot(stim_b.to('cpu').numpy()[:,0], label='b', color='C1')
plt.legend()
plt.subplot(2,1,2)
plt.plot(y_a.to('cpu').numpy(), label='ref', color='black', linestyle='--')
plt.plot(stim_c.to('cpu').numpy()[:,0], label='c', color='C2')
plt.plot(stim_d.to('cpu').numpy()[:,0], label='d', color='C3')
plt.legend()

# print('a')
# test_emd(model, net, stim_a, y_a)
# print('b')
# test_emd(model, net, stim_b, y_b)
# print('c')
# test_emd(model, net, stim_c, y_c)
# print('d')
# test_emd(model, net, stim_d, y_d)

test_all_emd(model, net, 50, 10)
test_all_emd(model, net, 100, 10)
test_all_emd(model, net, 150, 10)

plt.show()