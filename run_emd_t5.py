import numpy as np
import torch
from motion_vision_networks import gen_single_emd_off
from utilities import cutoff_fastest, gen_gratings, save_data, calc_cap_from_cutoff
from sns_toolbox.renderer import render
import matplotlib.pyplot as plt
import seaborn as sea
from tqdm import tqdm
import time

def gen_stimulus(num_cycles):
    stim_on_lr = np.array([[1.0, 1.0, 1.0],
                           [0.0, 1.0, 1.0],
                           [0.0, 0.0, 1.0],
                           [0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [1.0, 1.0, 0.0]])
    for i in range(num_cycles):
        if i == 0:
            stim_full = np.copy(stim_on_lr)
        else:
            stim_full = np.vstack((stim_on_lr, stim_full))
    return stim_full
def test_emd(dt, model, net, stim, interval):
    model.reset()

    num_samples = np.shape(stim)[0]
    t = np.linspace(0, dt*num_samples*interval, num=num_samples*interval)

    data = np.zeros([num_samples*interval, net.get_num_outputs_actual()])
    stim_l = np.zeros_like(t)
    stim_c = np.zeros_like(t)
    stim_r = np.zeros_like(t)

    # if interval:
    index = 0
    j = 0
    for i in tqdm(range(len(t))):
        if index < num_samples:
            data[i,:] = model(stim[index,:])
            stim_l[i] = stim[index, 0]
            stim_c[i] = stim[index, 1]
            stim_r[i] = stim[index, 2]
        else:
            data[i, :] = model(stim[-1, :])
            stim_l[i] = stim[-1, 0]
            stim_c[i] = stim[-1, 1]
            stim_r[i] = stim[-1, 2]
        j += 1
        if j == interval:
            index += 1
            j = 0
    # else:
    #     for i in range(len(t)):
    #         data[i, :] = model(stim[i, :])

    # data = data.to('cpu')
    data = data.transpose()
    r_l = data[0, :]
    r_c = data[1, :]
    r_r = data[2, :]
    l2_l = data[3, :]
    l2_c = data[4, :]
    l2_r = data[5, :]
    l3_l = data[6, :]
    l3_c = data[7, :]
    l3_r = data[8, :]
    tm1_l = data[9, :]
    tm1_c = data[10, :]
    tm1_r = data[11, :]
    tm9_l = data[12, :]
    tm9_c = data[13, :]
    tm9_r = data[14, :]
    ct1_l = data[15, :]
    ct1_c = data[16, :]
    ct1_r = data[17, :]
    t5_a = data[18, :]
    t5_b = data[19, :]


    return t, r_l, r_c, r_r, l2_l, l2_c, l2_r, l3_l, l3_c, l3_r, tm1_l, tm1_c, tm1_r, tm9_l, tm9_c, tm9_r, ct1_l, ct1_c,\
        ct1_r, stim_l, stim_c, stim_r, t5_a, t5_b

def plot_emd(dt, interval, stim, cutoff_fast, ratio_low, cutoff_ct1, cutoff_tm9, plot=False):

    #                   Retina          L2 low              L2 High         L3              Tm1     Tm9          CT1 Off          T4
    params = np.array([cutoff_fast, cutoff_fast/ratio_low, cutoff_fast, cutoff_fast, cutoff_fast, cutoff_tm9, cutoff_ct1, cutoff_fast])   # Good guess

    model, net = gen_single_emd_off(dt, params)
    # stim, y = gen_gratings((1,3), freq, 'lr', 5, dt, square=True, device=device)
    # render(net, view=True)
    t, r_l, r_c, r_r, l2_l, l2_c, l2_r, l3_l, l3_c, l3_r, tm1_l, tm1_c, tm1_r, tm9_l, tm9_c, tm9_r, ct1_l, ct1_c,\
            ct1_r, stim_l, stim_c, stim_r, t5_a, t5_b = test_emd(dt, model, net, stim, interval)


    if plot:
        plt.figure()
        plt.suptitle('Interval: %.2f ms'%(interval*dt))
        plt.subplot(5,1,1)
        plt.title('Stim')
        plt.plot(t, stim_l, label='left')
        plt.plot(t, stim_c, label='center')
        plt.plot(t, stim_r, label='right')
        plt.legend()
        sea.despine()
        plt.subplot(5,1,2)
        plt.title('Lamina')
        plt.plot(t, l2_l, label='left')
        plt.plot(t, l2_c, label='center')
        plt.plot(t, l2_r, label='right')
        plt.legend()
        sea.despine()
        plt.subplot(5, 1, 3)
        plt.title('PD')
        plt.plot(t, tm9_l, label='Tm9', color='C3')
        plt.plot(t, tm1_c, label='Tm1', color='C4')
        plt.plot(t, ct1_r, label='CT1 Off', color='C5')
        plt.legend()
        sea.despine()
        plt.subplot(5, 1, 4)
        plt.title('ND')
        plt.plot(t, ct1_l, label='CT1 Off', color='C5')
        plt.plot(t, tm1_c, label='Tm1', color='C4')
        plt.plot(t, tm9_r, label='Tm9', color='C3')
        plt.legend()
        sea.despine()
        plt.subplot(5, 1, 5)
        plt.plot(t, t5_a, color='C6', label='T5_a')
        plt.plot(t, t5_b, color='C7', label='T5_b')
        plt.legend()
        sea.despine()


def convert_deg_vel_to_interval(vel, dt):
    scaled_vel = vel/5    # columns per second
    interval = int((1/scaled_vel)/(dt/1000))
    return interval

def convert_interval_to_deg_vel(interval, dt):
    vel = 5000/(interval*dt)
    return vel

def t4_freq_response(stim, vels, params, dt, plot=False, save=True):
    cutoff_fast = params[0]
    ratio_low = params[1]
    cutoff_ct1 = params[2]
    cutoff_mi9 = params[3]
    c_inv = params[4]

    a_peaks = np.zeros_like(vels)
    b_peaks = np.zeros_like(vels)
    ratios = np.zeros_like(vels)
    dpi = 100
    size = (9,6)
    if plot:
        dir = 'T4 Velocity/'
        fig = plt.figure(figsize=size, dpi=dpi)
        plt.suptitle('Cutoff_fast: %.2f, Ratio_low: %.2f, Ratio_CT1: %.2f, Ratio_Mi9: %.2f, C_inv: %.2f' % (
        cutoff_fast, ratio_low, cutoff_ct1, cutoff_mi9, c_inv))
    for i in range(len(vels)):
        # print(vels[i])
        if plot:
            plt.subplot(len(vels),1,i+1)
        interval = convert_deg_vel_to_interval(vels[i], dt)
        # print(interval)
        a_peaks, b_peaks[i], ratios[i] = plot_emd(dt, interval, stim, cutoff_fast, ratio_low, cutoff_ct1, cutoff_mi9, c_inv, plot=plot, save=save)
    param_string = '_%i_%i_%i_%i_%i'%(cutoff_fast, ratio_low, cutoff_ct1, cutoff_mi9, c_inv)
    if plot:
        filetype = '.svg'
        plt.savefig(dir+'data'+param_string+filetype, dpi=dpi)

        fig1 = plt.figure(figsize=size, dpi=dpi)
        # print(fig1.dpi)
        plt.suptitle('Cutoff_fast: %.2f, Ratio_low: %.2f, Cutoff_CT1: %.2f, Cutoff_Mi9: %.2f, C_inv: %.2f'%(cutoff_fast, ratio_low, cutoff_ct1, cutoff_mi9, c_inv))
        plt.subplot(2,1,1)
        plt.plot(vels, b_peaks)
        sea.despine()
        plt.xscale('log')
        plt.title('B Peak')
        plt.subplot(2,1,2)
        plt.plot(vels, ratios)
        plt.title('B/A')
        plt.xscale('log')
        sea.despine()
        plt.savefig(dir+'curve' + param_string + filetype, dpi=dpi)
    if save:
        dir = 'T4 Velocity/'
        data = {'vels':     vels,
                'a_peaks':  a_peaks,
                'b_peaks':  b_peaks,
                'ratios':   ratios}
        filename = dir + 'set' + param_string + '.pc'
        save_data(data, filename)

sea.set_theme()
sea.set_style('ticks')
sea.color_palette('colorblind')
stim_off_lr = gen_stimulus(2)
num_intervals = 4
vels = np.linspace(10,180,num=num_intervals)
#
dt = 0.1
goal = np.linspace(1.0, 0.1, num=num_intervals)

params = [200, 10, 50, 200]
interval = 100
# start = time.time()
plot_emd(dt, interval, stim_off_lr, params[0], params[1], params[2], params[3], plot=True)
# end = time.time()-start
# print(end)

plt.show()
