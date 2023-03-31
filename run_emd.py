from utilities import load_data, save_data, gen_gratings
from motion_vision_networks import gen_motion_vision
from sns_toolbox.renderer import render
import torch
import numpy as np
from tqdm import tqdm

def get_stimulus():
    frame_0 = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    frame_1 = torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    frame_2 = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    frame_3 = torch.fliplr(frame_1)
    frame_4 = torch.fliplr(frame_0)
    frame_5 = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
    stim_0 = torch.vstack((torch.flatten(frame_0), torch.flatten(frame_1), torch.flatten(frame_2), torch.flatten(frame_3), torch.flatten(frame_4), torch.flatten(frame_5)))
    stim_90 = torch.vstack((torch.flatten(torch.rot90(frame_0)), torch.flatten(torch.rot90(frame_1)), torch.flatten(torch.rot90(frame_2)), torch.flatten(torch.rot90(frame_3)), torch.flatten(torch.rot90(frame_4)), torch.flatten(torch.rot90(frame_5))))
    stim_180 = torch.flipud(stim_0)
    stim_270 = torch.flipud(stim_90)

    frame_a = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    frame_b = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    frame_c = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    frame_d = frame_b.transpose(0,1)
    frame_e = frame_a.transpose(0,1)
    frame_f = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    stim_45 = torch.vstack((torch.flatten(frame_a), torch.flatten(frame_b), torch.flatten(frame_c),
                           torch.flatten(frame_d), torch.flatten(frame_e), torch.flatten(frame_f)))
    stim_135 = torch.vstack((torch.flatten(torch.rot90(frame_a)), torch.flatten(torch.rot90(frame_b)),
                            torch.flatten(torch.rot90(frame_c)), torch.flatten(torch.rot90(frame_d)),
                            torch.flatten(torch.rot90(frame_e)), torch.flatten(torch.rot90(frame_f))))
    stim_225 = torch.flipud(stim_45)
    stim_315 = torch.flipud(stim_135)

    stims = {'0': stim_0, '45': stim_45, '90': stim_90, '135': stim_135, '180': stim_180, '225': stim_225, '270': stim_270, '315': stim_315}

    return stims

def convert_deg_vel_to_interval(vel, dt):
    scaled_vel = vel/5    # columns per second
    interval = int((1/scaled_vel)/(dt/1000))
    return interval

def run(model, net, stim, device, size, dir, filename, dt, all, interval):
    model.reset()

    num_samples = stim.shape[0]
    t = np.linspace(0, dt * num_samples * interval, num=num_samples * interval)
    data = torch.zeros([num_samples*interval, net.get_num_outputs_actual()], device=device)
    stim_example = torch.zeros([len(t),3])

    index = 0
    j = 0
    for i in tqdm(range(len(t)), leave=False, colour='blue'):
        if index < num_samples:
            data[i, :] = model(stim[index,:])
            stim_example[i, 0] = stim[index, 23]
            stim_example[i, 1] = stim[index, 24]
            stim_example[i, 2] = stim[index, 25]
        else:
            data[i, :] = model(stim[-1,:])
            stim_example[i,0] = stim[-1,23]
            stim_example[i,1] = stim[-1,24]
            stim_example[i,2] = stim[-1,25]
        j += 1
        if j == interval:
            index += 1
            j = 0
    data = data.to('cpu')
    data = data.transpose(0,1)
    index = int(size/2)
    # inp = data[:size, :]
    # single_inp = inp[index-1:index+2, :]
    # del inp
    # bp = data[size:2*size, :]
    # single_bp = bp[index-1:index+2, :]
    # del bp
    # lp = data[2*size:3*size, :]
    # single_lp = lp[index-1:index+2, :]
    # del lp
    if all:
        e = data[3*size:4*size, :]
        single_e = e[index-1, :]
        del e
        d_on = data[4*size:5*size, :]
        single_d_on = d_on[index, :]
        del d_on
        d_off = data[5*size:6*size, :]
        single_d_off = d_off[index, :]
        del d_off
        s_on = data[6*size:7*size, :]
        single_s_on = s_on[index+1, :]
        del s_on
        s_off = data[7*size:8*size, :]
        single_s_off = s_off[index+1, :]
        del s_off
        emd_inputs = {'e': single_e,
                      'd_on': single_d_on,
                      's_on': single_s_on,
                      'd_off': single_d_off,
                      's_off': single_s_off}
        save_data(emd_inputs, '%s/emd_in_%s.pc' % (dir, filename))
        del emd_inputs
    on_a = data[8*size:9*size, :]
    single_on_a = on_a[index, :]
    del on_a
    on_b = data[9*size:10*size, :]
    single_on_b = on_b[index, :]
    del on_b
    on_c = data[10*size:11*size, :]
    single_on_c = on_c[index, :]
    del on_c
    on_d = data[11*size:12*size, :]
    single_on_d = on_d[index, :]
    del on_d
    off_a = data[12*size:13*size, :]
    single_off_a = off_a[index, :]
    del off_a
    off_b = data[13*size:14*size, :]
    single_off_b = off_b[index, :]
    del off_b
    off_c = data[14*size:15*size, :]
    single_off_c = off_c[index, :]
    del off_c
    off_d = data[15*size:, :]
    single_off_d = off_d[index, :]
    del off_d

    del data

    on_a_peak = torch.max(single_on_a)
    on_b_peak = torch.max(single_on_b)
    on_c_peak = torch.max(single_on_c)
    on_d_peak = torch.max(single_on_d)
    off_a_peak = torch.max(single_off_a)
    off_b_peak = torch.max(single_off_b)
    off_c_peak = torch.max(single_off_c)
    off_d_peak = torch.max(single_off_d)

    peaks = {'on_a': on_a_peak,
             'on_b': on_b_peak,
             'on_c': on_c_peak,
             'on_d': on_d_peak,
             'off_a': off_a_peak,
             'off_b': off_b_peak,
             'off_c': off_c_peak,
             'off_d': off_d_peak}

    save_data(peaks, '%s/peaks_%s.pc'%(dir,filename))
    del peaks

    if all:
        emd = {'on_a': single_on_a,
               'on_b': single_on_b,
               'on_c': single_on_c,
               'on_d': single_on_d,
               'off_a': single_off_a,
               'off_b': single_off_b,
               'off_c': single_off_c,
               'off_d': single_off_d}
        try:
            save_data(emd, '%s/emd_%s.pc'%(dir,filename))
        except:
            print('Too big, continuing...')
        del emd

    stim_example = stim_example.to('cpu')
    stim_params = {'t': t, 'stim': stim_example}
    save_data(stim_params, '%s/stim_%s.pc'%(dir, filename))


    return

def collect_data(vels, angles, stims, dir, all=False, suffix=None, center=False, params=None, scale=None):
    if params is None:
        params = load_data('params_net_20230327.pc')

    shape = (7, 7)
    flat_size = shape[0] * shape[1]
    backend = 'torch'
    device = 'cuda'
    model, net = gen_motion_vision(params, shape, backend, device, center=center, scale=scale)
    dt = params['dt']

    wavelength = 30.0
    fov_res = 5
    fov = shape[0] * fov_res
    num_cycles = 4
    for i in tqdm(range(len(vels)), leave=False, colour='green'):
        for j in tqdm(range(len(angles)), leave=False):
            interval = convert_deg_vel_to_interval(vels[i], dt)
            stim = torch.vstack((stims['%i'%angles[j]], stims['%i'%angles[j]], stims['%i'%angles[j]], stims['%i'%angles[j]])).to(device)
            if suffix is None:
                suffix = ''
            filename = '%i_%i%s'%(i, j, suffix)
            run(model, net, stim, device, flat_size, dir, filename, dt, all, interval)

stims = get_stimulus()

# EMD Behavior
vels = [30.0]
angles = [0, 180]
dir = 'Neuron Data'
collect_data(vels, angles, stims, dir, all=True, center=True, scale=False)

# Frequency Response
vels = np.geomspace(9,360,10)
angles = [0]
dir = 'Frequency Response'
collect_data(vels, angles, stims, dir, all=False,center=True, scale=False)

# Radar
vels = [30.0]
angles = [0, 45, 90, 135, 180, 225, 270, 315]
dir = 'Radar Data'
collect_data(vels, angles, stims, dir, all=False,center=True, scale=False)

# # Cutoff Fast
# vels = [30.0]
# angles = [0]
# dir = 'Vary Parameters/cutoff_fast'
# # [scale[0]*cutoff_fast, scale[1]*cutoff_low, scale[2]*cutoff_enhance, scale[3]*cutoff_suppress, scale[4]*div, scale[5]*ratio_enhance_off, scale[6]*cutoff_direct_on, scale[7]*cutoff_direct_off, dt]
# scale = [0.5, 0.75, 1.0, 1.25, 1.5]
# for i in range(len(scale)):
#     scales =
