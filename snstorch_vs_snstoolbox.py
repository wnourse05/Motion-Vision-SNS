from LivingMachines2023.utilities import load_data
from LivingMachines2023.motion_vision_networks import gen_motion_vision
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from motion_vision_net import SNSMotionVisionEye

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

def run_sns_toolbox(model, net, stim, device, size, dt, interval):
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
    inp = data[:size, :]
    single_inp = inp[index-1:index+2, :]
    del inp
    bp = data[size:2*size, :]
    single_bp = bp[index-1:index+2, :]
    del bp
    lp = data[2*size:3*size, :]
    single_lp = lp[index-1:index+2, :]
    del lp
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
    on_a = data[8*size:9*size, :]
    single_on_a = on_a[index, :]
    del on_a
    on_b = data[9*size:10*size, :]
    single_on_b = on_b[index, :]
    del on_b
    off_a = data[12*size:13*size, :]
    single_off_a = off_a[index, :]
    del off_a
    off_b = data[13*size:14*size, :]
    single_off_b = off_b[index, :]
    del off_b


    del data


    data = {'t': t,
            'inp': single_inp,
            'bp': single_bp,
            'lb': single_lp,
            'e_on': single_e,
            'd_on': single_d_on,
            's_on': single_s_on,
            'e_off': single_e,
            'd_off': single_d_off,
            's_off': single_s_off,
            'on_a': single_on_a,
            'on_b': single_on_b,
            'off_a': single_off_a,
            'off_b': single_off_b}


    return data

def collect_data_sns_toolbox(vel, angle, stims, center=False, params=None, scale=None):
    if params is None:
        params = load_data('LivingMachines2023/params_net_20230327.pc')

    shape = (7, 7)
    flat_size = shape[0] * shape[1]
    backend = 'torch'
    device = 'cuda'
    model, net = gen_motion_vision(params, shape, backend, device, center=center, scale=scale)
    dt = params['dt']

    interval = convert_deg_vel_to_interval(vel, dt)
    # stim = torch.vstack((stims['%i'%angle], stims['%i'%angle], stims['%i'%angle], stims['%i'%angle])).to(device)
    stim = torch.vstack((stims['%i' % angle],)).to(device)
    data = run_sns_toolbox(model, net, stim, device, flat_size, dt, interval)
    return data, net


stims = get_stimulus()
params_sns = load_data('LivingMachines2023/params_net_20230327.pc')

# EMD Behavior
vel = 30.0
angle = 0
data_sns_toolbox, net = collect_data_sns_toolbox(vel, angle, stims, center=True, scale=False)

# data_sns_toolbox = {'t': t,
#             'inp': single_inp,
#             'bp': single_bp,
#             'lb': single_lp,
#             'e_on': single_e,
#             'd_on': single_d_on,
#             's_on': single_s_on,
#             'e_off': single_e,
#             'd_off': single_d_off,
#             's_off': single_s_off,
#             'on_a': single_on_a,
#             'on_b': single_on_b,
#             'off_a': single_off_a,
#             'off_b': single_off_b}

dtype = torch.float32
device = 'cpu'

interval = convert_deg_vel_to_interval(vel, params_sns['dt'])
stim = torch.vstack((stims['%i' % angle],)).to(device)

params = nn.ParameterDict({
    'reversalEx': nn.Parameter(torch.tensor([5.0], dtype=dtype).to(device)),
    'reversalIn': nn.Parameter(torch.tensor([-2.0], dtype=dtype).to(device)),
    'reversalMod': nn.Parameter(torch.tensor([0.0], dtype=dtype).to(device)),
    'freqFast': nn.Parameter(torch.tensor([params_sns['in']['cutoff']],dtype=dtype).to(device)),
    'kernelConductanceInBO': nn.Parameter(torch.tensor([params_sns['bp']['g']['center']], dtype=dtype).to(device)),
    'kernelReversalInBO': nn.Parameter(torch.tensor([-2.0], dtype=dtype).to(device)),
    'freqBOFast': nn.Parameter(torch.tensor([params_sns['bp']['cutoffHigh']],dtype=dtype).to(device)),
    'freqBOSlow': nn.Parameter(torch.tensor([params_sns['bp']['cutoffLow']],dtype=dtype).to(device)),
    'kernelConductanceInL': nn.Parameter(torch.tensor([params_sns['lp']['g']['center']], dtype=dtype).to(device)),
    'kernelReversalInL': nn.Parameter(torch.tensor([-2.0], dtype=dtype).to(device)),
    'freqL': nn.Parameter(torch.tensor([params_sns['in']['cutoff']],dtype=dtype).to(device)),
    'kernelConductanceInBF': nn.Parameter(torch.tensor([params_sns['bp']['g']['center']], dtype=dtype).to(device)),
    'kernelReversalInBF': nn.Parameter(torch.tensor([-2.0], dtype=dtype).to(device)),
    'freqBFFast': nn.Parameter(torch.tensor([params_sns['bp']['cutoffHigh']],dtype=dtype).to(device)),
    'freqBFSlow': nn.Parameter(torch.tensor([params_sns['bp']['cutoffLow']],dtype=dtype).to(device)),
})
model_torch = SNSMotionVisionEye(params_sns['dt'],(7,7), 1, params=params, dtype=dtype, device=device)

num_samples = stim.shape[0]
t = np.linspace(0, params_sns['dt'] * num_samples * interval, num=num_samples * interval)
data_in = torch.zeros(num_samples*interval, device=device)
data_bo = torch.zeros(num_samples*interval, device=device)
data_l = torch.zeros(num_samples*interval, device=device)
data_bf = torch.zeros(num_samples*interval, device=device)
data = [data_in, data_bo, data_l, data_bf]

stim_example = torch.zeros([len(t),3])

shape = [7,7]
state_input = torch.zeros(shape, dtype=dtype).to(device)
state_bp_on_input = torch.ones(shape, dtype=dtype).to(device)
state_bp_on_fast = torch.zeros(shape, dtype=dtype).to(device)
state_bp_on_slow = torch.zeros(shape, dtype=dtype).to(device)
state_bp_on_output = torch.ones(shape, dtype=dtype).to(device)
state_lowpass = torch.ones(shape, dtype=dtype).to(device)
state_bp_off_input = torch.ones(shape, dtype=dtype).to(device)
state_bp_off_fast = torch.zeros(shape, dtype=dtype).to(device)
state_bp_off_slow = torch.zeros(shape, dtype=dtype).to(device)
state_bp_off_output = torch.ones(shape, dtype=dtype).to(device)
states = [state_input, state_bp_on_input, state_bp_on_fast, state_bp_on_slow, state_bp_on_output, state_lowpass,
          state_bp_off_input, state_bp_off_fast, state_bp_off_slow, state_bp_off_output]

def state_to_data(index, data, states):
    data[0][index] = states[0][3,3]
    # state_bp_on_input = states[1]
    # state_bp_on_fast = states[2]
    # state_bp_on_slow = states[3]
    data[1][index] = states[4][3,3]
    data[2][index] = states[5][3,3]
    # state_bp_off_input = states[6]
    # state_bp_off_fast = states[7]
    # state_bp_off_slow = states[8]
    data[3][index] = states[9][3,3]

    return data

index = 0
j = 0
for i in tqdm(range(len(t)), leave=False, colour='blue'):
    if index < num_samples:
        states = model_torch(torch.reshape(stim[index,:],(7,7)), states)
        stim_example[i, 0] = stim[index, 23]
        stim_example[i, 1] = stim[index, 24]
        stim_example[i, 2] = stim[index, 25]
    else:
        states = model_torch(stim[-1,:], states)
        stim_example[i,0] = stim[-1,23]
        stim_example[i,1] = stim[-1,24]
        stim_example[i,2] = stim[-1,25]
    # j += 1
    # if j == interval:
    #     index += 1
    #     j = 0
    data = state_to_data(i, data, states)



plt.figure()
plt.plot(data_sns_toolbox['t'], data_sns_toolbox['inp'][1, :], label='sns-toolbox')
plt.plot(data_sns_toolbox['t'], data[0], label='snsTorch')
plt.title('Input')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.figure()
plt.plot(data_sns_toolbox['t'], data_sns_toolbox['bp'][1, :], label='sns-toolbox')
plt.plot(data_sns_toolbox['t'], data[1], label='snsTorch')
plt.title('BPO')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.figure()
plt.plot(data_sns_toolbox['t'], data_sns_toolbox['lb'][1, :], label='sns-toolbox')
plt.plot(data_sns_toolbox['t'], data[2], label='snsTorch')
plt.title('LP')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.figure()
plt.plot(data_sns_toolbox['t'], data_sns_toolbox['bp'][1, :], label='sns-toolbox')
plt.plot(data_sns_toolbox['t'], data[3], label='snsTorch')
plt.title('BPF')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.show()