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
    stim_0_full = torch.cat((frame_0.unsqueeze(0), frame_1.unsqueeze(0), frame_2.unsqueeze(0), frame_3.unsqueeze(0), frame_4.unsqueeze(0), frame_5.unsqueeze(0)))
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

    stims = {'0': stim_0, '45': stim_45, '90': stim_90, '135': stim_135, '180': stim_180, '225': stim_225, '270': stim_270, '315': stim_315, 'full': stim_0_full}

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
    single_e = e[index, :]
    del e
    d_on = data[4*size:5*size, :]
    single_d_on = d_on[index, :]
    del d_on
    d_off = data[5*size:6*size, :]
    single_d_off = d_off[index, :]
    del d_off
    s_on = data[6*size:7*size, :]
    single_s_on = s_on[index, :]
    del s_on
    s_off = data[7*size:8*size, :]
    single_s_off = s_off[index, :]
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
    stim = torch.vstack((stims['%i'%angle], stims['%i'%angle], stims['%i'%angle], stims['%i'%angle],stims['%i'%angle])).to(device)
    # stim = torch.vstack((stims['%i' % angle],)).to(device)
    data = run_sns_toolbox(model, net, stim, device, flat_size, dt, interval)
    return data, net


stims = get_stimulus()
params_sns = load_data('LivingMachines2023/params_net_20230327.pc')

# EMD Behavior
vel = 30.0
angle = 0
# data_sns_toolbox, net = collect_data_sns_toolbox(vel, angle, stims, center=True, scale=False)

dtype = torch.float32
device = 'cpu'

interval = convert_deg_vel_to_interval(vel, params_sns['dt'])
stim = stims['full'].to(device)
stim = torch.cat((stims['full'], stims['full'], stims['full'], stims['full'], stims['full'])).to(device)

params = nn.ParameterDict({
    'reversalEx': nn.Parameter(torch.tensor([5.0], dtype=dtype).to(device)),
    'reversalIn': nn.Parameter(torch.tensor([-2.0], dtype=dtype).to(device)),
    'reversalMod': nn.Parameter(torch.tensor([-0.1], dtype=dtype).to(device)),
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
    'conductanceLEO': nn.Parameter(torch.tensor([params_sns['e']['g']],dtype=dtype).to(device)),
    'freqEO': nn.Parameter(torch.tensor([params_sns['e']['cutoff']],dtype=dtype).to(device)),
    'conductanceBODO': nn.Parameter(torch.tensor([params_sns['d_on']['g']],dtype=dtype).to(device)),
    'freqDO': nn.Parameter(torch.tensor([params_sns['d_on']['cutoff']],dtype=dtype).to(device)),
    'conductanceDOSO': nn.Parameter(torch.tensor([params_sns['s_on']['g']],dtype=dtype).to(device)),
    'freqSO': nn.Parameter(torch.tensor([params_sns['s_on']['cutoff']],dtype=dtype).to(device)),
    'conductanceLEF': nn.Parameter(torch.tensor([params_sns['e']['g']],dtype=dtype).to(device)),
    'freqEF': nn.Parameter(torch.tensor([params_sns['e']['cutoff']],dtype=dtype).to(device)),
    'conductanceBFDF': nn.Parameter(torch.tensor([params_sns['d_off']['g']],dtype=dtype).to(device)),
    'freqDF': nn.Parameter(torch.tensor([params_sns['d_off']['cutoff']],dtype=dtype).to(device)),
    'conductanceDFSF': nn.Parameter(torch.tensor([params_sns['s_off']['g']],dtype=dtype).to(device)),
    'freqSF': nn.Parameter(torch.tensor([params_sns['s_off']['cutoff']],dtype=dtype).to(device)),
    'conductanceEOOn': nn.Parameter(torch.tensor([params_sns['on']['g']['enhance']],dtype=dtype).to(device)),
    'conductanceDOOn': nn.Parameter(torch.tensor([params_sns['on']['g']['direct']],dtype=dtype).to(device)),
    'conductanceSOOn': nn.Parameter(torch.tensor([params_sns['on']['g']['suppress']],dtype=dtype).to(device)),
    'conductanceEFOff': nn.Parameter(torch.tensor([params_sns['off']['g']['enhance']],dtype=dtype).to(device)),
    'conductanceDFOff': nn.Parameter(torch.tensor([params_sns['off']['g']['direct']],dtype=dtype).to(device)),
    'conductanceSFOff': nn.Parameter(torch.tensor([0.5],dtype=dtype).to(device)),
})
model_torch = SNSMotionVisionEye(params_sns['dt'],(7,7), 1, params=params, dtype=dtype, device=device)
model_torch.eval()
model_torch = torch.jit.freeze(model_torch)
model_torch = torch.compile(model_torch)

num_samples = stim.shape[0]
t = np.linspace(0, params_sns['dt'] * num_samples * interval, num=num_samples * interval)
data_in = torch.zeros(num_samples*interval, device=device)
data_bo = torch.zeros(num_samples*interval, device=device)
data_l = torch.zeros(num_samples*interval, device=device)
data_bf = torch.zeros(num_samples*interval, device=device)
data_bo_in = torch.zeros(num_samples*interval, device=device)
data_bo_fast = torch.zeros(num_samples*interval, device=device)
data_bo_slow = torch.zeros(num_samples*interval, device=device)
data_enhance_on = torch.zeros(num_samples*interval, device=device)
data_direct_on = torch.zeros(num_samples*interval, device=device)
data_suppress_on = torch.zeros(num_samples*interval, device=device)
data_ccw_on = torch.zeros(num_samples*interval, device=device)
data_cw_on = torch.zeros(num_samples*interval, device=device)
data_ccw_off = torch.zeros(num_samples*interval, device=device)
data_cw_off = torch.zeros(num_samples*interval, device=device)
data = [data_in, data_bo, data_l, data_bf, data_bo_in, data_bo_fast, data_bo_slow, data_enhance_on, data_direct_on,
        data_suppress_on, data_ccw_on, data_cw_on, data_ccw_off, data_cw_off]

stim_example = torch.zeros([len(t),3])

shape = [7,7]
shape_emd = [x - 2 for x in shape]
# state_input = torch.zeros(shape, dtype=dtype).to(device)
# state_bp_on_input = torch.ones(shape, dtype=dtype).to(device)
# state_bp_on_fast = torch.zeros(shape, dtype=dtype).to(device)
# state_bp_on_slow = torch.zeros(shape, dtype=dtype).to(device)
# state_bp_on_output = torch.ones(shape, dtype=dtype).to(device)
# state_lowpass = torch.ones(shape, dtype=dtype).to(device)
# state_bp_off_input = torch.ones(shape, dtype=dtype).to(device)
# state_bp_off_fast = torch.zeros(shape, dtype=dtype).to(device)
# state_bp_off_slow = torch.zeros(shape, dtype=dtype).to(device)
# state_bp_off_output = torch.ones(shape, dtype=dtype).to(device)
# state_enhance_on = torch.ones(shape, dtype=dtype).to(device)
# state_direct_on = torch.zeros(shape, dtype=dtype).to(device)
# state_suppress_on = torch.zeros(shape, dtype=dtype).to(device)
# state_enhance_off = torch.ones(shape, dtype=dtype).to(device)
# state_direct_off = torch.zeros(shape, dtype=dtype).to(device)
# state_suppress_off = torch.zeros(shape, dtype=dtype).to(device)
# state_ccw_on = torch.zeros(shape_emd, dtype=dtype).to(device)
# state_cw_on = torch.zeros(shape_emd, dtype=dtype).to(device)
# state_ccw_off = torch.zeros(shape_emd, dtype=dtype).to(device)
# state_cw_off = torch.zeros(shape_emd, dtype=dtype).to(device)
# states = [state_input, state_bp_on_input, state_bp_on_fast, state_bp_on_slow, state_bp_on_output, state_lowpass,
#           state_bp_off_input, state_bp_off_fast, state_bp_off_slow, state_bp_off_output, state_enhance_on,
#           state_direct_on, state_suppress_on, state_enhance_off, state_direct_off, state_suppress_off, state_ccw_on,
#           state_cw_on, state_ccw_off, state_cw_off]

def state_to_data(index, data, model):
    data[0][index] = model.state_input[3,3]
    # data[4][index] = states[1][3,3]
    # data[5][index] = states[2][3,3]
    # data[6][index] = states[3][3,3]
    data[1][index] = model.bandpass_on.state_output[3,3]
    data[2][index] = model.state_lowpass[3,3]
    # state_bp_off_input = states[6]
    # state_bp_off_fast = states[7]
    # state_bp_off_slow = states[8]
    data[3][index] = model.bandpass_off.state_output[3,3]
    data[4][index] = model.state_enhance_on[3,3]
    data[5][index] = model.state_direct_on[3,3]
    data[6][index] = model.state_suppress_on[3,3]
    data[7][index] = model.state_enhance_off[3,3]
    data[8][index] = model.state_direct_off[3,3]
    data[9][index] = model.state_suppress_off[3,3]
    data[10][index] = model.state_ccw_on[2,2]
    data[11][index] = model.state_cw_on[2,2]
    data[12][index] = model.state_ccw_off[2,2]
    data[13][index] = model.state_cw_off[2,2]
    return data

index = 0
j = 0
with torch.no_grad():
    for i in tqdm(range(len(t)), leave=False, colour='blue'):
        if index < num_samples:
            states = model_torch(stim[index,:,:])
            # stim_example[i, 0] = stim[index, 23]
            # stim_example[i, 1] = stim[index, 24]
            # stim_example[i, 2] = stim[index, 25]
        else:
            states = model_torch(stim[-1,:,:])
            # stim_example[i,0] = stim[-1,23]
            # stim_example[i,1] = stim[-1,24]
            # stim_example[i,2] = stim[-1,25]
        j += 1
        if j == interval:
            index += 1
            j = 0
        data = state_to_data(i, data, model_torch)

toolbox = True
if toolbox:
    data_sns_toolbox, net = collect_data_sns_toolbox(vel, angle, stims, center=True, scale=False)

plt.figure()
num_rows = 2
num_cols = 7

plt.subplot(num_rows, num_cols, 1)
if toolbox:
    plt.plot(data_sns_toolbox['t'], data_sns_toolbox['inp'][1, :], label='sns-toolbox')
plt.plot(t, data[0].detach().to('cpu').numpy(), label='snsTorch')
plt.title('Input')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.subplot(num_rows, num_cols, 2)
if toolbox:
    plt.plot(data_sns_toolbox['t'], data_sns_toolbox['bp'][1, :], label='sns-toolbox')
plt.plot(t, data[1].detach().to('cpu').numpy(), label='snsTorch')
plt.title('BPO')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.subplot(num_rows, num_cols, 3)
if toolbox:
    plt.plot(data_sns_toolbox['t'], data_sns_toolbox['lb'][1, :], label='sns-toolbox')
plt.plot(t, data[2].detach().to('cpu').numpy(), label='snsTorch')
plt.title('LP')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.subplot(num_rows, num_cols, 4)
if toolbox:
    plt.plot(data_sns_toolbox['t'], data_sns_toolbox['bp'][1, :], label='sns-toolbox')
plt.plot(t, data[3].detach().to('cpu').numpy(), label='snsTorch')
plt.title('BPF')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.subplot(num_rows, num_cols, 5)
if toolbox:
    plt.plot(data_sns_toolbox['t'], data_sns_toolbox['e_on'], label='sns-toolbox')
plt.plot(t, data[4].detach().to('cpu').numpy(), label='snsTorch')
plt.title('EO')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.subplot(num_rows, num_cols, 6)
if toolbox:
    plt.plot(data_sns_toolbox['t'], data_sns_toolbox['d_on'], label='sns-toolbox')
plt.plot(t, data[5].detach().to('cpu').numpy(), label='snsTorch')
plt.title('DO')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.subplot(num_rows, num_cols, 7)
if toolbox:
    plt.plot(data_sns_toolbox['t'], data_sns_toolbox['s_on'], label='sns-toolbox')
plt.plot(t, data[6].detach().to('cpu').numpy(), label='snsTorch')
plt.title('SO')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.subplot(num_rows, num_cols, 8)
if toolbox:
    plt.plot(data_sns_toolbox['t'], data_sns_toolbox['e_off'], label='sns-toolbox')
plt.plot(t, data[7].detach().to('cpu').numpy(), label='snsTorch')
plt.title('EF')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.subplot(num_rows, num_cols, 9)
if toolbox:
    plt.plot(data_sns_toolbox['t'], data_sns_toolbox['d_off'], label='sns-toolbox')
plt.plot(t, data[8].detach().to('cpu').numpy(), label='snsTorch')
plt.title('DF')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.subplot(num_rows, num_cols, 10)
if toolbox:
    plt.plot(data_sns_toolbox['t'], data_sns_toolbox['s_off'], label='sns-toolbox')
plt.plot(t, data[9].detach().to('cpu').numpy(), label='snsTorch')
plt.title('SF')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.subplot(num_rows, num_cols, 11)
if toolbox:
    plt.plot(data_sns_toolbox['t'], data_sns_toolbox['on_b'], label='sns-toolbox')
plt.plot(t, data[10].detach().to('cpu').numpy(), label='snsTorch')
plt.title('CCW On')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.subplot(num_rows, num_cols, 12)
if toolbox:
    plt.plot(data_sns_toolbox['t'], data_sns_toolbox['on_a'], label='sns-toolbox')
plt.plot(t, data[11].detach().to('cpu').numpy(), label='snsTorch')
plt.title('CW On')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.subplot(num_rows, num_cols, 13)
if toolbox:
    plt.plot(data_sns_toolbox['t'], data_sns_toolbox['off_b'], label='sns-toolbox')
plt.plot(t, data[12].detach().to('cpu').numpy(), label='snsTorch')
plt.title('CCW Off')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.subplot(num_rows, num_cols, 14)
if toolbox:
    plt.plot(data_sns_toolbox['t'], data_sns_toolbox['off_a'], label='sns-toolbox')
plt.plot(t, data[13].detach().to('cpu').numpy(), label='snsTorch')
plt.title('CW Off')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.show()