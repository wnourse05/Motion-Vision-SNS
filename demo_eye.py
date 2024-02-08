from LivingMachines2023.utilities import load_data
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from motion_vision_net import SNSMotionVisionMerged

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
    stim_0_full = torch.cat((frame_5.unsqueeze(0), frame_4.unsqueeze(0), frame_3.unsqueeze(0), frame_2.unsqueeze(0),
                             frame_1.unsqueeze(0), frame_0.unsqueeze(0)))
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

stims = get_stimulus()
params_sns = load_data('LivingMachines2023/params_net_20230327.pc')

# EMD Behavior
vel = 30.0
angle = 0

dtype = torch.float32
device = 'cpu'

interval = convert_deg_vel_to_interval(vel, params_sns['dt'])
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
model_torch = SNSMotionVisionMerged(params_sns['dt'],(7,7), 1, params=params, dtype=dtype, device=device)
model_torch = torch.compile(model_torch)

num_samples = stim.shape[0]
t = np.linspace(0, params_sns['dt'] * num_samples * interval, num=num_samples * interval)
data_cw = torch.zeros(num_samples*interval, device=device)
data_ccw = torch.zeros(num_samples*interval, device=device)
data = [data_cw, data_ccw]

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
# state_horizontal = torch.zeros(2, dtype=dtype).to(device)
# states = [state_input, state_bp_on_input, state_bp_on_fast, state_bp_on_slow, state_bp_on_output, state_lowpass,
#           state_bp_off_input, state_bp_off_fast, state_bp_off_slow, state_bp_off_output, state_enhance_on,
#           state_direct_on, state_suppress_on, state_enhance_off, state_direct_off, state_suppress_off, state_ccw_on,
#           state_cw_on, state_ccw_off, state_cw_off, state_horizontal]

def state_to_data(index, data, model):
    data[0][index] = model.state_horizontal[0]
    data[1][index] = model.state_horizontal[1]
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


plt.figure()
plt.plot(t, data[0].detach().to('cpu').numpy(), label='cw')
plt.plot(t, data[1].detach().to('cpu').numpy(), label='ccw')
plt.title('Prediction')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.legend()

plt.show()