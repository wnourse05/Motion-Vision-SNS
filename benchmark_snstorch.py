from LivingMachines2023.utilities import load_data
import torch
import torch.nn as nn
from motion_vision_net import SNSMotionVisionEye
from timeit import default_timer
from tqdm import tqdm
import pickle
import numpy as np


params_sns = load_data('LivingMachines2023/params_net_20230327.pc')

dtype = torch.float32
device = 'cpu'
rows = np.geomspace(3,24)
cols = np.geomspace(5,64)
num_trials = 100
low_percentile = 0.05
high_percentile = 0.95

average = torch.zeros(len(rows))
std = torch.zeros(len(rows))
var = torch.zeros(len(rows))
low = torch.zeros(len(rows))
upp = torch.zeros(len(rows))

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

# shape = [24,64]
with torch.no_grad():
    for i in tqdm(range(len(rows))):
        shape = [int(rows[i]), int(cols[i])]
        # torch.jit.enable_onednn_fusion(True)  # slower
        model_torch = SNSMotionVisionEye(params_sns['dt'], shape, 1, params=params, dtype=dtype, device=device)
        model_torch.eval()
        model_torch = torch.jit.freeze(model_torch)
        model_torch = torch.compile(model_torch)
        # model_torch = torch.jit.optimize_for_inference(model_torch)   # slows down

        stim = torch.rand(shape,dtype=dtype, device=device)

        # Warmup
        for j in range(5):
            model_torch(stim.to(device))

        times = torch.zeros(num_trials)
        for j in range(num_trials):
            start = default_timer()
            states = model_torch(stim)
            if device == 'cuda':
                states = states.to('cpu')
            # output = states[20]#.to('cpu')
            end = default_timer()
            times[j] = (end-start)
        average[i] = torch.mean(times)
        std[i] = torch.std(times)
        var[i] = torch.var(times)
        low[i] = torch.quantile(times, low_percentile)
        upp[i] = torch.quantile(times, high_percentile)

        data = {'rows': rows,
                'cols': cols,
                'std': std,
                'var': var,
                'low': low,
                'upper': upp}
        name = 'vary_size_snstorch_'+device+'.p'
        pickle.dump(data, open(name, 'wb'))
print(average)
print(std)
print(var)
print(low)
print(upp)
