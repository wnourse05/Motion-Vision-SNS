from LivingMachines2023.utilities import load_data
from LivingMachines2023.motion_vision_networks import gen_motion_vision_no_output
import torch
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

# shape = [24,64]
with torch.no_grad():
    for i in tqdm(range(len(rows))):
        shape = [int(rows[i]), int(cols[i])]
        model, net = gen_motion_vision_no_output(params_sns, shape, 'torch', device)

        stim = torch.rand(shape,dtype=dtype, device=device).flatten()

        # Warmup
        for j in range(5):
            model(stim)

        times = torch.zeros(num_trials)
        for j in range(num_trials):
            start = default_timer()
            states = model(stim)
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
