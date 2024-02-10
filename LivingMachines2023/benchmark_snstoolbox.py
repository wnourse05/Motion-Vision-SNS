from motion_vision_networks import gen_motion_vision_no_output
import torch
from timeit import default_timer
from tqdm import tqdm
import pickle
import numpy as np
import blosc

filename = 'params_net_20230327.pc'
with open(filename, 'rb') as f:
        compressed = f.read()
decompressed = blosc.decompress(compressed)
params_sns = pickle.loads(decompressed)

dtype = torch.float32
device = 'cuda'
rows = np.geomspace(3,24)
cols = np.geomspace(5,64)
rows = [3,19,24,38,77,154,308,616,1232]
cols = [5,51,64,102,205,410,820,1640,3280]
num_trials = 10
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
        name = '../vary_size_snstoolbox_'+device+'.p'
        pickle.dump(data, open(name, 'wb'))
print(average)
print(std)
print(var)
print(low)
print(upp)
