from utilities import load_data, save_data, add_scaled_bandpass_filter
from sns_toolbox.networks import Network
from sns_toolbox.renderer import render
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

params = load_data('params_net_20230327.pc')

params_bp = params['bp']

net = Network()

add_scaled_bandpass_filter(net, params_bp['cutoffLow'], params_bp['cutoffHigh'],
                           params_bp['gain'], invert=True, name='BP')
net.add_input('BP_in')
net.add_output('BP_in')
net.add_output('BP_fast')
net.add_output('BP_slow')
net.add_output('BP_out')

dt = 0.01   # ms
model = net.compile(dt=dt)
# render(net)

t = np.arange(0,50, step=dt)
inp = np.zeros_like(t)
inp[1000:int(0.75*len(t))] = -1.0

data = np.zeros([len(t),4])

for i in tqdm(range(len(t))):
    data[i,:] = model([inp[i]])

data = data.transpose()

save_file = {'t': t, 'inp': inp, 'data_sns_toolbox': data}

save_data(save_file, 'bp_step_response.pc')


