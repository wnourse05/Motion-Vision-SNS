from utilities import load_data, save_data
from motion_vision_networks import gen_motion_vision
from sns_toolbox.renderer import render
import numpy as np
from tqdm import tqdm
import torch

def run_net(dt, model, net, stim, shape, device, vel_index, angle_index):
    model.reset()
    stim_size = stim.size()
    num_samples = list(stim_size)[0]
    t = torch.linspace(0, dt * num_samples, steps=num_samples)
    flat_size = shape[0] * shape[1]

    data = torch.zeros([num_samples, net.get_num_outputs_actual()], device=device)

    for i in tqdm(range(num_samples), leave=False):
        stim_i = stim[i,:]
        data[i,:] = model(stim_i.to(device))
        stim_i = stim_i.to('cpu')

    data = data.transpose(0,1)
    data = data.to('cpu')
    retina = data[:flat_size, :]
    l1 = data[flat_size:2*flat_size, :]
    l2 = data[2*flat_size:3*flat_size, :]
    l3 = data[3*flat_size:4*flat_size, :]
    mi1 = data[4*flat_size:5*flat_size, :]
    mi9 = data[5*flat_size:6*flat_size, :]
    tm1 = data[6*flat_size:7*flat_size, :]
    tm9 = data[7*flat_size:8*flat_size, :]
    ct1_on = data[8*flat_size:9*flat_size, :]
    ct1_off = data[9*flat_size:10*flat_size, :]
    t4a = data[10*flat_size:11*flat_size, :]
    t4b = data[11*flat_size:12*flat_size, :]
    t4c = data[12*flat_size:13*flat_size, :]
    t4d = data[13*flat_size:14*flat_size, :]
    t5a = data[14*flat_size:15*flat_size, :]
    t5b = data[15*flat_size:16*flat_size, :]
    t5c = data[16*flat_size:17*flat_size, :]
    t5d = data[17*flat_size:, :]

    del data
    t4_a_peak = torch.max(t4a[24,:])
    t4_b_peak = torch.max(t4b[24,:])
    t4_c_peak = torch.max(t4c[24,:])
    t4_d_peak = torch.max(t4d[24,:])
    t5_a_peak = torch.max(t5a[24,:])
    t5_b_peak = torch.max(t5b[24,:])
    t5_c_peak = torch.max(t5c[24,:])
    t5_d_peak = torch.max(t5d[24,:])

    sim_params = {'velindex': vel_index,
                  'angleindex': angle_index,
                  't': t}
    retina = {'retina': retina}
    lamina = {'l1': l1,
              'l2': l2,
              'l3': l3}
    medulla_on = {'mi1': mi1,
                  'mi9': mi9,
                  'ct1on': ct1_on}
    medulla_off = {'tm1': tm1,
                   'tm9': tm9,
                   'ct1off': ct1_off}
    emd_on = {'t4a': t4a,
              't4b': t4b,
              't4c': t4c,
              't4d': t4d}
    emd_off = {'t5a': t5a,
               't5b': t5b,
               't5c': t5c,
               't5d': t5d}
    peaks = {'t4apeak': t4_a_peak,
             't4bpeak': t4_b_peak,
             't4cpeak': t4_c_peak,
             't4dpeak': t4_d_peak,
             't5apeak': t5_a_peak,
             't5bpeak': t5_b_peak,
             't5cpeak': t5_c_peak,
             't5dpeak': t5_d_peak}
    save_data(sim_params,'Data/sim_params_%i_%i.pc'%(vel_index,angle_index))
    save_data(retina, 'Data/retina_%i_%i.pc' % (vel_index, angle_index))
    save_data(lamina, 'Data/lamina_%i_%i.pc' % (vel_index, angle_index))
    save_data(medulla_on, 'Data/medulla_on_%i_%i.pc' % (vel_index, angle_index))
    save_data(medulla_off, 'Data/medulla_off_%i_%i.pc' % (vel_index, angle_index))
    save_data(emd_on, 'Data/emd_on_%i_%i.pc' % (vel_index, angle_index))
    save_data(emd_off, 'Data/emd_off_%i_%i.pc' % (vel_index, angle_index))
    save_data(peaks, 'Data/peaks_%i_%i.pc' % (vel_index, angle_index))

print('Loading Params')
params = load_data('params_net_10_180.pc')
num_intervals = 4
vels = np.linspace(10, 180, num=num_intervals)
angular_res = 30
angles = np.arange(0, 360, angular_res)
dt = params['dt']
shape = (7,7)
device = 'cuda'
print('Building Model')
model, net = gen_motion_vision(params, shape, device)
# render(net)
print('Starting Sim Loop')
for vel in tqdm(range(num_intervals), leave=False, colour='green'):
    for angle in tqdm(range(len(angles)), leave=False, colour='blue'):
        stim = load_data('Stimuli/stim_%i_%i.pc'%(vel,angle))
        run_net(dt, model, net, stim['stim'], shape, device, vel, angle)
        del stim
