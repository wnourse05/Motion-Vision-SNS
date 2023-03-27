from utilities import gen_gratings, save_data, load_data
import numpy as np
from tqdm import tqdm

num_intervals = 4
vels = np.linspace(10, 180, num=num_intervals)
angular_res = 30
angles = np.arange(0, 360, angular_res)
dim = 7
fov_res = 5
fov = fov_res*dim
wavelength = fov
net_params = load_data('params_net_10_180.pc')
dt = net_params['dt']

for vel in tqdm(range(num_intervals), colour='green', leave=False):

        stim_period = 2 * wavelength / vels[vel]
        num_steps = int(stim_period / (dt / 1000))
        for angle in tqdm(range(len(angles)), colour='blue', leave=False):
            stim = {'vels': vels, 'velIndex': vel, 'angles': angles, 'angleIndex': angle, 'fov': fov, 'fovRes': fov_res,
                    'wavelength': wavelength,
                    'stim': gen_gratings(wavelength, angles[angle], vels[vel], dt, num_steps, fov=fov, res=fov_res, use_torch=True, square=True)}
            save_data(stim, 'Stimuli/stim_%i_%i.pc'%(vel, angle))
            del stim
