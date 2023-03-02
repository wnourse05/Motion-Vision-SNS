from utilities import save_data, gen_gratings
import numpy as np
freqs = np.array([5.0, 200.0])
dt = 0.1
print('Generating first stimulus')
stim_0, _ = gen_gratings([7,7], freqs[0], 'a', 3, dt, device='cpu')
print('Generating second stimulus')
stim_1, _ = gen_gratings([7,7], freqs[1], 'a', 5, dt, device='cpu')

data = {'dt': dt,
        'freqs': freqs,
        'stims': [stim_0, stim_1]}

save_data(data, 'mcmc_stims.p')
