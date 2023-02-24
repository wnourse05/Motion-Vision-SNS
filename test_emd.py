import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

from utilities import dt, cutoff_fastest
from motion_vision_networks import gen_test_emd
from sns_toolbox.renderer import render

#                   Retina          L1                                  L2                              L3                  Mi1         Mi9             Tm1             Tm9             CT1_On          CT1_Off
cutoffs = np.array([cutoff_fastest, cutoff_fastest/10, cutoff_fastest, cutoff_fastest/5, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest])

model, net = gen_test_emd((5,7))
# render(net)

on_stimulus = np.array([[0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1]])

class Stimulus:
    def __init__(self, stimulus_array, interval):
        self.stimulus_array = stimulus_array
        self.interval = interval
        self.index = 0
        self.interval_ctr = -1
        self.num_rows = np.shape(stimulus_array)[0]

    def get_stimulus(self):
        self.interval_ctr += 1
        if self.interval_ctr >= self.interval:
            self.interval_ctr = 0
            self.index += 1
            if self.index >= self.num_rows:
                self.index = 0

        return self.stimulus_array[self.index, :]

stim = Stimulus(on_stimulus,4)

for i in range(32):
    val = stim.get_stimulus()
    print(val)

