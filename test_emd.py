import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

from utilities import dt, cutoff_fastest
from motion_vision_networks import gen_test_emd

#                   Retina          L1                                  L2                              L3                  Mi1         Mi9             Tm1             Tm9             CT1_On          CT1_Off
cutoffs = np.array([cutoff_fastest, cutoff_fastest/10, cutoff_fastest, cutoff_fastest/5, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest])

model, net = gen_test_emd((5,7))
