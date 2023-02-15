from Tuning.gen_retina_params import tune_retina
from Tuning.gen_l1_params import tune_l1
from Tuning.gen_l2_params import tune_l2
from Tuning.gen_l3_params import tune_l3

from Tuning.gen_mi1_params import tune_mi1
from Tuning.gen_mi9_params import tune_mi9
from Tuning.gen_tm1_params import tune_tm1
from Tuning.gen_tm9_params import tune_tm9

from Tuning.gen_ct1_on_params import tune_ct1_on
from Tuning.gen_ct1_off_params import tune_ct1_off

import numpy as np
from utilities import cutoff_fastest

def debug_print(line, debug):
    if debug:
        print(line)
        return line
def tune_neurons(cutoffs, debug=False):
    cutoff_retina, cutoff_l1_low, cutoff_l1_high, cutoff_l2_low, cutoff_l2_high, cutoff_l3, cutoff_mi1, cutoff_mi9,\
        cutoff_tm1, cutoff_tm9, cutoff_ct1_on, cutoff_ct1_off = cutoffs

    if debug:
        print('Tuning Retina')
    tune_retina(cutoff_retina)
    if debug:
        print('Tuning L1')
    tune_l1(cutoff_l1_low, cutoff_l1_high)
    if debug:
        print('Tuning L2')
    tune_l2(cutoff_l2_low, cutoff_l2_high)
    if debug:
        print('Tuning L3')
    tune_l3(cutoff_l3)
    if debug:
        print('Tuning Mi1')
    tune_mi1(cutoff_mi1)
    if debug:
        print('Tuning Mi9')
    tune_mi9(cutoff_mi9)
    if debug:
        print('Tuning Tm1')
    tune_tm1(cutoff_tm1)
    if debug:
        print('Tuning Tm9')
    tune_tm9(cutoff_tm9)
    if debug:
        print('Tuning CT1 (On)')
    tune_ct1_on(cutoff_ct1_on)
    if debug:
        print('Tuning CT1 (Off)')
    tune_ct1_off(cutoff_ct1_off)
