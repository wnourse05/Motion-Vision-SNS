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

from Tuning.gen_t4_params import tune_t4

import numpy as np
from utilities import cutoff_fastest

def debug_print(line, debug):
    if debug:
        print(line)
        return line
def tune_neurons(cutoffs, mode, debug=False, save=True):
    if mode == 'on':
        cutoff_retina, cutoff_l1_low, cutoff_l1_high, cutoff_l3, cutoff_mi1, cutoff_mi9, cutoff_ct1_on, cutoff_t4 = cutoffs
    elif mode == 'off':
        cutoff_retina, cutoff_l2_low, cutoff_l2_high, cutoff_l3, cutoff_tm1, cutoff_tm9, cutoff_ct1_off = cutoffs
    else:
        cutoff_retina, cutoff_l1_low, cutoff_l1_high, cutoff_l2_low, cutoff_l2_high, cutoff_l3, cutoff_mi1, cutoff_mi9,\
            cutoff_tm1, cutoff_tm9, cutoff_ct1_on, cutoff_ct1_off = cutoffs

    if debug:
        print('Tuning Retina')
    retina = tune_retina(cutoff_retina, save=save)
    if debug:
        print('Tuning L3')
    l3, l3_conn = tune_l3(cutoff_l3, save=save)
    data = {'Retina': retina,
            'L3': l3, 'L3Conn': l3_conn}
    if mode == 'on' or mode != 'off':
        if debug:
            print('Tuning L1')
        l1, l1_conn = tune_l1(cutoff_l1_low, cutoff_l1_high, retina, save=save)
        if debug:
            print('Tuning Mi1')
        mi1, mi1_conn = tune_mi1(cutoff_mi1, retina, l1, save=save)
        if debug:
            print('Tuning Mi9')
        mi9, mi9_conn = tune_mi9(cutoff_mi9, save=save)
        if debug:
            print('Tuning CT1 (On)')
        ct1_on, ct1_on_conn = tune_ct1_on(cutoff_ct1_on, retina, l1, mi1, mi1_conn, save=save)
        if debug:
            print('Tuning Mi1 -> T4')
        t4, t4_conn = tune_t4(cutoff_t4, retina, l1, mi1, mi1_conn, save=save)
        data.update({'L1': l1, 'L1Conn': l1_conn,
                     'Mi1': mi1, 'Mi1Conn': mi1_conn,
                     'Mi9': mi9, 'Mi9Conn': mi9_conn,
                     'CT1On': ct1_on, 'CT1OnConn': ct1_on_conn,
                     'T4': t4, 'T4Conn': t4_conn})
    if mode == 'off' or mode != 'on':
        if debug:
            print('Tuning L2')
        l2, l2_conn = tune_l2(cutoff_l2_low, cutoff_l2_high, retina, save=save)
        if debug:
            print('Tuning Tm1')
        tm1, tm1_conn = tune_tm1(cutoff_tm1, retina, l2, save=save)
        if debug:
            print('Tuning Tm9')
        tm9, tm9_conn = tune_tm9(cutoff_tm9, save=save)
        if debug:
            print('Tuning CT1 (Off)')
        ct1_off, ct1_off_conn = tune_ct1_off(cutoff_ct1_off, retina, l2, tm1_conn, tm1, save=save)
        data.update({'L2': l2, 'L2Conn': l2_conn,
                     'Tm1': tm1, 'Tm1Conn': tm1_conn,
                     'Tm9': tm9, 'Tm9Conn': tm9_conn,
                     'CT1Off': ct1_off, 'CT1OffConn': ct1_off_conn, })

    return data
