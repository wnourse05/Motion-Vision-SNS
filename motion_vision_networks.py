import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

from sns_toolbox.connections import NonSpikingSynapse, NonSpikingPatternConnection
from sns_toolbox.networks import Network
from sns_toolbox.renderer import render
from utilities import add_lowpass_filter, activity_range, load_data, add_scaled_bandpass_filter, backend, NonSpikingOneToOneConnection, synapse_target, device, cutoff_fastest, reversal_ex, reversal_in, calc_cap_from_cutoff
from Tuning.tune_neurons import tune_neurons

def __gen_receptive_fields__(params):
    g_params = params['g']
    g = np.array([[g_params['outer'], g_params['outer'], g_params['outer'], g_params['outer'], g_params['outer']],
                  [g_params['outer'], g_params['middle'], g_params['middle'], g_params['middle'], g_params['outer']],
                  [g_params['outer'], g_params['middle'], g_params['center'], g_params['middle'], g_params['outer']],
                  [g_params['outer'], g_params['middle'], g_params['middle'], g_params['middle'], g_params['outer']],
                  [g_params['outer'], g_params['outer'], g_params['outer'], g_params['outer'], g_params['outer']]])
    reversal_params = params['reversal']
    reversal = np.array([[reversal_params['outer'], reversal_params['outer'], reversal_params['outer'],
                          reversal_params['outer'], reversal_params['outer']],
                         [reversal_params['outer'], reversal_params['middle'], reversal_params['middle'],
                          reversal_params['middle'], reversal_params['outer']],
                         [reversal_params['outer'], reversal_params['middle'], reversal_params['center'],
                          reversal_params['middle'], reversal_params['outer']],
                         [reversal_params['outer'], reversal_params['middle'], reversal_params['middle'],
                          reversal_params['middle'], reversal_params['outer']],
                         [reversal_params['outer'], reversal_params['outer'], reversal_params['outer'],
                          reversal_params['outer'], reversal_params['outer']]])
    e_lo = np.zeros_like(g)
    e_hi = np.zeros_like(g) + activity_range

    return g, reversal, e_lo, e_hi

def __all_quadrants__(matrix):
    c = np.rot90(matrix,1)
    a = np.rot90(matrix, 2)
    d = np.rot90(matrix, 3)
    return a, c, d

def __transmission_params__(k, excite=True):
    if excite:
        rev = reversal_ex
        cond = k*activity_range/(rev-k*activity_range)
    else:
        rev = reversal_in
        cond = -k*activity_range/(rev+k*activity_range)
    return cond, rev

def gen_single_column(dt, cutoffs=None):
    """
    ####################################################################################################################
    NETWORK
    """
    if cutoffs is not None:
        tune_neurons(cutoffs, 'all')
    net = Network('Motion Vision Single Column')

    """
    ####################################################################################################################
    RETINA
    """

    params_node_retina = load_data('params_node_retina.p')
    add_lowpass_filter(net, params_node_retina['params']['cutoff'], name='Retina', invert=params_node_retina['params']['invert'], initial_value=params_node_retina['params']['initialValue'], bias=params_node_retina['params']['bias'], color='black')

    net.add_input('Retina')
    net.add_output('Retina', name='OutR')

    """
    ####################################################################################################################
    LAMINA
    """
    params_conn_l1 = load_data('params_conn_l1.p')
    params_conn_l2 = load_data('params_conn_l2.p')
    params_conn_l3 = load_data('params_conn_l3.p')
    synapse_r_l1 = NonSpikingSynapse(max_conductance=params_conn_l1['g']['center'], reversal_potential=params_conn_l1['reversal']['center'], e_lo=0.0, e_hi=activity_range)
    synapse_r_l2 = NonSpikingSynapse(max_conductance=params_conn_l2['g']['center'], reversal_potential=params_conn_l2['reversal']['center'], e_lo=0.0, e_hi=activity_range)
    synapse_r_l3 = NonSpikingSynapse(max_conductance=params_conn_l3['g']['center'], reversal_potential=params_conn_l3['reversal']['center'], e_lo=0.0, e_hi=activity_range)

    params_node_l1 = load_data('params_node_l1.p')
    add_scaled_bandpass_filter(net, params_node_l1['params']['cutoffLow'], params_node_l1['params']['cutoffHigh'],
                               params_node_l1['params']['gain'], invert=params_node_l1['params']['invert'], name='L1')
    params_node_l2 = load_data('params_node_l2.p')
    add_scaled_bandpass_filter(net, params_node_l2['params']['cutoffLow'], params_node_l2['params']['cutoffHigh'],
                               params_node_l2['params']['gain'], invert=params_node_l2['params']['invert'], name='L2')
    params_node_l3 = load_data('params_node_l3.p')
    add_lowpass_filter(net, cutoff=params_node_l3['params']['cutoff'], name='L3', invert=params_node_l3['params']['invert'], initial_value=params_node_l3['params']['initialValue'], bias=params_node_l3['params']['bias'])
    # l5 = lowpass_filter(net, cutoff=cutoff_fastest, name='L5', invert=False, bias=activity_range)

    net.add_connection(synapse_r_l1, 'Retina', 'L1_in')
    net.add_connection(synapse_r_l2, 'Retina', 'L2_in')
    net.add_connection(synapse_r_l3, 'Retina', 'L3')
    # net.add_connection(synapse_r_l, 'L1_out', 'L5')

    net.add_output('L1_out', name='OutL1')
    net.add_output('L2_out', name='OutL2')
    net.add_output('L3', name='OutL3')
    # net.add_output('L5', name='OutL5')

    """
    ####################################################################################################################
    MEDULLA ON
    """
    params_node_mi1 = load_data('params_node_mi1.p')
    params_node_mi9 = load_data('params_node_mi9.p')
    add_lowpass_filter(net, cutoff=params_node_mi1['params']['cutoff'], name='Mi1', invert=params_node_mi1['params']['invert'], bias=params_node_mi1['params']['bias'], initial_value=params_node_mi1['params']['initialValue'])
    add_lowpass_filter(net, cutoff=params_node_mi9['params']['cutoff'], name='Mi9', invert=params_node_mi9['params']['invert'], bias=params_node_mi9['params']['bias'], initial_value=params_node_mi9['params']['initialValue'])

    params_conn_mi1 = load_data('params_conn_mi1.p')
    params_conn_mi9 = load_data('params_conn_mi9.p')
    synapse_l1_mi1 = NonSpikingSynapse(max_conductance=params_conn_mi1['g'], reversal_potential=params_conn_mi1['reversal'], e_lo=0.0, e_hi=activity_range)
    synapse_l3_mi9 = NonSpikingSynapse(max_conductance=params_conn_mi9['g'], reversal_potential=params_conn_mi9['reversal'], e_lo=0.0, e_hi=activity_range)
    net.add_connection(synapse_l1_mi1, 'L1_out', 'Mi1')
    net.add_connection(synapse_l3_mi9, 'L3', 'Mi9')

    net.add_output('Mi1', name='OutMi1')
    net.add_output('Mi9', name='OutMi9')

    """
    ####################################################################################################################
    MEDULLA OFF
    """
    params_conn_tm1 = load_data('params_conn_tm1.p')
    params_conn_tm9 = load_data('params_conn_tm9.p')
    synapse_l2_tm1 = NonSpikingSynapse(max_conductance=params_conn_tm1['g'], reversal_potential=params_conn_tm1['reversal'], e_lo=0.0, e_hi=activity_range)
    synapse_l3_tm9 = NonSpikingSynapse(max_conductance=params_conn_tm9['g'], reversal_potential=params_conn_tm9['reversal'], e_lo=0.0, e_hi=activity_range)

    params_node_tm1 = load_data('params_node_tm1.p')
    params_node_tm9 = load_data('params_node_tm9.p')
    add_lowpass_filter(net, cutoff=params_node_tm1['params']['cutoff'], name='Tm1', invert=params_node_tm1['params']['invert'], bias=params_node_tm1['params']['bias'], initial_value=params_node_tm1['params']['initialValue'])
    add_lowpass_filter(net, cutoff=params_node_tm9['params']['cutoff'], name='Tm9', invert=params_node_tm9['params']['invert'], bias=params_node_tm9['params']['bias'], initial_value=params_node_tm9['params']['initialValue'])
    net.add_connection(synapse_l2_tm1, 'L2_out', 'Tm1')
    net.add_connection(synapse_l3_tm9, 'L3', 'Tm9')

    net.add_output('Tm1', name='OutTm1')
    net.add_output('Tm9', name='OutTm9')

    """
    ####################################################################################################################
    CT1 COMPARTMENTS
    """
    params_conn_ct1_on = load_data('params_conn_ct1_on.p')
    params_conn_ct1_off = load_data('params_conn_ct1_off.p')
    synapse_mi1_ct1on = NonSpikingSynapse(max_conductance=params_conn_ct1_on['g'], reversal_potential=params_conn_ct1_on['reversal'], e_lo=0.0, e_hi=activity_range)
    synapse_tm1_ct1off = NonSpikingSynapse(max_conductance=params_conn_ct1_off['g'], reversal_potential=params_conn_ct1_off['reversal'], e_lo=0.0, e_hi=activity_range)

    params_node_ct1_on = load_data('params_node_ct1_on.p')
    params_node_ct1_off = load_data('params_node_ct1_off.p')
    add_lowpass_filter(net, cutoff=params_node_ct1_on['params']['cutoff'], name='CT1_On', invert=params_node_ct1_on['params']['invert'], bias=params_node_ct1_on['params']['bias'], initial_value=params_node_ct1_on['params']['initialValue'])
    add_lowpass_filter(net, cutoff=params_node_ct1_off['params']['cutoff'], name='CT1_Off', invert=params_node_ct1_off['params']['invert'], bias=params_node_ct1_off['params']['bias'], initial_value=params_node_ct1_off['params']['initialValue'])

    net.add_connection(synapse_mi1_ct1on, 'Mi1', 'CT1_On')
    net.add_connection(synapse_tm1_ct1off, 'Tm1', 'CT1_Off')

    net.add_output('CT1_On', name='OutCT1On')
    net.add_output('CT1_Off', name='OutCT1Off')

    """
    ####################################################################################################################
    EXPORT
    """
    # render(net, view=True)
    model = net.compile(dt, backend='torch', device='cpu')

    return model, net

def gen_test_emd(dt, shape, k_mi1=0.5, k_mi9=0.5, k_ct1on=0.5, k_tm1=0.5, k_tm9=0.5, k_ct1off=0.5, k_t4=0.1, k_t5=0.1,
                 cutoffs=None, output_retina=False, output_mi1=False, output_mi9=False, output_tm1=False,
                 output_tm9=False, output_ct1on=False, output_ct1off=False, output_t4a=False, output_t4b=False,
                 output_t4c=False, output_t4d=False, output_t5a=False, output_t5b=False, output_t5c=False,
                 output_t5d=False):
    """
    ####################################################################################################################
    GATHER PROPERTIES
    """
    if cutoffs is not None:
        tune_neurons(cutoffs, 'all')

    """
    ####################################################################################################################
    INITIALIZE NETWORK
    """
    net = Network('Motion Vision EMD')
    flat_size = int(shape[0]*shape[1])

    """
    ####################################################################################################################
    RETINA
    """

    params_node_retina = load_data('params_node_retina.p')
    add_lowpass_filter(net, params_node_retina['params']['cutoff'], name='Retina',
                       invert=params_node_retina['params']['invert'],
                       initial_value=params_node_retina['params']['initialValue'],
                       bias=params_node_retina['params']['bias'], shape=shape, color='black')

    net.add_input('Retina', size=flat_size)
    if output_retina:
        net.add_output('Retina', name='OutR')

    """
    ####################################################################################################################
    LAMINA
    """
    params_conn_l1 = load_data('params_conn_l1.p')
    params_conn_l2 = load_data('params_conn_l2.p')
    params_conn_l3 = load_data('params_conn_l3.p')
    r_l1_g, r_l1_reversal, e_lo, e_hi = __gen_receptive_fields__(params_conn_l1)
    r_l2_g, r_l2_reversal, _, _ = __gen_receptive_fields__(params_conn_l2)
    r_l3_g, r_l3_reversal, _, _ = __gen_receptive_fields__(params_conn_l3)
    synapse_r_l1 = NonSpikingPatternConnection(max_conductance_kernel=r_l1_g, reversal_potential_kernel=r_l1_reversal, e_lo_kernel=e_lo, e_hi_kernel=e_hi)
    synapse_r_l2 = NonSpikingPatternConnection(max_conductance_kernel=r_l2_g, reversal_potential_kernel=r_l2_reversal, e_lo_kernel=e_lo, e_hi_kernel=e_hi)
    synapse_r_l3 = NonSpikingPatternConnection(max_conductance_kernel=r_l3_g, reversal_potential_kernel=r_l3_reversal, e_lo_kernel=e_lo, e_hi_kernel=e_hi)

    params_node_l1 = load_data('params_node_l1.p')
    add_scaled_bandpass_filter(net, params_node_l1['params']['cutoffLow'], params_node_l1['params']['cutoffHigh'],
                               params_node_l1['params']['gain'], invert=params_node_l1['params']['invert'], name='L1',
                               shape=shape, color='darkgreen')
    params_node_l2 = load_data('params_node_l2.p')
    add_scaled_bandpass_filter(net, params_node_l2['params']['cutoffLow'], params_node_l2['params']['cutoffHigh'],
                               params_node_l2['params']['gain'], invert=params_node_l2['params']['invert'], name='L2',
                               shape=shape, color='green')
    params_node_l3 = load_data('params_node_l3.p')
    add_lowpass_filter(net, cutoff=params_node_l3['params']['cutoff'], name='L3',
                       invert=params_node_l3['params']['invert'],
                       initial_value=params_node_l3['params']['initialValue'], bias=params_node_l3['params']['bias'],
                       shape=shape, color='lightgreen')

    net.add_connection(synapse_r_l1, 'Retina', 'L1_in')
    net.add_connection(synapse_r_l2, 'Retina', 'L2_in')
    net.add_connection(synapse_r_l3, 'Retina', 'L3')

    # net.add_output('L1_out', name='OutL1')
    # net.add_output('L2_out', name='OutL2')
    # net.add_output('L3', name='OutL3')

    """
    ####################################################################################################################
    MEDULLA ON
    """
    params_node_mi1 = load_data('params_node_mi1.p')
    params_node_mi9 = load_data('params_node_mi9.p')
    add_lowpass_filter(net, cutoff=params_node_mi1['params']['cutoff'], name='Mi1',
                       invert=params_node_mi1['params']['invert'], bias=params_node_mi1['params']['bias'],
                       initial_value=params_node_mi1['params']['initialValue'], shape=shape, color='red')
    add_lowpass_filter(net, cutoff=params_node_mi9['params']['cutoff'], name='Mi9',
                       invert=params_node_mi9['params']['invert'], bias=params_node_mi9['params']['bias'],
                       initial_value=params_node_mi9['params']['initialValue'], shape=shape, color='indianred')

    params_conn_mi1 = load_data('params_conn_mi1.p')
    params_conn_mi9 = load_data('params_conn_mi9.p')
    synapse_l1_mi1 = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_conn_mi1['g'],
                                                  reversal_potential=params_conn_mi1['reversal'], e_lo=0.0,
                                                  e_hi=activity_range)
    synapse_l3_mi9 = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_conn_mi9['g'],
                                                  reversal_potential=params_conn_mi9['reversal'], e_lo=0.0,
                                                  e_hi=activity_range)
    net.add_connection(synapse_l1_mi1, 'L1_out', 'Mi1')
    net.add_connection(synapse_l3_mi9, 'L3', 'Mi9')

    if output_mi1:
        net.add_output('Mi1', name='OutMi1')
    if output_mi9:
        net.add_output('Mi9', name='OutMi9')

    """
    ####################################################################################################################
    MEDULLA OFF
    """
    params_conn_tm1 = load_data('params_conn_tm1.p')
    params_conn_tm9 = load_data('params_conn_tm9.p')
    synapse_l2_tm1 = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_conn_tm1['g'],
                                                  reversal_potential=params_conn_tm1['reversal'], e_lo=0.0,
                                                  e_hi=activity_range)
    synapse_l3_tm9 = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_conn_tm9['g'],
                                                  reversal_potential=params_conn_tm9['reversal'], e_lo=0.0,
                                                  e_hi=activity_range)

    params_node_tm1 = load_data('params_node_tm1.p')
    params_node_tm9 = load_data('params_node_tm9.p')
    add_lowpass_filter(net, cutoff=params_node_tm1['params']['cutoff'], name='Tm1',
                       invert=params_node_tm1['params']['invert'], bias=params_node_tm1['params']['bias'],
                       initial_value=params_node_tm1['params']['initialValue'], shape=shape, color='blue')
    add_lowpass_filter(net, cutoff=params_node_tm9['params']['cutoff'], name='Tm9',
                       invert=params_node_tm9['params']['invert'], bias=params_node_tm9['params']['bias'],
                       initial_value=params_node_tm9['params']['initialValue'], shape=shape, color='lightblue')
    net.add_connection(synapse_l2_tm1, 'L2_out', 'Tm1')
    net.add_connection(synapse_l3_tm9, 'L3', 'Tm9')

    if output_tm1:
        net.add_output('Tm1', name='OutTm1')
    if output_tm9:
        net.add_output('Tm9', name='OutTm9')

    """
    ####################################################################################################################
    CT1 COMPARTMENTS
    """
    params_conn_ct1_on = load_data('params_conn_ct1_on.p')
    params_conn_ct1_off = load_data('params_conn_ct1_off.p')
    synapse_mi1_ct1on = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_conn_ct1_on['g'],
                                                     reversal_potential=params_conn_ct1_on['reversal'], e_lo=0.0,
                                                     e_hi=activity_range)
    synapse_tm1_ct1off = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_conn_ct1_off['g'],
                                                      reversal_potential=params_conn_ct1_off['reversal'], e_lo=0.0,
                                                      e_hi=activity_range)

    params_node_ct1_on = load_data('params_node_ct1_on.p')
    params_node_ct1_off = load_data('params_node_ct1_off.p')
    add_lowpass_filter(net, cutoff=params_node_ct1_on['params']['cutoff'], name='CT1_On',
                       invert=params_node_ct1_on['params']['invert'], bias=params_node_ct1_on['params']['bias'],
                       initial_value=params_node_ct1_on['params']['initialValue'], shape=shape, color='gold')
    add_lowpass_filter(net, cutoff=params_node_ct1_off['params']['cutoff'], name='CT1_Off',
                       invert=params_node_ct1_off['params']['invert'], bias=params_node_ct1_off['params']['bias'],
                       initial_value=params_node_ct1_off['params']['initialValue'], shape=shape, color='goldenrod')

    net.add_connection(synapse_mi1_ct1on, 'Mi1', 'CT1_On')
    net.add_connection(synapse_tm1_ct1off, 'Tm1', 'CT1_Off')

    if output_ct1on:
        net.add_output('CT1_On', name='OutCT1On')
    if output_ct1off:
        net.add_output('CT1_Off', name='OutCT1Off')

    """
    ####################################################################################################################
    T4 CELLS
    """
    cond_mi1, rev_mi1 = __transmission_params__(k_mi1, excite=True)
    cond_mi9, rev_mi9 = __transmission_params__(k_mi9, excite=False)
    cond_ct1_on, rev_ct1_on = __transmission_params__(k_ct1on, excite=False)
    cond_t4_self, rev_t4_self = __transmission_params__(k_t4, excite=True)

    cond_mi9_kernel_b = np.array([[0, 0, 0],
                                [cond_mi9, 0, 0],
                                [0, 0, 0]])
    rev_mi9_kernel_b = np.array([[0, 0, 0],
                               [rev_mi9, 0, 0],
                               [0, 0, 0]])
    cond_ct1_on_kernel_b = np.array([[0, 0, 0],
                                [0, 0, cond_ct1_on],
                                [0, 0, 0]])
    rev_ct1_on_kernel_b = np.array([[0, 0, 0],
                               [0, 0, rev_ct1_on],
                               [0, 0, 0]])
    cond_mi9_kernel_a, cond_mi9_kernel_c, cond_mi9_kernel_d = __all_quadrants__(cond_mi9_kernel_b)
    cond_ct1_on_kernel_a, cond_ct1_on_kernel_c, cond_ct1_on_kernel_d = __all_quadrants__(cond_ct1_on_kernel_b)
    rev_mi9_kernel_a, rev_mi9_kernel_c, rev_mi9_kernel_d = __all_quadrants__(rev_mi9_kernel_b)
    rev_ct1_on_kernel_a, rev_ct1_on_kernel_c, rev_ct1_on_kernel_d = __all_quadrants__(rev_ct1_on_kernel_b)
    e_lo_kernel = np.zeros([3,3])
    e_hi_kernel = np.zeros([3,3]) + activity_range
    cond_t4_self_kernel = np.array([[cond_t4_self, cond_t4_self, cond_t4_self],
                                    [cond_t4_self, 0.0         , cond_t4_self],
                                    [cond_t4_self, cond_t4_self, cond_t4_self]])
    rev_t4_self_kernel = np.array([[rev_t4_self, rev_t4_self, rev_t4_self],
                                   [rev_t4_self, 0.0        , rev_t4_self],
                                   [rev_t4_self, rev_t4_self, rev_t4_self]])

    synapse_mi1_t4 = NonSpikingOneToOneConnection(shape=shape, max_conductance=cond_mi1, reversal_potential=rev_mi1,
                                                  e_lo=0.0, e_hi=activity_range)

    synapse_mi9_t4_a = NonSpikingPatternConnection(max_conductance_kernel=cond_mi9_kernel_a,
                                                   reversal_potential_kernel=rev_mi9_kernel_a, e_lo_kernel=e_lo_kernel,
                                                   e_hi_kernel=e_hi_kernel)
    synapse_ct1on_t4_a = NonSpikingPatternConnection(max_conductance_kernel=cond_ct1_on_kernel_a,
                                                     reversal_potential_kernel=rev_ct1_on_kernel_a,
                                                     e_lo_kernel=e_lo_kernel,
                                                     e_hi_kernel=e_hi_kernel)
    synapse_mi9_t4_b = NonSpikingPatternConnection(max_conductance_kernel=cond_mi9_kernel_b,
                                                    reversal_potential_kernel=rev_mi9_kernel_b, e_lo_kernel=e_lo_kernel,
                                                    e_hi_kernel=e_hi_kernel)
    synapse_ct1on_t4_b = NonSpikingPatternConnection(max_conductance_kernel=cond_ct1_on_kernel_b,
                                                      reversal_potential_kernel=rev_ct1_on_kernel_b, e_lo_kernel=e_lo_kernel,
                                                      e_hi_kernel=e_hi_kernel)
    synapse_mi9_t4_c = NonSpikingPatternConnection(max_conductance_kernel=cond_mi9_kernel_c,
                                                   reversal_potential_kernel=rev_mi9_kernel_c, e_lo_kernel=e_lo_kernel,
                                                   e_hi_kernel=e_hi_kernel)
    synapse_ct1on_t4_c = NonSpikingPatternConnection(max_conductance_kernel=cond_ct1_on_kernel_c,
                                                     reversal_potential_kernel=rev_ct1_on_kernel_c,
                                                     e_lo_kernel=e_lo_kernel,
                                                     e_hi_kernel=e_hi_kernel)
    synapse_mi9_t4_d = NonSpikingPatternConnection(max_conductance_kernel=cond_mi9_kernel_d,
                                                   reversal_potential_kernel=rev_mi9_kernel_d, e_lo_kernel=e_lo_kernel,
                                                   e_hi_kernel=e_hi_kernel)
    synapse_ct1on_t4_d = NonSpikingPatternConnection(max_conductance_kernel=cond_ct1_on_kernel_d,
                                                     reversal_potential_kernel=rev_ct1_on_kernel_d,
                                                     e_lo_kernel=e_lo_kernel,
                                                     e_hi_kernel=e_hi_kernel)
    synapse_t4_self = NonSpikingPatternConnection(max_conductance_kernel=cond_t4_self_kernel,
                                                  reversal_potential_kernel=rev_t4_self_kernel,
                                                  e_lo_kernel=e_lo_kernel,
                                                  e_hi_kernel=e_hi_kernel)

    add_lowpass_filter(net, cutoff_fastest, name='T4_a',
                       invert=False,
                       initial_value=0.0,
                       bias=activity_range, shape=shape, color='purple')
    add_lowpass_filter(net, cutoff_fastest, name='T4_b',
                       invert=False,
                       initial_value=0.0,
                       bias=activity_range, shape=shape, color='purple')
    add_lowpass_filter(net, cutoff_fastest, name='T4_c',
                       invert=False,
                       initial_value=0.0,
                       bias=activity_range, shape=shape, color='purple')
    add_lowpass_filter(net, cutoff_fastest, name='T4_d',
                       invert=False,
                       initial_value=0.0,
                       bias=activity_range, shape=shape, color='purple')

    net.add_connection(synapse_mi1_t4, 'Mi1', 'T4_a')
    net.add_connection(synapse_mi9_t4_a, 'Mi9', 'T4_a')
    net.add_connection(synapse_ct1on_t4_a, 'CT1_On', 'T4_a')
    net.add_connection(synapse_mi1_t4, 'Mi1', 'T4_b')
    net.add_connection(synapse_mi9_t4_b, 'Mi9', 'T4_b')
    net.add_connection(synapse_ct1on_t4_b, 'CT1_On', 'T4_b')
    net.add_connection(synapse_mi1_t4, 'Mi1', 'T4_c')
    net.add_connection(synapse_mi9_t4_c, 'Mi9', 'T4_c')
    net.add_connection(synapse_ct1on_t4_c, 'CT1_On', 'T4_c')
    net.add_connection(synapse_mi1_t4, 'Mi1', 'T4_d')
    net.add_connection(synapse_mi9_t4_d, 'Mi9', 'T4_d')
    net.add_connection(synapse_ct1on_t4_d, 'CT1_On', 'T4_d')
    net.add_connection(synapse_t4_self, 'T4_a', 'T4_a')
    net.add_connection(synapse_t4_self, 'T4_b', 'T4_b')
    net.add_connection(synapse_t4_self, 'T4_c', 'T4_c')
    net.add_connection(synapse_t4_self, 'T4_d', 'T4_d')

    if output_t4a:
        net.add_output('T4_a')
    if output_t4b:
        net.add_output('T4_b')
    if output_t4c:
        net.add_output('T4_c')
    if output_t4d:
        net.add_output('T4_d')

    """
    ####################################################################################################################
    T5 CELLS
    """
    cond_tm9, rev_tm9 = __transmission_params__(k_tm9, excite=True)
    cond_tm1, rev_tm1 = __transmission_params__(k_tm1, excite=True)
    cond_ct1_off, rev_ct1_off = __transmission_params__(k_ct1off, excite=False)
    cond_t5_self, rev_t5_self = __transmission_params__(k_t5, excite=True)

    cond_tm9_kernel_b = np.array([[0, 0, 0],
                                  [cond_tm9, 0, 0],
                                  [0, 0, 0]])
    rev_tm9_kernel_b = np.array([[0, 0, 0],
                                 [rev_tm9, 0, 0],
                                 [0, 0, 0]])
    cond_ct1_off_kernel_b = np.array([[0, 0, 0],
                                      [0, 0, cond_ct1_off],
                                      [0, 0, 0]])
    rev_ct1_off_kernel_b = np.array([[0, 0, 0],
                                     [0, 0, rev_ct1_off],
                                     [0, 0, 0]])
    cond_tm9_kernel_a, cond_tm9_kernel_c, cond_tm9_kernel_d = __all_quadrants__(cond_tm9_kernel_b)
    cond_ct1_off_kernel_a, cond_ct1_off_kernel_c, cond_ct1_off_kernel_d = __all_quadrants__(cond_ct1_off_kernel_b)
    rev_tm9_kernel_a, rev_tm9_kernel_c, rev_tm9_kernel_d = __all_quadrants__(rev_tm9_kernel_b)
    rev_ct1_off_kernel_a, rev_ct1_off_kernel_c, rev_ct1_off_kernel_d = __all_quadrants__(rev_ct1_off_kernel_b)
    e_lo_kernel = np.zeros([3, 3])
    e_hi_kernel = np.zeros([3, 3]) + activity_range
    cond_t5_self_kernel = np.array([[cond_t5_self, cond_t5_self, cond_t5_self],
                                    [cond_t5_self, 0.0         , cond_t5_self],
                                    [cond_t5_self, cond_t5_self, cond_t5_self]])
    rev_t5_self_kernel = np.array([[rev_t5_self, rev_t5_self, rev_t5_self],
                                   [rev_t5_self, 0.0        , rev_t5_self],
                                   [rev_t5_self, rev_t5_self, rev_t5_self]])

    synapse_tm1_t5 = NonSpikingOneToOneConnection(shape=shape, max_conductance=cond_tm1, reversal_potential=rev_tm1,
                                                  e_lo=0.0, e_hi=activity_range)

    synapse_tm9_t5_a = NonSpikingPatternConnection(max_conductance_kernel=cond_tm9_kernel_a,
                                                   reversal_potential_kernel=rev_tm9_kernel_a, e_lo_kernel=e_lo_kernel,
                                                   e_hi_kernel=e_hi_kernel)
    synapse_ct1off_t5_a = NonSpikingPatternConnection(max_conductance_kernel=cond_ct1_off_kernel_a,
                                                     reversal_potential_kernel=rev_ct1_off_kernel_a,
                                                     e_lo_kernel=e_lo_kernel,
                                                     e_hi_kernel=e_hi_kernel)
    synapse_tm9_t5_b = NonSpikingPatternConnection(max_conductance_kernel=cond_tm9_kernel_b,
                                                   reversal_potential_kernel=rev_tm9_kernel_b, e_lo_kernel=e_lo_kernel,
                                                   e_hi_kernel=e_hi_kernel)
    synapse_ct1off_t5_b = NonSpikingPatternConnection(max_conductance_kernel=cond_ct1_off_kernel_b,
                                                     reversal_potential_kernel=rev_ct1_off_kernel_b,
                                                     e_lo_kernel=e_lo_kernel,
                                                     e_hi_kernel=e_hi_kernel)
    synapse_tm9_t5_c = NonSpikingPatternConnection(max_conductance_kernel=cond_tm9_kernel_c,
                                                   reversal_potential_kernel=rev_tm9_kernel_c, e_lo_kernel=e_lo_kernel,
                                                   e_hi_kernel=e_hi_kernel)
    synapse_ct1off_t5_c = NonSpikingPatternConnection(max_conductance_kernel=cond_ct1_off_kernel_c,
                                                     reversal_potential_kernel=rev_ct1_off_kernel_c,
                                                     e_lo_kernel=e_lo_kernel,
                                                     e_hi_kernel=e_hi_kernel)
    synapse_tm9_t5_d = NonSpikingPatternConnection(max_conductance_kernel=cond_tm9_kernel_d,
                                                   reversal_potential_kernel=rev_tm9_kernel_d, e_lo_kernel=e_lo_kernel,
                                                   e_hi_kernel=e_hi_kernel)
    synapse_ct1off_t5_d = NonSpikingPatternConnection(max_conductance_kernel=cond_ct1_off_kernel_d,
                                                     reversal_potential_kernel=rev_ct1_off_kernel_d,
                                                     e_lo_kernel=e_lo_kernel,
                                                     e_hi_kernel=e_hi_kernel)
    synapse_t5_self = NonSpikingPatternConnection(max_conductance_kernel=cond_t5_self_kernel,
                                                  reversal_potential_kernel=rev_t5_self_kernel,
                                                  e_lo_kernel=e_lo_kernel,
                                                  e_hi_kernel=e_hi_kernel)

    add_lowpass_filter(net, cutoff_fastest, name='T5_a',
                       invert=False,
                       initial_value=0.0,
                       bias=0.0, shape=shape, color='plum')
    add_lowpass_filter(net, cutoff_fastest, name='T5_b',
                       invert=False,
                       initial_value=0.0,
                       bias=0.0, shape=shape, color='plum')
    add_lowpass_filter(net, cutoff_fastest, name='T5_c',
                       invert=False,
                       initial_value=0.0,
                       bias=0.0, shape=shape, color='plum')
    add_lowpass_filter(net, cutoff_fastest, name='T5_d',
                       invert=False,
                       initial_value=0.0,
                       bias=0.0, shape=shape, color='plum')

    net.add_connection(synapse_tm1_t5, 'Tm1', 'T5_a')
    net.add_connection(synapse_tm9_t5_a, 'Tm9', 'T5_a')
    net.add_connection(synapse_ct1off_t5_a, 'CT1_Off', 'T5_a')
    net.add_connection(synapse_tm1_t5, 'Tm1', 'T5_b')
    net.add_connection(synapse_tm9_t5_b, 'Tm9', 'T5_b')
    net.add_connection(synapse_ct1off_t5_b, 'CT1_Off', 'T5_b')
    net.add_connection(synapse_tm1_t5, 'Tm1', 'T5_c')
    net.add_connection(synapse_tm9_t5_c, 'Tm9', 'T5_c')
    net.add_connection(synapse_ct1off_t5_c, 'CT1_Off', 'T5_c')
    net.add_connection(synapse_tm1_t5, 'Tm1', 'T5_d')
    net.add_connection(synapse_tm9_t5_d, 'Tm9', 'T5_d')
    net.add_connection(synapse_ct1off_t5_d, 'CT1_Off', 'T5_d')
    net.add_connection(synapse_t5_self, 'T5_a', 'T5_a')
    net.add_connection(synapse_t5_self, 'T5_b', 'T5_b')
    net.add_connection(synapse_t5_self, 'T5_c', 'T5_c')
    net.add_connection(synapse_t5_self, 'T5_d', 'T5_d')

    if output_t5a:
        net.add_output('T5_a')
    if output_t5b:
        net.add_output('T5_b')
    if output_t5c:
        net.add_output('T5_c')
    if output_t5d:
        net.add_output('T5_d')

    """
    ####################################################################################################################
    EXPORT
    """
    # render(net, view=True)
    model = net.compile(dt, backend=backend, device=device)

    return model, net

def gen_emd_on_mcmc(params, dt, device):
    """
    ####################################################################################################################
    GATHER PROPERTIES
    """
    cutoffs = params[:7]
    data = tune_neurons(cutoffs, 'on', debug=False)
    cutoff_t4 = params[7]
    gains = params[8:]
    shape = (7,7)
    """
    ####################################################################################################################
    INITIALIZE NETWORK
    """
    net = Network('Motion Vision EMD')
    flat_size = int(shape[0]*shape[1])

    """
    ####################################################################################################################
    RETINA
    """

    params_node_retina = data['Retina']
    add_lowpass_filter(net, params_node_retina['params']['cutoff'], name='Retina',
                       invert=params_node_retina['params']['invert'],
                       initial_value=params_node_retina['params']['initialValue'],
                       bias=params_node_retina['params']['bias'], shape=shape, color='black')

    net.add_input('Retina', size=flat_size)

    """
    ####################################################################################################################
    LAMINA
    """
    params_conn_l1 = data['L1Conn']
    params_conn_l3 = data['L3Conn']
    r_l1_g, r_l1_reversal, e_lo, e_hi = __gen_receptive_fields__(params_conn_l1)
    r_l3_g, r_l3_reversal, _, _ = __gen_receptive_fields__(params_conn_l3)
    synapse_r_l1 = NonSpikingPatternConnection(max_conductance_kernel=r_l1_g, reversal_potential_kernel=r_l1_reversal, e_lo_kernel=e_lo, e_hi_kernel=e_hi)
    synapse_r_l3 = NonSpikingPatternConnection(max_conductance_kernel=r_l3_g, reversal_potential_kernel=r_l3_reversal, e_lo_kernel=e_lo, e_hi_kernel=e_hi)

    params_node_l1 = data['L1']
    add_scaled_bandpass_filter(net, params_node_l1['params']['cutoffLow'], params_node_l1['params']['cutoffHigh'],
                               params_node_l1['params']['gain'], invert=params_node_l1['params']['invert'], name='L1',
                               shape=shape, color='darkgreen')
    params_node_l3 = data['L3']
    add_lowpass_filter(net, cutoff=params_node_l3['params']['cutoff'], name='L3',
                       invert=params_node_l3['params']['invert'],
                       initial_value=params_node_l3['params']['initialValue'], bias=params_node_l3['params']['bias'],
                       shape=shape, color='lightgreen')

    net.add_connection(synapse_r_l1, 'Retina', 'L1_in')
    net.add_connection(synapse_r_l3, 'Retina', 'L3')

    """
    ####################################################################################################################
    MEDULLA ON
    """
    params_node_mi1 = data['Mi1']
    params_node_mi9 = data['Mi9']
    add_lowpass_filter(net, cutoff=params_node_mi1['params']['cutoff'], name='Mi1',
                       invert=params_node_mi1['params']['invert'], bias=params_node_mi1['params']['bias'],
                       initial_value=params_node_mi1['params']['initialValue'], shape=shape, color='red')
    add_lowpass_filter(net, cutoff=params_node_mi9['params']['cutoff'], name='Mi9',
                       invert=params_node_mi9['params']['invert'], bias=params_node_mi9['params']['bias'],
                       initial_value=params_node_mi9['params']['initialValue'], shape=shape, color='indianred')

    params_conn_mi1 = data['Mi1Conn']
    params_conn_mi9 = data['Mi9Conn']
    synapse_l1_mi1 = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_conn_mi1['g'],
                                                  reversal_potential=params_conn_mi1['reversal'], e_lo=0.0,
                                                  e_hi=activity_range)
    synapse_l3_mi9 = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_conn_mi9['g'],
                                                  reversal_potential=params_conn_mi9['reversal'], e_lo=0.0,
                                                  e_hi=activity_range)
    net.add_connection(synapse_l1_mi1, 'L1_out', 'Mi1')
    net.add_connection(synapse_l3_mi9, 'L3', 'Mi9')

    """
    ####################################################################################################################
    CT1 COMPARTMENTS
    """
    params_conn_ct1_on = data['CT1OnConn']
    synapse_mi1_ct1on = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_conn_ct1_on['g'],
                                                     reversal_potential=params_conn_ct1_on['reversal'], e_lo=0.0,
                                                     e_hi=activity_range)

    params_node_ct1_on = data['CT1On']
    add_lowpass_filter(net, cutoff=params_node_ct1_on['params']['cutoff'], name='CT1_On',
                       invert=params_node_ct1_on['params']['invert'], bias=params_node_ct1_on['params']['bias'],
                       initial_value=params_node_ct1_on['params']['initialValue'], shape=shape, color='gold')

    net.add_connection(synapse_mi1_ct1on, 'Mi1', 'CT1_On')

    """
    ####################################################################################################################
    T4 CELLS
    """
    cond_mi1, rev_mi1 = __transmission_params__(gains[0], excite=True)
    cond_mi9, rev_mi9 = __transmission_params__(gains[1], excite=False)
    cond_ct1_on, rev_ct1_on = __transmission_params__(gains[2], excite=False)
    cond_t4_self, rev_t4_self = __transmission_params__(gains[3], excite=True)

    cond_mi9_kernel_b = np.array([[0, 0, 0],
                                [cond_mi9, 0, 0],
                                [0, 0, 0]])
    rev_mi9_kernel_b = np.array([[0, 0, 0],
                               [rev_mi9, 0, 0],
                               [0, 0, 0]])
    cond_ct1_on_kernel_b = np.array([[0, 0, 0],
                                [0, 0, cond_ct1_on],
                                [0, 0, 0]])
    rev_ct1_on_kernel_b = np.array([[0, 0, 0],
                               [0, 0, rev_ct1_on],
                               [0, 0, 0]])
    cond_mi9_kernel_a, _, _ = __all_quadrants__(cond_mi9_kernel_b)
    cond_ct1_on_kernel_a, _, _ = __all_quadrants__(cond_ct1_on_kernel_b)
    rev_mi9_kernel_a, _, _ = __all_quadrants__(rev_mi9_kernel_b)
    rev_ct1_on_kernel_a, _, _ = __all_quadrants__(rev_ct1_on_kernel_b)
    e_lo_kernel = np.zeros([3,3])
    e_hi_kernel = np.zeros([3,3]) + activity_range
    cond_t4_self_kernel = np.array([[cond_t4_self, cond_t4_self, cond_t4_self],
                                    [cond_t4_self, 0.0         , cond_t4_self],
                                    [cond_t4_self, cond_t4_self, cond_t4_self]])
    rev_t4_self_kernel = np.array([[rev_t4_self, rev_t4_self, rev_t4_self],
                                   [rev_t4_self, 0.0        , rev_t4_self],
                                   [rev_t4_self, rev_t4_self, rev_t4_self]])

    synapse_mi1_t4 = NonSpikingOneToOneConnection(shape=shape, max_conductance=cond_mi1, reversal_potential=rev_mi1,
                                                  e_lo=0.0, e_hi=activity_range)

    synapse_mi9_t4_a = NonSpikingPatternConnection(max_conductance_kernel=cond_mi9_kernel_a,
                                                   reversal_potential_kernel=rev_mi9_kernel_a, e_lo_kernel=e_lo_kernel,
                                                   e_hi_kernel=e_hi_kernel)
    synapse_ct1on_t4_a = NonSpikingPatternConnection(max_conductance_kernel=cond_ct1_on_kernel_a,
                                                     reversal_potential_kernel=rev_ct1_on_kernel_a,
                                                     e_lo_kernel=e_lo_kernel,
                                                     e_hi_kernel=e_hi_kernel)
    synapse_mi9_t4_b = NonSpikingPatternConnection(max_conductance_kernel=cond_mi9_kernel_b,
                                                    reversal_potential_kernel=rev_mi9_kernel_b, e_lo_kernel=e_lo_kernel,
                                                    e_hi_kernel=e_hi_kernel)
    synapse_ct1on_t4_b = NonSpikingPatternConnection(max_conductance_kernel=cond_ct1_on_kernel_b,
                                                      reversal_potential_kernel=rev_ct1_on_kernel_b, e_lo_kernel=e_lo_kernel,
                                                      e_hi_kernel=e_hi_kernel)
    synapse_t4_self = NonSpikingPatternConnection(max_conductance_kernel=cond_t4_self_kernel,
                                                  reversal_potential_kernel=rev_t4_self_kernel,
                                                  e_lo_kernel=e_lo_kernel,
                                                  e_hi_kernel=e_hi_kernel)

    add_lowpass_filter(net, cutoff_t4, name='T4_a',
                       invert=False,
                       initial_value=0.0,
                       bias=activity_range, shape=shape, color='purple')
    add_lowpass_filter(net, cutoff_t4, name='T4_b',
                       invert=False,
                       initial_value=0.0,
                       bias=activity_range, shape=shape, color='purple')

    net.add_connection(synapse_mi1_t4, 'Mi1', 'T4_a')
    net.add_connection(synapse_mi9_t4_a, 'Mi9', 'T4_a')
    net.add_connection(synapse_ct1on_t4_a, 'CT1_On', 'T4_a')
    net.add_connection(synapse_mi1_t4, 'Mi1', 'T4_b')
    net.add_connection(synapse_mi9_t4_b, 'Mi9', 'T4_b')
    net.add_connection(synapse_ct1on_t4_b, 'CT1_On', 'T4_b')
    net.add_connection(synapse_t4_self, 'T4_a', 'T4_a')
    net.add_connection(synapse_t4_self, 'T4_b', 'T4_b')

    net.add_output('T4_a')
    net.add_output('T4_b')

    """
    ####################################################################################################################
    EXPORT
    """
    # render(net, view=True)
    model = net.compile(dt, backend='numpy', device=device)

    return model, net
