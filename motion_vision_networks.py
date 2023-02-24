import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

from sns_toolbox.connections import NonSpikingSynapse, NonSpikingPatternConnection
from sns_toolbox.networks import Network
from sns_toolbox.renderer import render
from utilities import add_lowpass_filter, activity_range, dt, load_data, add_scaled_bandpass_filter, backend, NonSpikingOneToOneConnection, synapse_target
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

def gen_single_column(cutoffs=None):
    """
    ####################################################################################################################
    NETWORK
    """
    if cutoffs is not None:
        tune_neurons(cutoffs)
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

def gen_test_emd(shape, cutoffs=None):
    """
    ####################################################################################################################
    GATHER PROPERTIES
    """
    if cutoffs is not None:
        tune_neurons(cutoffs)

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

    # net.add_output('Mi1', name='OutMi1')
    # net.add_output('Mi9', name='OutMi9')

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

    # net.add_output('Tm1', name='OutTm1')
    # net.add_output('Tm9', name='OutTm9')

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

    # net.add_output('CT1_On', name='OutCT1On')
    # net.add_output('CT1_Off', name='OutCT1Off')

    """
    ####################################################################################################################
    T4 CELLS
    """
    cond_mi1, rev_mi1 = synapse_target(activity_range, 0.0)
    cond_mi9, rev_mi9 = synapse_target(0.0, activity_range)
    cond_ct1, rev_ct1 = synapse_target(0.0, activity_range)

    cond_mi9_kernel = np.array([[0, 0, 0],
                                [cond_mi9, 0, 0],
                                [0, 0, 0]])
    rev_mi9_kernel = np.array([[0, 0, 0],
                               [rev_mi9, 0, 0],
                               [0, 0, 0]])
    cond_ct1_kernel = np.array([[0, 0, 0],
                                [0, 0, cond_ct1],
                                [0, 0, 0]])
    rev_ct1_kernel = np.array([[0, 0, 0],
                               [0, 0, rev_ct1],
                               [0, 0, 0]])
    e_lo_kernel = np.zeros([3,3])
    e_hi_kernel = np.zeros([3,3]) + activity_range

    synapse_mi1_t4 = NonSpikingOneToOneConnection(shape=shape, max_conductance=cond_mi1, reversal_potential=rev_mi1,
                                                  e_lo=0.0, e_hi=activity_range)
    synapse_mi9_t4_bf = NonSpikingPatternConnection(max_conductance_kernel=cond_mi9_kernel,
                                                    reversal_potential_kernel=rev_mi9_kernel, e_lo_kernel=e_lo_kernel,
                                                    e_hi_kernel=e_hi_kernel)
    synapse_ct1on_t4_bf = NonSpikingPatternConnection(max_conductance_kernel=cond_ct1_kernel,
                                                      reversal_potential_kernel=rev_ct1_kernel, e_lo_kernel=e_lo_kernel,
                                                      e_hi_kernel=e_hi_kernel)
    add_lowpass_filter(net, params_node_retina['params']['cutoff'], name='T4_bf',
                       invert=params_node_retina['params']['invert'],
                       initial_value=params_node_retina['params']['initialValue'],
                       bias=params_node_retina['params']['bias'], shape=shape, color='purple')

    net.add_connection(synapse_mi1_t4, 'Mi1', 'T4_bf')
    net.add_connection(synapse_mi9_t4_bf, 'Mi9', 'T4_bf')
    net.add_connection(synapse_ct1on_t4_bf, 'CT1_On', 'T4_bf')

    net.add_output('T4_bf')

    """
    ####################################################################################################################
    EXPORT
    """
    # render(net, view=True)
    model = net.compile(dt, backend=backend)

    return model, net
