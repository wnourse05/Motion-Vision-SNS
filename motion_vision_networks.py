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

def gen_single_emd_on(dt, params):
    """
    ####################################################################################################################
    GATHER PROPERTIES
    """
    cutoffs = params[:-1]
    data = tune_neurons(cutoffs, 'on')
    c = params[-1]

    """
    ####################################################################################################################
    INITIALIZE NETWORK
    """
    net = Network('Motion Vision On B EMD')
    suffixes = ['L', 'C', 'R']
    num_cols = len(suffixes)
    """
    ####################################################################################################################
    RETINA
    """

    params_node_retina = data['Retina']
    for i in range(num_cols):
        add_lowpass_filter(net, params_node_retina['params']['cutoff'], name='R_'+suffixes[i],
                           invert=params_node_retina['params']['invert'],
                           initial_value=params_node_retina['params']['initialValue'],
                           bias=params_node_retina['params']['bias'], color='black')

        net.add_input('R_'+suffixes[i], name='I'+suffixes[i])

        net.add_output('R_'+suffixes[i], name='O'+suffixes[i])

    """
    ####################################################################################################################
    LAMINA
    """
    params_conn_l1 = data['L1Conn']
    params_conn_l3 = data['L3Conn']

    r_l1_g = params_conn_l1['g']['center']
    r_l1_reversal = params_conn_l1['reversal']['center']
    r_l3_g = params_conn_l3['g']['center']
    r_l3_reversal = params_conn_l3['reversal']['center']

    synapse_r_l1 = NonSpikingSynapse(max_conductance=r_l1_g, reversal_potential=r_l1_reversal, e_lo=0.0, e_hi=activity_range)

    synapse_r_l3 = NonSpikingSynapse(max_conductance=r_l3_g, reversal_potential=r_l3_reversal, e_lo=0.0, e_hi=activity_range)

    params_node_l1 = data['L1']
    for i in range(num_cols):
        add_scaled_bandpass_filter(net, params_node_l1['params']['cutoffLow'], params_node_l1['params']['cutoffHigh'],
                                   params_node_l1['params']['gain'], invert=params_node_l1['params']['invert'],
                                   name='L1_'+suffixes[i], color='darkgreen')
        net.add_connection(synapse_r_l1, 'R_'+suffixes[i], 'L1_'+suffixes[i]+'_in')
        net.add_output('L1_'+suffixes[i]+'_out', name='OutL1_'+suffixes[i])

    params_node_l3 = data['L3']
    for i in range(num_cols):
        add_lowpass_filter(net, cutoff=params_node_l3['params']['cutoff'], name='L3_'+suffixes[i],
                           invert=params_node_l3['params']['invert'],
                           initial_value=params_node_l3['params']['initialValue'], bias=params_node_l3['params']['bias'],
                           color='lightgreen')


        net.add_connection(synapse_r_l3, 'R_'+suffixes[i], 'L3_'+suffixes[i])
        net.add_output('L3_'+suffixes[i], name='OutL3_'+suffixes[i])

    """
    ####################################################################################################################
    MEDULLA ON
    """
    params_node_mi1 = data['Mi1']
    params_node_mi9 = data['Mi9']

    params_conn_mi1 = data['Mi1Conn']
    params_conn_mi9 = data['Mi9Conn']
    for i in range(num_cols):
        add_lowpass_filter(net, cutoff=params_node_mi1['params']['cutoff'], name='Mi1_'+suffixes[i],
                           invert=params_node_mi1['params']['invert'], bias=params_node_mi1['params']['bias'],
                           initial_value=params_node_mi1['params']['initialValue'], color='red')
        synapse_l1_mi1 = NonSpikingSynapse(max_conductance=params_conn_mi1['g'],
                                           reversal_potential=params_conn_mi1['reversal'], e_lo=0.0, e_hi=activity_range)
        net.add_connection(synapse_l1_mi1, 'L1_'+suffixes[i]+'_out', 'Mi1_'+suffixes[i])
        net.add_output('Mi1_'+suffixes[i], name='OutMi1_'+suffixes[i])

    for i in range(num_cols):
        add_lowpass_filter(net, cutoff=params_node_mi9['params']['cutoff'], name='Mi9_'+suffixes[i],
                           invert=params_node_mi9['params']['invert'], bias=params_node_mi9['params']['bias'],
                           initial_value=params_node_mi9['params']['initialValue'], color='indianred')

        synapse_l3_mi9 = NonSpikingSynapse(max_conductance=params_conn_mi9['g'],
                                           reversal_potential=params_conn_mi9['reversal'], e_lo=0.0, e_hi=activity_range)

        net.add_connection(synapse_l3_mi9, 'L3_'+suffixes[i], 'Mi9_'+suffixes[i])

        net.add_output('Mi9_'+suffixes[i], name='OutMi9'+suffixes[i])

    """
    ####################################################################################################################
    CT1 COMPARTMENTS
    """
    params_conn_ct1_on = data['CT1OnConn']
    synapse_mi1_ct1on = NonSpikingSynapse(max_conductance=params_conn_ct1_on['g'],
                                          reversal_potential=params_conn_ct1_on['reversal'], e_lo=0.0,
                                          e_hi=activity_range)
    params_node_ct1_on = data['CT1On']
    for i in range(num_cols):
        add_lowpass_filter(net, cutoff=params_node_ct1_on['params']['cutoff'], name='CT1_On_'+suffixes[i],
                           invert=params_node_ct1_on['params']['invert'], bias=params_node_ct1_on['params']['bias'],
                           initial_value=params_node_ct1_on['params']['initialValue'], color='gold')

        net.add_connection(synapse_mi1_ct1on, 'Mi1_'+suffixes[i], 'CT1_On_'+suffixes[i])
        net.add_output('CT1_On_'+suffixes[i], name='OutCT1On_'+suffixes[i])

    """
    ####################################################################################################################
    T4 CELLS
    """
    params_node_t4 = data['T4']
    params_conn_t4 = data['T4Conn']

    # g_mi1_t4, rev_mi1_t4 = synapse_target(activity_range, 0.0)
    g_pd_t4, rev_pd_t4 = synapse_target(0.0, activity_range)
    g_nd_t4 = 1/c-1
    rev_nd_t4 = 0.0
    synapse_mi1_t4 = NonSpikingSynapse(max_conductance=params_conn_t4['g'], reversal_potential=params_conn_t4['reversal'], e_lo=0.0, e_hi=activity_range)
    synapse_pd_t4 = NonSpikingSynapse(max_conductance=g_pd_t4, reversal_potential=rev_pd_t4, e_lo=0.0, e_hi=activity_range)
    synapse_nd_t4 = NonSpikingSynapse(max_conductance=g_nd_t4, reversal_potential=rev_nd_t4, e_lo=0.0, e_hi=activity_range)

    add_lowpass_filter(net, params_node_t4['params']['cutoff'], name='T4_a', invert=False, initial_value=0.0, bias=0.0, color='purple')
    add_lowpass_filter(net, params_node_t4['params']['cutoff'], name='T4_b', invert=False, initial_value=0.0, bias=0.0, color='purple')

    net.add_connection(synapse_mi1_t4, 'Mi1_C', 'T4_a')
    net.add_connection(synapse_mi1_t4, 'Mi1_C', 'T4_b')

    net.add_connection(synapse_pd_t4, 'CT1_On_L', 'T4_a')
    net.add_connection(synapse_pd_t4, 'CT1_On_R', 'T4_b')

    net.add_connection(synapse_nd_t4, 'Mi9_R', 'T4_a')
    net.add_connection(synapse_nd_t4, 'Mi9_L', 'T4_b')

    net.add_output('T4_a')
    net.add_output('T4_b')

    """
    ####################################################################################################################
    EXPORT
    """
    # render(net, view=True)
    model = net.compile(dt, backend='numpy')

    return model, net

def gen_single_emd_off(dt, params):
    """
    ####################################################################################################################
    GATHER PROPERTIES
    """
    cutoffs = params[:-2]
    data = tune_neurons(cutoffs[:-1], 'off')
    # c = params[-1]

    """
    ####################################################################################################################
    INITIALIZE NETWORK
    """
    net = Network('Motion Vision On B EMD')
    suffixes = ['L', 'C', 'R']
    num_cols = len(suffixes)
    """
    ####################################################################################################################
    RETINA
    """

    params_node_retina = data['Retina']
    for i in range(num_cols):
        add_lowpass_filter(net, params_node_retina['params']['cutoff'], name='R_'+suffixes[i],
                           invert=params_node_retina['params']['invert'],
                           initial_value=params_node_retina['params']['initialValue'],
                           bias=params_node_retina['params']['bias'], color='black')

        net.add_input('R_'+suffixes[i], name='I'+suffixes[i])

        net.add_output('R_'+suffixes[i], name='O'+suffixes[i])

    """
    ####################################################################################################################
    LAMINA
    """
    params_conn_l2 = data['L2Conn']
    params_conn_l3 = data['L3Conn']

    r_l2_g = params_conn_l2['g']['center']
    r_l2_reversal = params_conn_l2['reversal']['center']
    r_l3_g = params_conn_l3['g']['center']
    r_l3_reversal = params_conn_l3['reversal']['center']

    synapse_r_l2 = NonSpikingSynapse(max_conductance=r_l2_g, reversal_potential=r_l2_reversal, e_lo=0.0, e_hi=activity_range)

    synapse_r_l3 = NonSpikingSynapse(max_conductance=r_l3_g, reversal_potential=r_l3_reversal, e_lo=0.0, e_hi=activity_range)

    params_node_l2 = data['L2']
    for i in range(num_cols):
        add_scaled_bandpass_filter(net, params_node_l2['params']['cutoffLow'], params_node_l2['params']['cutoffHigh'],
                                   params_node_l2['params']['gain'], invert=params_node_l2['params']['invert'],
                                   name='L2_'+suffixes[i], color='darkgreen')
        net.add_connection(synapse_r_l2, 'R_'+suffixes[i], 'L2_'+suffixes[i]+'_in')
        net.add_output('L2_'+suffixes[i]+'_out', name='OutL1_'+suffixes[i])

    params_node_l3 = data['L3']
    for i in range(num_cols):
        add_lowpass_filter(net, cutoff=params_node_l3['params']['cutoff'], name='L3_'+suffixes[i],
                           invert=params_node_l3['params']['invert'],
                           initial_value=params_node_l3['params']['initialValue'], bias=params_node_l3['params']['bias'],
                           color='lightgreen')


        net.add_connection(synapse_r_l3, 'R_'+suffixes[i], 'L3_'+suffixes[i])
        net.add_output('L3_'+suffixes[i], name='OutL3_'+suffixes[i])

    """
    ####################################################################################################################
    MEDULLA OFF
    """
    params_node_tm1 = data['Tm1']
    params_node_tm9 = data['Tm9']

    params_conn_tm1 = data['Tm1Conn']
    params_conn_tm9 = data['Tm9Conn']
    for i in range(num_cols):
        add_lowpass_filter(net, cutoff=params_node_tm1['params']['cutoff'], name='Tm1_'+suffixes[i],
                           invert=params_node_tm1['params']['invert'], bias=params_node_tm1['params']['bias'],
                           initial_value=params_node_tm1['params']['initialValue'], color='red')
        synapse_l2_tm1 = NonSpikingSynapse(max_conductance=params_conn_tm1['g'],
                                           reversal_potential=params_conn_tm1['reversal'], e_lo=activity_range, e_hi=2*activity_range)
        net.add_connection(synapse_l2_tm1, 'L2_'+suffixes[i]+'_out', 'Tm1_'+suffixes[i])
        net.add_output('Tm1_'+suffixes[i], name='OutTm1_'+suffixes[i])

    for i in range(num_cols):
        add_lowpass_filter(net, cutoff=params_node_tm9['params']['cutoff'], name='Tm9_'+suffixes[i],
                           invert=params_node_tm9['params']['invert'], bias=params_node_tm9['params']['bias'],
                           initial_value=params_node_tm9['params']['initialValue'], color='indianred')

        synapse_l3_mi9 = NonSpikingSynapse(max_conductance=params_conn_tm9['g'],
                                           reversal_potential=params_conn_tm9['reversal'], e_lo=0.0, e_hi=activity_range)

        net.add_connection(synapse_l3_mi9, 'L3_'+suffixes[i], 'Tm9_'+suffixes[i])

        net.add_output('Tm9_'+suffixes[i], name='OutTm9'+suffixes[i])

    """
    ####################################################################################################################
    CT1 COMPARTMENTS
    """
    params_conn_ct1_off = data['CT1OffConn']
    synapse_tm1_ct1off = NonSpikingSynapse(max_conductance=params_conn_ct1_off['g'],
                                           reversal_potential=params_conn_ct1_off['reversal'], e_lo=0.0,
                                           e_hi=activity_range)
    params_node_ct1_off = data['CT1Off']
    for i in range(num_cols):
        add_lowpass_filter(net, cutoff=params_node_ct1_off['params']['cutoff'], name='CT1_Off_'+suffixes[i],
                           invert=params_node_ct1_off['params']['invert'], bias=params_node_ct1_off['params']['bias'],#+0.5,
                           initial_value=params_node_ct1_off['params']['initialValue'], color='gold')

        net.add_connection(synapse_tm1_ct1off, 'Tm1_'+suffixes[i], 'CT1_Off_'+suffixes[i])
        net.add_output('CT1_Off_'+suffixes[i], name='OutCT1Off_'+suffixes[i])

    """
    ####################################################################################################################
    T5 CELLS
    """
    # params_node_t4 = data['T4']
    # params_conn_t4 = data['T4Conn']

    # g_mi1_t4, rev_mi1_t4 = synapse_target(activity_range, 0.0)
    # g_pd_t5, rev_pd_t5 = synapse_target(0.0, activity_range)
    g_nd_t5, rev_nd_t5 = synapse_target(activity_range, 0.0)
    g_cd_t5 = g_nd_t5
    rev_cd_t5 = rev_nd_t5
    g_pd_t5 = params[-2]
    rev_pd_t5 = params[-1]

    synapse_cd_t5 = NonSpikingSynapse(max_conductance=g_cd_t5/2, reversal_potential=rev_cd_t5, e_lo=0.0, e_hi=activity_range)
    synapse_pd_t5 = NonSpikingSynapse(max_conductance=g_pd_t5, reversal_potential=rev_pd_t5, e_lo=0.0, e_hi=activity_range)
    synapse_nd_t5 = NonSpikingSynapse(max_conductance=g_nd_t5/2, reversal_potential=rev_nd_t5, e_lo=0.0, e_hi=activity_range)

    add_lowpass_filter(net, cutoffs[0], name='T5_a', invert=False, initial_value=0.0, bias=0.0, color='purple')
    add_lowpass_filter(net, cutoffs[0], name='T5_b', invert=False, initial_value=0.0, bias=0.0, color='purple')

    net.add_connection(synapse_cd_t5, 'Tm1_C', 'T5_a')
    net.add_connection(synapse_cd_t5, 'Tm1_C', 'T5_b')

    net.add_connection(synapse_pd_t5, 'CT1_Off_L', 'T5_a')
    net.add_connection(synapse_pd_t5, 'CT1_Off_R', 'T5_b')

    net.add_connection(synapse_nd_t5, 'Tm9_R', 'T5_a')
    net.add_connection(synapse_nd_t5, 'Tm9_L', 'T5_b')

    net.add_output('T5_a')
    net.add_output('T5_b')

    """
    ####################################################################################################################
    EXPORT
    """
    # render(net, view=True)
    model = net.compile(dt, backend='numpy')

    return model, net

def gen_motion_vision(params, shape, device):
    """
    ####################################################################################################################
    INITIALIZE NETWORK
    """
    net = Network('Motion Vision Network')
    flat_size = int(shape[0]*shape[1])

    """
    ####################################################################################################################
    RETINA
    """
    params_node_retina = params['Retina']
    add_lowpass_filter(net, params_node_retina['params']['cutoff'], name='Retina',
                       invert=params_node_retina['params']['invert'],
                       initial_value=params_node_retina['params']['initialValue'],
                       bias=params_node_retina['params']['bias'], shape=shape, color='black')

    net.add_input('Retina', size=flat_size)
    net.add_output('Retina')

    """
    ####################################################################################################################
    LAMINA
    """
    params_conn_l1 = params['L1Conn']
    params_conn_l2 = params['L2Conn']
    params_conn_l3 = params['L3Conn']
    r_l1_g, r_l1_reversal, e_lo_l1, e_hi_l1 = __gen_receptive_fields__(params_conn_l1)
    r_l2_g, r_l2_reversal, e_lo_l2, e_hi_l2 = __gen_receptive_fields__(params_conn_l2)
    r_l3_g, r_l3_reversal, e_lo_l3, e_hi_l3 = __gen_receptive_fields__(params_conn_l3)
    synapse_r_l1 = NonSpikingPatternConnection(max_conductance_kernel=r_l1_g, reversal_potential_kernel=r_l1_reversal, e_lo_kernel=e_lo_l1, e_hi_kernel=e_hi_l1)
    synapse_r_l2 = NonSpikingPatternConnection(max_conductance_kernel=r_l1_g, reversal_potential_kernel=r_l1_reversal, e_lo_kernel=e_lo_l1, e_hi_kernel=e_hi_l1)
    synapse_r_l3 = NonSpikingPatternConnection(max_conductance_kernel=r_l3_g, reversal_potential_kernel=r_l3_reversal, e_lo_kernel=e_lo_l3, e_hi_kernel=e_hi_l3)

    params_node_l1 = params['L1']
    add_scaled_bandpass_filter(net, params_node_l1['params']['cutoffLow'], params_node_l1['params']['cutoffHigh'],
                               params_node_l1['params']['gain'], invert=params_node_l1['params']['invert'], name='L1',
                               shape=shape, color='darkgreen')
    params_node_l2 = params['L2']
    add_scaled_bandpass_filter(net, params_node_l2['params']['cutoffLow'], params_node_l2['params']['cutoffHigh'],
                               params_node_l2['params']['gain'], invert=params_node_l2['params']['invert'], name='L2',
                               shape=shape, color='green')
    params_node_l3 = params['L3']
    add_lowpass_filter(net, cutoff=params_node_l3['params']['cutoff'], name='L3',
                       invert=params_node_l3['params']['invert'],
                       initial_value=params_node_l3['params']['initialValue'], bias=params_node_l3['params']['bias'],
                       shape=shape, color='lightgreen')

    net.add_connection(synapse_r_l1, 'Retina', 'L1_in')
    net.add_connection(synapse_r_l2, 'Retina', 'L2_in')
    net.add_connection(synapse_r_l3, 'Retina', 'L3')
    net.add_output('L1_out')
    net.add_output('L2_out')
    net.add_output('L3')

    """
    ####################################################################################################################
    MEDULLA ON
    """
    params_node_mi1 = params['Mi1']
    params_node_mi9 = params['Mi9']
    add_lowpass_filter(net, cutoff=params_node_mi1['params']['cutoff'], name='Mi1',
                       invert=params_node_mi1['params']['invert'], bias=params_node_mi1['params']['bias'],
                       initial_value=params_node_mi1['params']['initialValue'], shape=shape, color='red')
    add_lowpass_filter(net, cutoff=params_node_mi9['params']['cutoff'], name='Mi9',
                       invert=params_node_mi9['params']['invert'], bias=params_node_mi9['params']['bias'],
                       initial_value=params_node_mi9['params']['initialValue'], shape=shape, color='indianred')

    params_conn_mi1 = params['Mi1Conn']
    params_conn_mi9 = params['Mi9Conn']
    synapse_l1_mi1 = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_conn_mi1['g'],
                                                  reversal_potential=params_conn_mi1['reversal'], e_lo=0.0,
                                                  e_hi=activity_range)
    synapse_l3_mi9 = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_conn_mi9['g'],
                                                  reversal_potential=params_conn_mi9['reversal'], e_lo=0.0,
                                                  e_hi=activity_range)
    net.add_connection(synapse_l1_mi1, 'L1_out', 'Mi1')
    net.add_connection(synapse_l3_mi9, 'L3', 'Mi9')

    net.add_output('Mi1')
    net.add_output('Mi9')

    """
    ####################################################################################################################
    MEDULLA OFF
    """
    params_node_tm1 = params['Tm1']
    params_node_tm9 = params['Tm9']
    add_lowpass_filter(net, cutoff=params_node_tm1['params']['cutoff'], name='Tm1',
                       invert=params_node_tm1['params']['invert'], bias=params_node_tm1['params']['bias'],
                       initial_value=params_node_tm1['params']['initialValue'], shape=shape, color='navy')
    add_lowpass_filter(net, cutoff=params_node_tm9['params']['cutoff'], name='Tm9',
                       invert=params_node_tm9['params']['invert'], bias=params_node_tm9['params']['bias'],
                       initial_value=params_node_tm9['params']['initialValue'], shape=shape, color='royalblue')

    params_conn_tm1 = params['Tm1Conn']
    params_conn_tm9 = params['Tm9Conn']
    synapse_l2_tm1 = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_conn_tm1['g'],
                                                  reversal_potential=params_conn_tm1['reversal'], e_lo=activity_range,
                                                  e_hi=2*activity_range)
    synapse_l3_tm9 = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_conn_tm9['g'],
                                                  reversal_potential=params_conn_tm9['reversal'], e_lo=0.0,
                                                  e_hi=activity_range)
    net.add_connection(synapse_l2_tm1, 'L2_out', 'Tm1')
    net.add_connection(synapse_l3_tm9, 'L3', 'Tm9')

    net.add_output('Tm1')
    net.add_output('Tm9')

    """
    ####################################################################################################################
    CT1 COMPARTMENTS
    """
    params_conn_ct1_on = params['CT1OnConn']
    synapse_mi1_ct1_on = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_conn_ct1_on['g'],
                                                      reversal_potential=params_conn_ct1_on['reversal'], e_lo=0.0,
                                                      e_hi=activity_range)
    params_conn_ct1_off = params['CT1OffConn']
    synapse_tm1_ct1_off = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_conn_ct1_off['g'],
                                                       reversal_potential=params_conn_ct1_off['reversal'], e_lo=0.0,
                                                       e_hi=activity_range)

    params_node_ct1_on = params['CT1On']
    add_lowpass_filter(net, cutoff=params_node_ct1_on['params']['cutoff'], name='CT1_On',
                       invert=params_node_ct1_on['params']['invert'], bias=params_node_ct1_on['params']['bias'],
                       initial_value=params_node_ct1_on['params']['initialValue'], shape=shape, color='gold')
    params_node_ct1_off = params['CT1Off']
    add_lowpass_filter(net, cutoff=params_node_ct1_off['params']['cutoff'], name='CT1_Off',
                       invert=params_node_ct1_off['params']['invert'], bias=params_node_ct1_off['params']['bias'],
                       initial_value=params_node_ct1_off['params']['initialValue'], shape=shape, color='darkgoldenrod')

    net.add_connection(synapse_mi1_ct1_on, 'Mi1', 'CT1_On')
    net.add_connection(synapse_tm1_ct1_off, 'Tm1', 'CT1_Off')

    net.add_output('CT1_On')
    net.add_output('CT1_Off')

    """
    ####################################################################################################################
    T4 CELLS
    """
    params_node_t4 = params['T4']
    params_conn_t4 = params['T4Conn']

    # g_mi1_t4, rev_mi1_t4 = synapse_target(activity_range, 0.0)
    g_pd_t4, rev_pd_t4 = synapse_target(0.0, activity_range)

    g_nd_t4 = params['CInv'] - 1
    rev_nd_t4 = 0.0

    cond_mi1, rev_mi1 = params_conn_t4['g'], params_conn_t4['reversal']
    cond_mi9, rev_mi9 = g_nd_t4, rev_nd_t4
    cond_ct1_on, rev_ct1_on = g_pd_t4, g_nd_t4

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


    add_lowpass_filter(net, params_node_t4['params']['cutoff'], name='T4_a',
                       invert=False,
                       initial_value=0.0,
                       bias=activity_range, shape=shape, color='purple')
    add_lowpass_filter(net, params_node_t4['params']['cutoff'], name='T4_b',
                       invert=False,
                       initial_value=0.0,
                       bias=activity_range, shape=shape, color='purple')
    add_lowpass_filter(net, params_node_t4['params']['cutoff'], name='T4_c',
                       invert=False,
                       initial_value=0.0,
                       bias=activity_range, shape=shape, color='purple')
    add_lowpass_filter(net, params_node_t4['params']['cutoff'], name='T4_d',
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

    net.add_output('T4_a')
    net.add_output('T4_b')
    net.add_output('T4_c')
    net.add_output('T4_d')

    """
    ####################################################################################################################
    T5 CELLS
    """
    params_node_t5 = params['T5']

    g_nd_t5, rev_nd_t5 = synapse_target(activity_range, 0.0)
    g_cd_t5 = g_nd_t5
    rev_cd_t5 = rev_nd_t5
    g_pd_t5 = params['CT1OffG']
    rev_pd_t5 = params['CT1OffReversal']

    cond_tm1, rev_tm1 = g_cd_t5/2, rev_cd_t5
    cond_tm9, rev_tm9 = g_nd_t5/2, rev_nd_t5
    cond_ct1_off, rev_ct1_off = g_pd_t5, rev_pd_t5

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

    synapse_tm1_t5 = NonSpikingOneToOneConnection(shape=shape, max_conductance=cond_tm1, reversal_potential=rev_tm1,
                                                  e_lo=0.0, e_hi=activity_range)

    synapse_tm9_t5_a = NonSpikingPatternConnection(max_conductance_kernel=cond_tm9_kernel_a,
                                                   reversal_potential_kernel=rev_tm9_kernel_a, e_lo_kernel=e_lo_kernel,
                                                   e_hi_kernel=e_hi_kernel)
    synapse_ct1_off_t5_a = NonSpikingPatternConnection(max_conductance_kernel=cond_ct1_off_kernel_a,
                                                     reversal_potential_kernel=rev_ct1_off_kernel_a,
                                                     e_lo_kernel=e_lo_kernel,
                                                     e_hi_kernel=e_hi_kernel)
    synapse_tm9_t5_b = NonSpikingPatternConnection(max_conductance_kernel=cond_tm9_kernel_b,
                                                   reversal_potential_kernel=rev_tm9_kernel_b, e_lo_kernel=e_lo_kernel,
                                                   e_hi_kernel=e_hi_kernel)
    synapse_ct1_off_t5_b = NonSpikingPatternConnection(max_conductance_kernel=cond_ct1_off_kernel_b,
                                                     reversal_potential_kernel=rev_ct1_off_kernel_b,
                                                     e_lo_kernel=e_lo_kernel,
                                                     e_hi_kernel=e_hi_kernel)
    synapse_tm9_t5_c = NonSpikingPatternConnection(max_conductance_kernel=cond_tm9_kernel_c,
                                                   reversal_potential_kernel=rev_tm9_kernel_c, e_lo_kernel=e_lo_kernel,
                                                   e_hi_kernel=e_hi_kernel)
    synapse_ct1_off_t5_c = NonSpikingPatternConnection(max_conductance_kernel=cond_ct1_off_kernel_c,
                                                     reversal_potential_kernel=rev_ct1_off_kernel_c,
                                                     e_lo_kernel=e_lo_kernel,
                                                     e_hi_kernel=e_hi_kernel)
    synapse_tm9_t5_d = NonSpikingPatternConnection(max_conductance_kernel=cond_tm9_kernel_d,
                                                   reversal_potential_kernel=rev_tm9_kernel_d, e_lo_kernel=e_lo_kernel,
                                                   e_hi_kernel=e_hi_kernel)
    synapse_ct1_off_t5_d = NonSpikingPatternConnection(max_conductance_kernel=cond_ct1_off_kernel_d,
                                                     reversal_potential_kernel=rev_ct1_off_kernel_d,
                                                     e_lo_kernel=e_lo_kernel,
                                                     e_hi_kernel=e_hi_kernel)

    add_lowpass_filter(net, params_node_t5['params']['cutoff'], name='T5_a',
                       invert=False,
                       initial_value=0.0,
                       bias=activity_range, shape=shape, color='plum')
    add_lowpass_filter(net, params_node_t5['params']['cutoff'], name='T5_b',
                       invert=False,
                       initial_value=0.0,
                       bias=activity_range, shape=shape, color='plum')
    add_lowpass_filter(net, params_node_t5['params']['cutoff'], name='T5_c',
                       invert=False,
                       initial_value=0.0,
                       bias=activity_range, shape=shape, color='plum')
    add_lowpass_filter(net, params_node_t5['params']['cutoff'], name='T5_d',
                       invert=False,
                       initial_value=0.0,
                       bias=activity_range, shape=shape, color='plum')

    net.add_connection(synapse_tm1_t5, 'Tm1', 'T5_a')
    net.add_connection(synapse_tm9_t5_a, 'Tm9', 'T5_a')
    net.add_connection(synapse_ct1_off_t5_a, 'CT1_Off', 'T5_a')
    net.add_connection(synapse_tm1_t5, 'Tm1', 'T5_b')
    net.add_connection(synapse_tm9_t5_b, 'Tm9', 'T5_b')
    net.add_connection(synapse_ct1_off_t5_b, 'CT1_Off', 'T5_b')
    net.add_connection(synapse_tm1_t5, 'Tm1', 'T5_c')
    net.add_connection(synapse_tm9_t5_c, 'Tm9', 'T5_c')
    net.add_connection(synapse_ct1_off_t5_c, 'CT1_Off', 'T5_c')
    net.add_connection(synapse_tm1_t5, 'Tm1', 'T5_d')
    net.add_connection(synapse_tm9_t5_d, 'Tm9', 'T5_d')
    net.add_connection(synapse_ct1_off_t5_d, 'CT1_Off', 'T5_d')

    net.add_output('T5_a')
    net.add_output('T5_b')
    net.add_output('T5_c')
    net.add_output('T5_d')

    """
    ####################################################################################################################
    EXPORT
    """
    # render(net, view=True)
    model = net.compile(params['dt'], backend='torch', device=device)

    return model, net
