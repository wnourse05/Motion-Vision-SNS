import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

from sns_toolbox.connections import NonSpikingSynapse, NonSpikingPatternConnection
from sns_toolbox.networks import Network
from sns_toolbox.renderer import render
from utilities import add_lowpass_filter, activity_range, load_data, add_scaled_bandpass_filter, backend, NonSpikingOneToOneConnection, synapse_target, device, cutoff_fastest, reversal_ex, reversal_in, calc_cap_from_cutoff
from Tuning.tune_neurons_old import tune_neurons

def __gen_receptive_fields__(params, center=False):
    g_params = params['g']
    if center:
        g_params['outer'] = 0
        g_params['middle'] = 0
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

def gen_single_column(params):
    """
    ####################################################################################################################
    NETWORK
    """
    net = Network('Motion Vision Single Column')

    """
    ####################################################################################################################
    INPUT
    """

    params_in = params['in']
    add_lowpass_filter(net, params_in['cutoff'], name='In', invert=False, initial_value=0.0, bias=0.0, color='black')

    net.add_input('In')
    net.add_output('In', name='OutIn')

    """
    ####################################################################################################################
    LAYER 1
    """
    params_bp = params['bp']
    params_lp = params['lp']
    synapse_in_bp = NonSpikingSynapse(max_conductance=params_bp['g']['center'], reversal_potential=params_bp['reversal']['center'], e_lo=0.0, e_hi=activity_range)
    synapse_in_lp = NonSpikingSynapse(max_conductance=params_lp['g']['center'], reversal_potential=params_lp['reversal']['center'], e_lo=0.0, e_hi=activity_range)

    add_scaled_bandpass_filter(net, params_bp['cutoffLow'], params_bp['cutoffHigh'],
                               params_bp['gain'], invert=params_bp['invert'], name='BP')
    add_lowpass_filter(net, cutoff=params_lp['cutoff'], name='LP', invert=params_lp['invert'], initial_value=params_lp['initialValue'], bias=params_lp['bias'])

    net.add_connection(synapse_in_bp, 'In', 'BP_in')
    net.add_connection(synapse_in_lp, 'In', 'LP')

    net.add_output('BP_out', name='OutBP')
    net.add_output('LP', name='OutLP')

    """
    ####################################################################################################################
    LAYER 2
    """
    params_e = params['e']
    add_lowpass_filter(net, cutoff=params_e['cutoff'], name='E', invert=params_e['invert'], bias=params_e['bias'], initial_value=params_e['initialValue'])
    synapse_lp_e = NonSpikingSynapse(max_conductance=params_e['g'], reversal_potential=params_e['reversal'], e_lo=0.0, e_hi=activity_range)
    net.add_connection(synapse_lp_e, 'LP', 'E')
    net.add_output('E')

    params_d_on = params['d_on']
    add_lowpass_filter(net, cutoff=params_d_on['cutoff'], name='D On', invert=params_d_on['invert'], bias=params_d_on['bias'],
                       initial_value=params_d_on['initialValue'])
    synapse_bp_d_on = NonSpikingSynapse(max_conductance=params_d_on['g'], reversal_potential=params_d_on['reversal'], e_lo=0.0,
                                        e_hi=activity_range)
    net.add_connection(synapse_bp_d_on, 'BP_out', 'D On')
    net.add_output('D On')

    params_d_off = params['d_off']
    add_lowpass_filter(net, cutoff=params_d_off['cutoff'], name='D Off', invert=params_d_off['invert'],
                       bias=params_d_off['bias'],
                       initial_value=params_d_off['initialValue'])
    synapse_bp_d_off = NonSpikingSynapse(max_conductance=params_d_off['g'], reversal_potential=params_d_off['reversal'],
                                         e_lo=activity_range,
                                         e_hi=2*activity_range)
    net.add_connection(synapse_bp_d_off, 'BP_out', 'D Off')
    net.add_output('D Off')

    """
    ####################################################################################################################
    LAYER 2.5
    """
    params_s_on = params['s_on']
    params_s_off = params['s_off']
    synapse_d_on_s_on = NonSpikingSynapse(max_conductance=params_s_on['g'], reversal_potential=params_s_on['reversal'], e_lo=0.0, e_hi=activity_range)
    synapse_d_off_s_off = NonSpikingSynapse(max_conductance=params_s_off['g'], reversal_potential=params_s_off['reversal'], e_lo=0.0, e_hi=activity_range)

    add_lowpass_filter(net, cutoff=params_s_on['cutoff'], name='S On', invert=params_s_on['invert'], bias=params_s_on['bias'], initial_value=params_s_on['initialValue'])
    add_lowpass_filter(net, cutoff=params_s_off['cutoff'], name='S Off', invert=params_s_off['invert'], bias=params_s_off['bias'], initial_value=params_s_off['initialValue'])

    net.add_connection(synapse_d_on_s_on, 'D On', 'S On')
    net.add_connection(synapse_d_off_s_off, 'D Off', 'S Off')

    net.add_output('S On')
    net.add_output('S Off')

    """
    ####################################################################################################################
    EXPORT
    """
    # render(net, view=True)
    model = net.compile(params['dt'], backend='numpy', device='cpu')

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

    synapse_cd_t5 = NonSpikingSynapse(max_conductance=g_cd_t5*0.3, reversal_potential=rev_cd_t5, e_lo=0.0, e_hi=activity_range)
    synapse_pd_t5 = NonSpikingSynapse(max_conductance=g_pd_t5, reversal_potential=rev_pd_t5, e_lo=0.0, e_hi=activity_range)
    synapse_nd_t5 = NonSpikingSynapse(max_conductance=g_nd_t5*0.6, reversal_potential=rev_nd_t5, e_lo=0.0, e_hi=activity_range)

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

def gen_motion_vision(params, shape, backend, device, center=False):
    """
    ####################################################################################################################
    INITIALIZE NETWORK
    """
    net = Network('Motion Vision Network')
    flat_size = int(shape[0]*shape[1])

    """
    ####################################################################################################################
    INPUT
    """
    params_in = params['in']
    add_lowpass_filter(net, params_in['cutoff'], name='In',
                       invert=False,
                       initial_value=0.0,
                       bias=0.0, shape=shape, color='black')

    net.add_input('In', size=flat_size)
    net.add_output('In')

    """
    ####################################################################################################################
    LAYER 1
    """
    params_bp = params['bp']
    params_lp = params['lp']
    g_in_bp, reversal_in_bp, e_lo_in_bp, e_hi_in_bp = __gen_receptive_fields__(params_bp, center=center)
    g_in_lp, reversal_in_lp, e_lo_in_lp, e_hi_in_lp = __gen_receptive_fields__(params_lp, center=center)
    synapse_in_bp = NonSpikingPatternConnection(max_conductance_kernel=g_in_bp, reversal_potential_kernel=reversal_in_bp, e_lo_kernel=e_lo_in_bp, e_hi_kernel=e_hi_in_bp)
    synapse_in_lp = NonSpikingPatternConnection(max_conductance_kernel=g_in_lp, reversal_potential_kernel=reversal_in_lp, e_lo_kernel=e_lo_in_lp, e_hi_kernel=e_hi_in_lp)

    add_scaled_bandpass_filter(net, params_bp['cutoffLow'], params_bp['cutoffHigh'],
                               params_bp['gain'], invert=params_bp['invert'], name='BP',
                               shape=shape, color='darkgreen')
    add_lowpass_filter(net, cutoff=params_lp['cutoff'], name='LP',
                       invert=params_lp['invert'],
                       initial_value=params_lp['initialValue'], bias=params_lp['bias'],
                       shape=shape, color='lightgreen')

    net.add_connection(synapse_in_bp, 'In', 'BP_in')
    net.add_connection(synapse_in_lp, 'In', 'LP')
    net.add_output('BP_out')
    net.add_output('LP')

    """
    ####################################################################################################################
    LAYER 2
    """
    params_e = params['e']
    add_lowpass_filter(net, cutoff=params_e['cutoff'], name='E',
                       invert=params_e['invert'], bias=params_e['bias'],
                       initial_value=params_e['initialValue'], shape=shape, color='indianred')
    synapse_lp_e = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_e['g'],
                                                  reversal_potential=params_e['reversal'], e_lo=0.0,
                                                  e_hi=activity_range)
    net.add_connection(synapse_lp_e, 'LP', 'E')
    net.add_output('E')


    params_d_on = params['d_on']

    add_lowpass_filter(net, cutoff=params_d_on['cutoff'], name='D On',
                       invert=params_d_on['invert'], bias=params_d_on['bias'],
                       initial_value=params_d_on['initialValue'], shape=shape, color='red')
    synapse_bp_d_on = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_d_on['g'],
                                                  reversal_potential=params_d_on['reversal'], e_lo=0.0,
                                                  e_hi=activity_range)

    net.add_connection(synapse_bp_d_on, 'BP_out', 'D On')
    net.add_output('D On')

    params_d_off = params['d_off']
    add_lowpass_filter(net, cutoff=params_d_off['cutoff'], name='D Off',
                       invert=params_d_off['invert'], bias=params_d_off['bias'],
                       initial_value=params_d_off['initialValue'], shape=shape, color='navy')
    synapse_bp_d_off = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_d_off['g'],
                                                  reversal_potential=params_d_off['reversal'], e_lo=activity_range,
                                                  e_hi=2*activity_range)

    net.add_connection(synapse_bp_d_off, 'BP_out', 'D Off')

    net.add_output('D Off')

    """
    ####################################################################################################################
    LAYER 2.5
    """
    params_s_on = params['s_on']
    synapse_d_on_s_on = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_s_on['g'],
                                                      reversal_potential=params_s_on['reversal'], e_lo=0.0,
                                                      e_hi=activity_range)
    params_s_off = params['s_off']
    synapse_d_off_s_off = NonSpikingOneToOneConnection(shape=shape, max_conductance=params_s_off['g'],
                                                       reversal_potential=params_s_off['reversal'], e_lo=0.0,
                                                       e_hi=activity_range)

    add_lowpass_filter(net, cutoff=params_s_on['cutoff'], name='S On',
                       invert=params_s_on['invert'], bias=params_s_on['bias'],
                       initial_value=params_s_on['initialValue'], shape=shape, color='gold')
    add_lowpass_filter(net, cutoff=params_s_off['cutoff'], name='S Off',
                       invert=params_s_off['invert'], bias=params_s_off['bias'],
                       initial_value=params_s_off['initialValue'], shape=shape, color='darkgoldenrod')

    net.add_connection(synapse_d_on_s_on, 'D On', 'S On')
    net.add_connection(synapse_d_off_s_off, 'D Off', 'S Off')

    net.add_output('S On')
    net.add_output('S Off')

    """
    ####################################################################################################################
    ON
    """
    params_on = params['on']

    g_e_on = params_on['g']['enhance']
    g_d_on = params_on['g']['direct']
    g_s_on = params_on['g']['suppress']

    rev_e_on = params_on['reversal']['enhance']
    rev_d_on = params_on['reversal']['direct']
    rev_s_on = params_on['reversal']['suppress']

    g_e_on_kernel_b = np.array([[0, 0, 0],
                                [g_e_on, 0, 0],
                                [0, 0, 0]])
    rev_e_on_kernel_b = np.array([[0, 0, 0],
                               [rev_e_on, 0, 0],
                               [0, 0, 0]])
    g_s_on_kernel_b = np.array([[0, 0, 0],
                                [0, 0, g_s_on],
                                [0, 0, 0]])
    rev_s_on_kernel_b = np.array([[0, 0, 0],
                               [0, 0, rev_s_on],
                               [0, 0, 0]])
    g_e_on_kernel_a, g_e_on_kernel_c, g_e_on_kernel_d = __all_quadrants__(g_e_on_kernel_b)
    g_s_on_kernel_a, g_s_on_kernel_c, g_s_on_kernel_d = __all_quadrants__(g_s_on_kernel_b)
    rev_e_on_kernel_a, rev_e_on_kernel_c, rev_e_on_kernel_d = __all_quadrants__(rev_e_on_kernel_b)
    rev_s_on_kernel_a, rev_s_on_kernel_c, rev_s_on_kernel_d = __all_quadrants__(rev_s_on_kernel_b)
    e_lo_kernel = np.zeros([3,3])
    e_hi_kernel = np.zeros([3,3]) + activity_range

    synapse_d_on_on = NonSpikingOneToOneConnection(shape=shape, max_conductance=g_d_on, reversal_potential=rev_d_on,
                                                  e_lo=0.0, e_hi=activity_range)

    synapse_e_on_on_a = NonSpikingPatternConnection(max_conductance_kernel=g_e_on_kernel_a,
                                                   reversal_potential_kernel=rev_e_on_kernel_a, e_lo_kernel=e_lo_kernel,
                                                   e_hi_kernel=e_hi_kernel)
    synapse_s_on_on_a = NonSpikingPatternConnection(max_conductance_kernel=g_s_on_kernel_a,
                                                     reversal_potential_kernel=rev_s_on_kernel_a,
                                                     e_lo_kernel=e_lo_kernel,
                                                     e_hi_kernel=e_hi_kernel)
    synapse_e_on_on_b = NonSpikingPatternConnection(max_conductance_kernel=g_e_on_kernel_b,
                                                    reversal_potential_kernel=rev_e_on_kernel_b, e_lo_kernel=e_lo_kernel,
                                                    e_hi_kernel=e_hi_kernel)
    synapse_s_on_on_b = NonSpikingPatternConnection(max_conductance_kernel=g_s_on_kernel_b,
                                                      reversal_potential_kernel=rev_s_on_kernel_b, e_lo_kernel=e_lo_kernel,
                                                      e_hi_kernel=e_hi_kernel)
    synapse_e_on_on_c = NonSpikingPatternConnection(max_conductance_kernel=g_e_on_kernel_c,
                                                   reversal_potential_kernel=rev_e_on_kernel_c, e_lo_kernel=e_lo_kernel,
                                                   e_hi_kernel=e_hi_kernel)
    synapse_s_on_on_c = NonSpikingPatternConnection(max_conductance_kernel=g_s_on_kernel_c,
                                                     reversal_potential_kernel=rev_s_on_kernel_c,
                                                     e_lo_kernel=e_lo_kernel,
                                                     e_hi_kernel=e_hi_kernel)
    synapse_e_on_on_d = NonSpikingPatternConnection(max_conductance_kernel=g_e_on_kernel_d,
                                                   reversal_potential_kernel=rev_e_on_kernel_d, e_lo_kernel=e_lo_kernel,
                                                   e_hi_kernel=e_hi_kernel)
    synapse_s_on_on_d = NonSpikingPatternConnection(max_conductance_kernel=g_s_on_kernel_d,
                                                     reversal_potential_kernel=rev_s_on_kernel_d,
                                                     e_lo_kernel=e_lo_kernel,
                                                     e_hi_kernel=e_hi_kernel)


    add_lowpass_filter(net, params_on['cutoff'], name='On A',
                       invert=False,
                       initial_value=0.0,
                       bias=0.0, shape=shape, color='purple')
    add_lowpass_filter(net, params_on['cutoff'], name='On B',
                       invert=False,
                       initial_value=0.0,
                       bias=0.0, shape=shape, color='purple')
    add_lowpass_filter(net, params_on['cutoff'], name='On C',
                       invert=False,
                       initial_value=0.0,
                       bias=0.0, shape=shape, color='purple')
    add_lowpass_filter(net, params_on['cutoff'], name='On D',
                       invert=False,
                       initial_value=0.0,
                       bias=0.0, shape=shape, color='purple')

    net.add_connection(synapse_d_on_on, 'D On', 'On A')
    net.add_connection(synapse_e_on_on_a, 'E', 'On A')
    net.add_connection(synapse_s_on_on_a, 'S On', 'On A')
    net.add_connection(synapse_d_on_on, 'D On', 'On B')
    net.add_connection(synapse_e_on_on_b, 'E', 'On B')
    net.add_connection(synapse_s_on_on_b, 'S On', 'On B')
    net.add_connection(synapse_d_on_on, 'D On', 'On C')
    net.add_connection(synapse_e_on_on_c, 'E', 'On C')
    net.add_connection(synapse_s_on_on_c, 'S On', 'On C')
    net.add_connection(synapse_d_on_on, 'D On', 'On D')
    net.add_connection(synapse_e_on_on_d, 'E', 'On D')
    net.add_connection(synapse_s_on_on_d, 'S On', 'On D')

    net.add_output('On A')
    net.add_output('On B')
    net.add_output('On C')
    net.add_output('On D')

    """
    ####################################################################################################################
    T5 CELLS
    """
    params_off = params['off']

    g_e_off = params_off['g']['enhance']
    g_d_off = params_off['g']['direct']
    g_s_off = params_off['g']['suppress']

    rev_e_off = params_off['reversal']['enhance']
    rev_d_off = params_off['reversal']['direct']
    rev_s_off = params_off['reversal']['suppress']

    g_e_off_kernel_b = np.array([[0, 0, 0],
                                [g_e_off, 0, 0],
                                [0, 0, 0]])
    rev_e_off_kernel_b = np.array([[0, 0, 0],
                                  [rev_e_off, 0, 0],
                                  [0, 0, 0]])
    g_s_off_kernel_b = np.array([[0, 0, 0],
                                [0, 0, g_s_off],
                                [0, 0, 0]])
    rev_s_off_kernel_b = np.array([[0, 0, 0],
                                  [0, 0, rev_s_off],
                                  [0, 0, 0]])
    g_e_off_kernel_a, g_e_off_kernel_c, g_e_off_kernel_d = __all_quadrants__(g_e_off_kernel_b)
    g_s_off_kernel_a, g_s_off_kernel_c, g_s_off_kernel_d = __all_quadrants__(g_s_off_kernel_b)
    rev_e_off_kernel_a, rev_e_off_kernel_c, rev_e_off_kernel_d = __all_quadrants__(rev_e_off_kernel_b)
    rev_s_off_kernel_a, rev_s_off_kernel_c, rev_s_off_kernel_d = __all_quadrants__(rev_s_off_kernel_b)
    e_lo_kernel = np.zeros([3, 3])
    e_hi_kernel = np.zeros([3, 3]) + activity_range

    synapse_d_off_off = NonSpikingOneToOneConnection(shape=shape, max_conductance=g_d_off, reversal_potential=rev_d_off,
                                                   e_lo=0.0, e_hi=activity_range)

    synapse_e_off_off_a = NonSpikingPatternConnection(max_conductance_kernel=g_e_off_kernel_a,
                                                    reversal_potential_kernel=rev_e_off_kernel_a,
                                                    e_lo_kernel=e_lo_kernel,
                                                    e_hi_kernel=e_hi_kernel)
    synapse_s_off_off_a = NonSpikingPatternConnection(max_conductance_kernel=g_s_off_kernel_a,
                                                    reversal_potential_kernel=rev_s_off_kernel_a,
                                                    e_lo_kernel=e_lo_kernel,
                                                    e_hi_kernel=e_hi_kernel)
    synapse_e_off_off_b = NonSpikingPatternConnection(max_conductance_kernel=g_e_off_kernel_b,
                                                    reversal_potential_kernel=rev_e_off_kernel_b,
                                                    e_lo_kernel=e_lo_kernel,
                                                    e_hi_kernel=e_hi_kernel)
    synapse_s_off_off_b = NonSpikingPatternConnection(max_conductance_kernel=g_s_off_kernel_b,
                                                    reversal_potential_kernel=rev_s_off_kernel_b,
                                                    e_lo_kernel=e_lo_kernel,
                                                    e_hi_kernel=e_hi_kernel)
    synapse_e_off_off_c = NonSpikingPatternConnection(max_conductance_kernel=g_e_off_kernel_c,
                                                    reversal_potential_kernel=rev_e_off_kernel_c,
                                                    e_lo_kernel=e_lo_kernel,
                                                    e_hi_kernel=e_hi_kernel)
    synapse_s_off_off_c = NonSpikingPatternConnection(max_conductance_kernel=g_s_off_kernel_c,
                                                    reversal_potential_kernel=rev_s_off_kernel_c,
                                                    e_lo_kernel=e_lo_kernel,
                                                    e_hi_kernel=e_hi_kernel)
    synapse_e_off_off_d = NonSpikingPatternConnection(max_conductance_kernel=g_e_off_kernel_d,
                                                    reversal_potential_kernel=rev_e_off_kernel_d,
                                                    e_lo_kernel=e_lo_kernel,
                                                    e_hi_kernel=e_hi_kernel)
    synapse_s_off_off_d = NonSpikingPatternConnection(max_conductance_kernel=g_s_off_kernel_d,
                                                    reversal_potential_kernel=rev_s_off_kernel_d,
                                                    e_lo_kernel=e_lo_kernel,
                                                    e_hi_kernel=e_hi_kernel)

    add_lowpass_filter(net, params_off['cutoff'], name='Off A',
                       invert=False,
                       initial_value=0.0,
                       bias=0.0, shape=shape, color='purple')
    add_lowpass_filter(net, params_off['cutoff'], name='Off B',
                       invert=False,
                       initial_value=0.0,
                       bias=0.0, shape=shape, color='purple')
    add_lowpass_filter(net, params_off['cutoff'], name='Off C',
                       invert=False,
                       initial_value=0.0,
                       bias=0.0, shape=shape, color='purple')
    add_lowpass_filter(net, params_off['cutoff'], name='Off D',
                       invert=False,
                       initial_value=0.0,
                       bias=0.0, shape=shape, color='purple')

    net.add_connection(synapse_d_off_off, 'D Off', 'Off A')
    net.add_connection(synapse_d_off_off, 'D Off', 'Off B')
    net.add_connection(synapse_d_off_off, 'D Off', 'Off C')
    net.add_connection(synapse_d_off_off, 'D Off', 'Off D')
    net.add_connection(synapse_e_off_off_a, 'E', 'Off A')
    net.add_connection(synapse_e_off_off_b, 'E', 'Off B')
    net.add_connection(synapse_e_off_off_c, 'E', 'Off C')
    net.add_connection(synapse_e_off_off_d, 'E', 'Off D')
    net.add_connection(synapse_s_off_off_a, 'S Off', 'Off A')
    net.add_connection(synapse_s_off_off_b, 'S Off', 'Off B')
    net.add_connection(synapse_s_off_off_c, 'S Off', 'Off C')
    net.add_connection(synapse_s_off_off_d, 'S Off', 'Off D')

    net.add_output('Off A')
    net.add_output('Off B')
    net.add_output('Off C')
    net.add_output('Off D')

    """
    ####################################################################################################################
    EXPORT
    """
    # render(net, view=True)
    model = net.compile(params['dt'], backend=backend, device=device)

    return model, net
