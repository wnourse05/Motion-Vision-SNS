import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from sns_toolbox.renderer import render
from utilities import add_lowpass_filter, activity_range, dt, load_data, add_scaled_bandpass_filter
from Tuning.tune_neurons import tune_neurons

def gen_single_column(cutoffs):
    """
    ########################################################################################################################
    NETWORK
    """
    tune_neurons(cutoffs)
    net = Network('Motion Vision Single Column')

    """
    ########################################################################################################################
    RETINA
    """

    params_node_retina = load_data('params_node_retina.p')
    add_lowpass_filter(net, params_node_retina['params']['cutoff'], name='Retina', invert=params_node_retina['params']['invert'], initial_value=params_node_retina['params']['initialValue'], bias=params_node_retina['params']['bias'])

    net.add_input('Retina')
    net.add_output('Retina', name='OutR')

    """
    ########################################################################################################################
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
    l3 = add_lowpass_filter(net, cutoff=params_node_l3['params']['cutoff'], name='L3', invert=params_node_l3['params']['invert'], initial_value=params_node_l3['params']['initialValue'], bias=params_node_l3['params']['bias'])
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
    ########################################################################################################################
    MEDULLA ON
    """
    params_node_mi1 = load_data('params_node_mi1.p')
    params_node_mi9 = load_data('params_node_mi9.p')
    mi1 = add_lowpass_filter(net, cutoff=params_node_mi1['params']['cutoff'], name='Mi1', invert=params_node_mi1['params']['invert'], bias=params_node_mi1['params']['bias'], initial_value=params_node_mi1['params']['initialValue'])
    mi9 = add_lowpass_filter(net, cutoff=params_node_mi9['params']['cutoff'], name='Mi9', invert=params_node_mi9['params']['invert'], bias=params_node_mi9['params']['bias'], initial_value=params_node_mi9['params']['initialValue'])

    params_conn_mi1 = load_data('params_conn_mi1.p')
    params_conn_mi9 = load_data('params_conn_mi9.p')
    synapse_l1_mi1 = NonSpikingSynapse(max_conductance=params_conn_mi1['g'], reversal_potential=params_conn_mi1['reversal'], e_lo=0.0, e_hi=activity_range)
    synapse_l3_mi9 = NonSpikingSynapse(max_conductance=params_conn_mi9['g'], reversal_potential=params_conn_mi9['reversal'], e_lo=0.0, e_hi=activity_range)
    net.add_connection(synapse_l1_mi1, 'L1_out', 'Mi1')
    net.add_connection(synapse_l3_mi9, 'L3', 'Mi9')

    net.add_output('Mi1', name='OutMi1')
    net.add_output('Mi9', name='OutMi9')

    """
    ########################################################################################################################
    MEDULLA OFF
    """
    params_conn_tm1 = load_data('params_conn_tm1.p')
    params_conn_tm9 = load_data('params_conn_tm9.p')
    synapse_l2_tm1 = NonSpikingSynapse(max_conductance=params_conn_tm1['g'], reversal_potential=params_conn_tm1['reversal'], e_lo=0.0, e_hi=activity_range)
    synapse_l3_tm9 = NonSpikingSynapse(max_conductance=params_conn_tm9['g'], reversal_potential=params_conn_tm9['reversal'], e_lo=0.0, e_hi=activity_range)

    params_node_tm1 = load_data('params_node_tm1.p')
    params_node_tm9 = load_data('params_node_tm9.p')
    tm1 = add_lowpass_filter(net, cutoff=params_node_tm1['params']['cutoff'], name='Tm1', invert=params_node_tm1['params']['invert'], bias=params_node_tm1['params']['bias'], initial_value=params_node_tm1['params']['initialValue'])
    tm9 = add_lowpass_filter(net, cutoff=params_node_tm9['params']['cutoff'], name='Tm9', invert=params_node_tm9['params']['invert'], bias=params_node_tm9['params']['bias'], initial_value=params_node_tm9['params']['initialValue'])
    net.add_connection(synapse_l2_tm1, 'L2_out', 'Tm1')
    net.add_connection(synapse_l3_mi9, 'L3', 'Tm9')

    net.add_output('Tm1', name='OutTm1')
    net.add_output('Tm9', name='OutTm9')

    """
    ########################################################################################################################
    CT1 COMPARTMENTS
    """
    params_conn_ct1_on = load_data('params_conn_ct1_on.p')
    params_conn_ct1_off = load_data('params_conn_ct1_off.p')
    synapse_mi1_ct1on = NonSpikingSynapse(max_conductance=params_conn_ct1_on['g'], reversal_potential=params_conn_ct1_on['reversal'], e_lo=0.0, e_hi=activity_range)
    synapse_tm1_ct1off = NonSpikingSynapse(max_conductance=params_conn_ct1_off['g'], reversal_potential=params_conn_ct1_off['reversal'], e_lo=0.0, e_hi=activity_range)

    params_node_ct1_on = load_data('params_node_ct1_on.p')
    params_node_ct1_off = load_data('params_node_ct1_off.p')
    ct1_on = add_lowpass_filter(net, cutoff=params_node_ct1_on['params']['cutoff'], name='CT1_On', invert=params_node_ct1_on['params']['invert'], bias=params_node_ct1_on['params']['bias'], initial_value=params_node_ct1_on['params']['initialValue'])
    ct1_off = add_lowpass_filter(net, cutoff=params_node_ct1_off['params']['cutoff'], name='CT1_Off', invert=params_node_ct1_off['params']['invert'], bias=params_node_ct1_off['params']['bias'], initial_value=params_node_ct1_off['params']['initialValue'])

    net.add_connection(synapse_mi1_ct1on, 'Mi1', 'CT1_On')
    net.add_connection(synapse_tm1_ct1off, 'Tm1', 'CT1_Off')

    net.add_output('CT1_On', name='OutCT1On')
    net.add_output('CT1_Off', name='OutCT1Off')

    """
    ########################################################################################################################
    SIMULATE
    """
    # render(net, view=True)
    model = net.compile(dt, backend='torch', device='cpu')

    return model, net