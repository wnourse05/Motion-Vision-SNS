from utilities import save_data, synapse_target, activity_range
import numpy as np

def generate_synaptic_matrices(target_outer, target_middle, target_center, bias):
    target_field = np.array([[target_outer, target_outer,  target_outer,  target_outer,  target_outer],
                             [target_outer, target_middle, target_middle, target_middle, target_outer],
                             [target_outer, target_middle, target_center, target_middle, target_outer],
                             [target_outer, target_middle, target_middle, target_middle, target_outer],
                             [target_outer, target_outer,  target_outer,  target_outer,  target_outer]])

    outer_conductance, outer_reversal = synapse_target(target_outer, bias)
    middle_conductance, middle_reversal = synapse_target(target_middle, bias)
    center_conductance, center_reversal = synapse_target(target_center, bias)

    conductance_field = np.array([[outer_conductance,  outer_conductance,  outer_conductance,  outer_conductance,  outer_conductance],
                                  [outer_conductance, middle_conductance, middle_conductance, middle_conductance, outer_conductance],
                                  [outer_conductance, middle_conductance, center_conductance, middle_conductance, outer_conductance],
                                  [outer_conductance, middle_conductance, middle_conductance, middle_conductance, outer_conductance],
                                  [outer_conductance,  outer_conductance,  outer_conductance,  outer_conductance,  outer_conductance]])

    reversal_field = np.array([[outer_reversal,  outer_reversal,  outer_reversal,  outer_reversal, outer_reversal],
                                     [outer_reversal, middle_reversal, middle_reversal, middle_reversal, outer_reversal],
                                     [outer_reversal, middle_reversal, center_reversal, middle_reversal, outer_reversal],
                                     [outer_reversal, middle_reversal, middle_reversal, middle_reversal, outer_reversal],
                                     [outer_reversal,  outer_reversal,  outer_reversal,  outer_reversal, outer_reversal]])

    return conductance_field, reversal_field, target_field

target_center_l1l2 = 0.0
target_middle_l1l2 = 7/8
target_outer_l1l2 = 1.0

target_center_l3 = 0.0
target_middle_l3 = 1.0
target_outer_l3 = 17/16

field_l1l2_conductance, field_l1l2_reversal, field_l1l2_target = generate_synaptic_matrices(target_outer_l1l2, target_middle_l1l2, target_center_l1l2, activity_range)
field_l3_conductance, field_l3_reversal, field_l3_target = generate_synaptic_matrices(target_outer_l3, target_middle_l3, target_center_l3, activity_range)

data_l1l2_field = {'conductance': field_l1l2_conductance,
                   'reversal': field_l1l2_reversal,
                   'target': field_l1l2_target}
data_l3_field = {'conductance': field_l3_conductance,
                   'reversal': field_l3_reversal,
                   'target': field_l3_target}

save_data(data_l1l2_field, 'params_conn_retina_l1_l2.p')
save_data(data_l3_field, 'params_conn_retina_l3.p')
