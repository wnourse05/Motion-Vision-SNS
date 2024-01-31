from utilities import activity_range, synapse_target

def tune_lp(cutoff):
    target_center = 0.0
    target_middle = 1.0 * activity_range
    target_outer = 17 / 16 * activity_range
    outer_conductance, outer_reversal = synapse_target(target_outer, activity_range)
    middle_conductance, middle_reversal = synapse_target(target_middle, activity_range)
    center_conductance, center_reversal = synapse_target(target_center, activity_range)
    g = {'outer': outer_conductance,
         'middle': middle_conductance,
         'center': center_conductance}
    reversal = {'outer': outer_reversal,
                'middle': middle_reversal,
                'center': center_reversal}


    params = {'cutoff': cutoff,
              'invert': True,
              'initialValue': activity_range,
              'bias': 0.0,
              'g': g,
              'reversal': reversal}


    return params
