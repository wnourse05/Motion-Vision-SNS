# from Tuning.gen_retina_params import tune_retina
from Tuning.gen_l1_params import tune_l1
from Tuning.gen_l2_params import tune_l2
# from Tuning.gen_l3_params import tune_l3

from Tuning.gen_mi1_params import tune_mi1
# from Tuning.gen_mi9_params import tune_mi9
from Tuning.gen_tm1_params import tune_tm1
# from Tuning.gen_tm9_params import tune_tm9

from Tuning.gen_ct1_on_params import tune_ct1_on
from Tuning.gen_ct1_off_params import tune_ct1_off

def tune_neurons(cutoffs):
    # tune_retina(cutoff_retina)
    tune_l1(cutoff_l1_low, cutoff_lamina_high)