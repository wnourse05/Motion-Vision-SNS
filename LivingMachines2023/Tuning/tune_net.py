from Tuning.tune_bp import tune_bp
from Tuning.tune_lp import tune_lp
from Tuning.tune_e import tune_e
from Tuning.tune_d_on import tune_d_on
from Tuning.tune_d_off import tune_d_off
from Tuning.tune_s_on import tune_s_on
from Tuning.tune_s_off import tune_s_off
from Tuning.tune_on import tune_on
from Tuning.tune_off import tune_off
def tune(params):
    cutoff_fast, cutoff_low, cutoff_enhance, cutoff_suppress, div, ratio_enhance_off, cutoff_direct_on, cutoff_direct_off, dt = params

    params_in = {'cutoff': cutoff_fast}
    params_bp = tune_bp(cutoff_fast, cutoff_low)
    params_lp = tune_lp(cutoff_fast)
    params_e = tune_e(cutoff_enhance)
    params_d_on = tune_d_on(cutoff_direct_on, params_in, params_bp)
    params_d_off = tune_d_off(cutoff_direct_off, params_in, params_bp)
    params_s_on = tune_s_on(cutoff_suppress, params_in, params_bp, params_d_on)
    params_s_off = tune_s_off(cutoff_suppress, params_in, params_bp, params_d_off)
    params_on = tune_on(cutoff_fast, params_in, params_bp, params_d_on, div)
    params_off = tune_off(cutoff_fast, div, ratio_enhance_off)

    param_dict = {'dt': dt,
                  'in': params_in,
                  'bp': params_bp,
                  'lp': params_lp,
                  'e': params_e,
                  'd_on': params_d_on,
                  'd_off': params_d_off,
                  's_on': params_s_on,
                  's_off': params_s_off,
                  'on': params_on,
                  'off': params_off}

    return param_dict