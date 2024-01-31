from utilities import calc_cap_from_cutoff, save_data
from Tuning.tune_net import tune

def gen_params(scale=None):
    if scale is None:
        scale = [1,1,1,1,1,1,1,1]
    v_slow = 10
    v_fast = 360
    wavelength = 30
    dt = 0.1

    cap_fast = 10*dt
    cutoff_fast = calc_cap_from_cutoff(cap_fast)
    cap_low = 100*wavelength/v_fast
    cutoff_low = calc_cap_from_cutoff(cap_low)
    cap_enhance = 1000/v_slow
    cutoff_enhance = calc_cap_from_cutoff(cap_enhance)
    cutoff_suppress = cutoff_fast
    div = 10
    cutoff_emd = cutoff_fast

    ratio_enhance_off = 0.5
    cutoff_direct_off = 10*v_slow/wavelength
    cutoff_direct_on = cutoff_fast

    param_vector = [scale[0]*cutoff_fast, scale[1]*cutoff_low, scale[2]*cutoff_enhance, scale[3]*cutoff_suppress, scale[4]*div, scale[5]*ratio_enhance_off, scale[6]*cutoff_direct_on, scale[7]*cutoff_direct_off, dt]
    print(param_vector)

    param_dict = tune(param_vector)
    return param_dict

def main():
    param_dict = gen_params()
    save_data(param_dict, 'params_net_20230327.pc')

if __name__ == '__main__':
    main()
