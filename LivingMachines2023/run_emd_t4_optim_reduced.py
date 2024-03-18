import matplotlib.pyplot as plt
import numpy as np
from motion_vision_networks import gen_single_emd_on
from utilities import cutoff_fastest, gen_gratings, save_data, calc_cap_from_cutoff
from multiprocess import Pool
import os
os.environ['MKL_NUM_THREADS'] = "4"
import datetime
from typing import Dict, Union
import load_conf as lc
import dill
import pypesto
import pypesto.sample
import h5py
import logging
logging.disable()

from pathlib import Path

def plot_2_vars(t, a, b):
    plt.figure()
    plt.plot(t,a)
    plt.plot(t, b)

def convert_deg_vel_to_interval(vel, dt):
    scaled_vel = vel/5    # columns per second
    interval = ((1/scaled_vel)/(dt/1000))
    return interval

def convert_interval_to_deg_vel(interval, dt):
    vel = 5000/(interval*dt)
    return vel

def gen_stimulus(num_cycles):
    stim_on_lr = np.array([[1.0, 0.0, 0.0],
                           [1.0, 1.0, 0.0],
                           [1.0, 1.0, 1.0]])
    for i in range(num_cycles):
        if i == 0:
            stim_full = np.copy(stim_on_lr)
        else:
            stim_full = np.vstack((stim_on_lr, stim_full))
    return stim_full

def sum_of_lsq(goal, data):
    error_vector = (goal-data)**2
    error_lsq = np.sum(error_vector)
    return error_lsq

def test_emd(dt, model, net, stim, interval):
    model.reset()
    interval = int(interval)
    num_samples = np.shape(stim)[0]
    t = np.linspace(0, dt*num_samples*interval, num=num_samples*int(interval))

    data = np.zeros([num_samples*int(interval), net.get_num_outputs_actual()])

    # if interval:
    index = 0
    j = 0
    for i in (range(len(t))):
        if index < num_samples:
            data[i,:] = model(stim[index,:])
        else:
            data[i, :] = model(stim[-1, :])
        j += 1
        if j == interval:
            index += 1
            j = 0
    # else:
    #     for i in range(len(t)):
    #         data_sns_toolbox[i, :] = model_toolbox(stim[i, :])

    # data_sns_toolbox = data_sns_toolbox.to('cpu')
    data = data.transpose()
    # r_l = data_sns_toolbox[0, :]
    # r_c = data_sns_toolbox[1, :]
    # r_r = data_sns_toolbox[2, :]
    # l1_l = data_sns_toolbox[3, :]
    # l1_c = data_sns_toolbox[4, :]
    # l1_r = data_sns_toolbox[5, :]
    # l3_l = data_sns_toolbox[6, :]
    # l3_c = data_sns_toolbox[7, :]
    # l3_r = data_sns_toolbox[8, :]
    # mi1_l = data_sns_toolbox[9, :]
    # mi1_c = data_sns_toolbox[10, :]
    # mi1_r = data_sns_toolbox[11, :]
    # mi9_l = data_sns_toolbox[12, :]
    # mi9_c = data_sns_toolbox[13, :]
    # mi9_r = data_sns_toolbox[14, :]
    # ct1_l = data_sns_toolbox[15, :]
    # ct1_c = data_sns_toolbox[16, :]
    # ct1_r = data_sns_toolbox[17, :]
    t4_a = data[18, :]
    t4_b = data[19, :]

    a_peak = np.max(t4_a)
    b_peak = np.max(t4_b)
    ratio = b_peak / a_peak


    # trial = {'interval':    interval,
    #          't':           t,
    #          'r_l':         r_l,
    #          'r_c':         r_c,
    #          'r_r':         r_r,
    #          'l1_l':        l1_l,
    #          'l1_c':        l1_c,
    #          'l1_r':        l1_r,
    #          'l3_l':        l3_l,
    #          'l3_c':        l3_c,
    #          'l3_r':        l3_r,
    #          'mi1_l':       mi1_l,
    #          'mi1_c':       mi1_c,
    #          'mi1_r':       mi1_r,
    #          'mi9_l':       mi9_l,
    #          'mi9_c':       mi9_c,
    #          'mi9_r':       mi9_r,
    #          'ct1_l':       ct1_l,
    #          'ct1_c':       ct1_c,
    #          'ct1_r':       ct1_r,
    #          't4_a':        t4_a,
    #          't4_b':        t4_b,
    #          'a_peak':      a_peak,
    #          'b_peak':      b_peak}

    # vel_index = convert_interval_to_deg_vel(interval, dt)
    # param_string = '_%i_%i_%i_%i_%i' % (cutoff_fast, ratio_low, cutoff_ct1, cutoff_mi9, c_inv)
    # filename = dir + str(int(vel_index)) + '_trial' + param_string
    # save_data(trial, filename)

    return a_peak, b_peak, ratio

def freq_response_emd(dt, intervals, num_intervals, stim, params_mcmc):
    cutoff_fast = params_mcmc[0]
    ratio_low = params_mcmc[1]
    cutoff_ct1 = params_mcmc[2]
    cutoff_mi9 = params_mcmc[3]
    c_inv = params_mcmc[4]

    #                   Retina          L1 low              L1 High         L3              Mi1     Mi9          CT1 On          T4     Div gain
    params = np.array([cutoff_fast, cutoff_fast/ratio_low, cutoff_fast, cutoff_fast, cutoff_fast, cutoff_mi9, cutoff_ct1, cutoff_fast, 1/c_inv])   # Good guess
    # print(params)

    model, net = gen_single_emd_on(dt, params)

    # param_string = '_%i_%i_%i_%i_%i.pc' % (cutoff_fast, ratio_low, cutoff_ct1, cutoff_mi9, c_inv)
    # dir = 'T4 Velocity/'

    a_peaks = np.zeros_like(intervals)
    b_peaks = np.zeros_like(intervals)
    ratios = np.zeros_like(intervals)

    for i in range(num_intervals):
        a_peaks[i], b_peaks[i], ratios[i] = test_emd(dt, model, net, stim, intervals[i])

    # data_sns_toolbox = {'vels': vels,
    #         'a_peaks': a_peaks,
    #         'b_peaks': b_peaks,
    #         'ratios': ratios}
    # filename = dir + 'set' + param_string
    # save_data(data_sns_toolbox, filename)

    return a_peaks, b_peaks, ratios

def cost_function(a_peaks, b_peaks, ratios, goal):
    cost = 100*sum_of_lsq(b_peaks, goal)
    if np.isnan(cost):
        cost =1e5

    return cost

def evaluate(params_mcmc, dt, intervals, num_intervals, stim, goal):
    a_peaks, b_peaks, ratios = freq_response_emd(dt, intervals, num_intervals, stim, params_mcmc)
    cost = cost_function(a_peaks, b_peaks, ratios, goal)
    return cost

def easy_neg_log_likelihood(sample_params: np.ndarray, actual_indices: list, rest_params: np.ndarray, vels, num_intervals, stim, goal) -> Union[float, np.ndarray]:

    """
    likelihood function to compare actual vs estimated params with negative log
    :param sample_params: estimated parameters
    :param actual_indices: indices of parameter vector which are sampled
    :param rest_params:
    :return: loss
    """
    real_params = rest_params.copy()
    for s_idx, replace_idx in enumerate(actual_indices):
        real_params[replace_idx] = sample_params[s_idx]

    dt = min(calc_cap_from_cutoff(real_params[0])/10, 0.1)
    intervals = convert_deg_vel_to_interval(vels, dt)
    the_cost = evaluate(real_params, dt, intervals, num_intervals, stim, goal)

    current_datetime = datetime.datetime.now().strftime('%Y-%b-%d_%H-%M')
    the_cost = np.clip(the_cost, a_min=0.0, a_max=10e5)
    print(f"{current_datetime} cost: {the_cost}", flush=True)
    return the_cost   * 1e3

def easy_neg_log_prior(num_params: int) -> pypesto.objective.NegLogParameterPriors:
    """
    prior distribution or prior knowledge about parameters
    @param num_params: number of parameters
    :param ub_param: array of upper bounds in order TODO: maybe dont do this here
    :param lb_param: array of lower bounds in order
    @return: negative log prior
    """
    prior_list = []
    for i in range(num_params):
        # needs to be within the bounds of the paramters
        prior_list.append(pypesto.objective.priors.get_parameter_prior_dict(i, 'uniform', [-10.0, 10.0]))
    # create the prior
    neg_log_prior = pypesto.objective.NegLogParameterPriors(prior_list)
    return neg_log_prior

def save_estimated_data(result_sampler: pypesto.Result) -> None:
    """
    save estimated data_sns_toolbox in hdf5
    @param result_sampler: sample results
    @return: None
    """
    result_sampler_dict = result_sampler.sample_result
    del_auto_corr = result_sampler_dict.pop('auto_correlation')
    del_message = result_sampler_dict.pop('message')
    del_eff_sample_size = result_sampler_dict.pop('effective_sample_size')
    dump_filename = datetime.datetime.now().strftime('%Y-%b-%d_%H-%M.t4a.h5')
    with h5py.File(dump_filename, 'w') as le_file:
        for key, value in result_sampler_dict.items():
            if value is not None:
                le_file.create_dataset(key, data=value)
    # Note: how to open file example - list(F['trace_x'])
    return None

def run_t4_estimation(vels, num_intervals, stim, goal) -> pypesto.Result:
    """
    starts the simulation for the rat hindlimb model_toolbox

    :param num_chains: number of chains
    :param n_iterations: number of iterations
    :return: sampler result
    """
    # Setup simulator
    path_config = Path("conf_t4_reduced.toml")

    random_seed = 7
    the_rng = np.random.default_rng(seed=random_seed)

    # Prior initialization
    num_chains, n_iterations, dim_full, params_to_use = lc.load_sampler_config(path_config)
    print(f"iterations: {n_iterations}; num_chains: {num_chains}")
    # _ = input('check OK, then hit Enter')

    lb_param, ub_param, test_params = lc.load_bounds(path_config)
    prior_term = easy_neg_log_prior(dim_full)  # Uniform priors, bounds set in Problem object

    # Sampler problem setup
    rest_params_random = np.array([the_rng.uniform(low=lb_param[i], high=ub_param[i]) for i in range(test_params.size)])
    rest_params = np.copy(test_params)
    rest_params[params_to_use] = np.copy(rest_params_random[params_to_use])

    likelihood = pypesto.Objective(easy_neg_log_likelihood, fun_args=(params_to_use, rest_params, vels, num_intervals, stim, goal))

    objective1 = pypesto.objective.AggregatedObjective([likelihood, prior_term])
    objective1.initialize()
    # NOTE: we are not using the prior
    problem = pypesto.Problem(objective=likelihood, lb=lb_param[params_to_use], ub=ub_param[params_to_use])

    print(lb_param)
    print(ub_param)
    print(params_to_use)
    print('Guessing...')
    print(test_params[params_to_use])
    debug_params = np.array([100, 500, 150, 250, 10])
    test = problem.objective(test_params[params_to_use])
    # test = problem.objective(debug_params)
    print(f'check that this number (bad params loss) is positive: {test}')
    _ = input('check OK, then hit Enter')

    # Run the actual sampler now
    with Pool(num_chains) as pool:
        print('Finding some decent initial starts... this may take a while')
        x0 = []
        start_idx = 0
        while len(x0) < num_chains:
            print(start_idx, len(x0))
            possible_guesses = [np.array([the_rng.uniform(low=lb_param[i], high=ub_param[i])
                                for i in params_to_use])
                                for _ in range(num_chains)]
            possible_guesses_evals = np.zeros(num_chains)
            for i in range(num_chains):
                possible_guesses_evals[i] = problem.objective(possible_guesses[i])
                # print('     ' + str(possible_guesses_evals[i]))
            # possible_guesses_evals = pool.map(problem.objective, possible_guesses)
            # a_guess_eval = problem.objective(a_guess)
            for guess_cost, guess in zip(possible_guesses_evals, possible_guesses):
                if guess_cost < 1e5:
                    x0.append(guess)
            start_idx += 1


        # x0 = [np.array([the_rng.uniform(low=lb_param[i], high=ub_param[i]) for i in range(dim_full)])
        #           for _ in range(num_chains)]

        print('finished problem setup')


        sampler = pypesto.sample.PoolAdaptiveParallelTemperingSampler(
            internal_sampler=pypesto.sample.AdaptiveMetropolisSampler(),  # used to be AdaptiveMetropolisSampler
            n_chains=num_chains)
        #     parallel_pool=pool
        # )
        print('finished sampler setup')

        print('starting sampler')
        try:
            result_sampler = pypesto.sample.sample(problem, n_iterations, sampler, x0=x0)
            # pypesto.sample.geweke_test(result=result_sampler)
            print('finished result sampler')
            save_estimated_data(result_sampler)  # TODO: comes from somewhere
            print('saved data_sns_toolbox')
            return result_sampler
        except Exception as e:
            logging.exception(e)
            dump_filename = datetime.datetime.now().strftime('%Y-%b-%d_%H-%M.error.dillpkl')
            dill.dump(sampler, open(dump_filename, "wb"))
            print(f'dumped result_sampler to: {dump_filename}')
            raise e

def main():
    stim_on_lr = gen_stimulus(1)
    num_intervals = 4
    vels = np.linspace(10, 180, num=num_intervals)
    # goal = np.linspace(1.0, 0.1, num=num_intervals)
    goal_x = np.linspace(0.0, 4.0, num=num_intervals)
    goal = (4 * np.exp(goal_x)) / (1 + np.exp(goal_x)) ** 2
    # dir = 'T4 Velocity/'
    run_t4_estimation(vels, num_intervals, stim_on_lr, goal)

if __name__ == '__main__':
    # logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
    #                     level=logging.INFO)
    main()
