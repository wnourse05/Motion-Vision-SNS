# from torch.multiprocessing import Pool, set_start_method
import os
os.environ['MKL_NUM_THREADS'] = "4"

import numpy as np
import torch
import time
from tqdm import tqdm
import pandas as pd
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

from utilities import cutoff_fastest, calc_cap_from_cutoff, load_data
from motion_vision_networks import gen_emd_on_mcmc

def test_emd(model, stim, device):
    model.reset()
    size = 7*7
    start = 3*7

    num_samples = np.shape(stim)[0]

    data = torch.zeros([num_samples, 2*size], device=device)

    for i in (range(num_samples)):
        data[i,:] = model(stim[i,:])

    data = data.to('cpu')
    data = data.transpose(0,1)
    a = data[:size, :]
    b = data[size:, :]

    a_row = a[start:start+7,:]
    a_single = a_row[3,:]
    b_row = b[start:start+7,:]
    b_single = b_row[3,:]

    a_peak = torch.max(a_single[int(num_samples / 2):])
    b_peak = torch.max(b_single[int(num_samples / 2):])
    ratio = a_peak/b_peak
    # if np.isnan(a_peak):
    #     raise ValueError('Nan')

    return a_peak, b_peak, ratio

def freq_response_emd(params, dt, freqs, stims, device):
    num_freqs = len(freqs)
    a_peaks = np.zeros_like(freqs)
    b_peaks = np.zeros_like(freqs)
    ratios = np.zeros_like(freqs)
    model, _ = gen_emd_on_mcmc(params, dt, device)
    for i in (range(num_freqs)):
        # print('Sample %i/%i: %f Hz'%(i+1, num_freqs, freqs[i]))
        a_peaks[i], b_peaks[i], ratios[i] = test_emd(model, stims[i], device)

    return a_peaks, b_peaks, ratios

def cost_function(a_peaks, b_peaks, ratios):
    peak_range = a_peaks[0] - a_peaks[1]
    peak_sums = np.sum(a_peaks)
    ratios_shifted = ratios - 1
    ratios_sum = np.sum(ratios_shifted)
    w0 = 1
    w1 = w0
    w2 = w0

    cost = 1/(w0*peak_range + w1*peak_sums + w2*ratios_sum)
    if np.isnan(cost):
        cost = 1e5

    return cost
def evaluate(params, dt, freqs, stims, device):
    a_peaks, b_peaks, ratios = freq_response_emd(params, dt, freqs, stims, device)
    cost = cost_function(a_peaks, b_peaks, ratios)

    return cost

def easy_neg_log_likelihood(sample_params: np.ndarray, actual_indices: list, rest_params: np.ndarray, dt, freqs, stims, device) -> Union[float, np.ndarray]:

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

    the_cost = evaluate(real_params, dt, freqs, stims, device)

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
    dump_filename = datetime.datetime.now().strftime('%Y-%b-%d_%H-%M.rf_hindlimb.h5')
    with h5py.File(dump_filename, 'w') as le_file:
        for key, value in result_sampler_dict.items():
            if value is not None:
                le_file.create_dataset(key, data=value)
    # Note: how to open file example - list(F['trace_x'])
    return None

def run_t4_estimation(dt, freqs, stims, device) -> pypesto.Result:
    """
    starts the simulation for the rat hindlimb model

    :param num_chains: number of chains
    :param n_iterations: number of iterations
    :return: sampler result
    """
    # Setup simulator
    path_config = Path("conf_t4.toml")

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

    likelihood = pypesto.Objective(easy_neg_log_likelihood, fun_args=(params_to_use, rest_params, dt, freqs, stims, device))

    objective1 = pypesto.objective.AggregatedObjective([likelihood, prior_term])
    objective1.initialize()
    # NOTE: we are not using the prior
    problem = pypesto.Problem(objective=likelihood, lb=lb_param[params_to_use], ub=ub_param[params_to_use])

    print(lb_param)
    print(ub_param)
    print(params_to_use)
    test = problem.objective(test_params[params_to_use])
    print(f'check that this number (bad params loss) is positive: {test}')
    _ = input('check OK, then hit Enter')

    # Run the actual sampler now
    # with Pool(num_chains) as pool:
    print('Finding some decent initial starts... this may take a while')
    x0 = []
    start_idx = 0
    while len(x0) < num_chains:
        print(start_idx)
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


    sampler = pypesto.sample.AdaptiveParallelTemperingSampler(
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
    device = 'cuda'
    stim_params = load_data('mcmc_stims.p')
    dt = stim_params['dt']
    freqs = stim_params['freqs']
    stims = stim_params['stims']
    stims[0] = stims[0].to(device)
    stims[1] = stims[1].to(device)
    # if device == 'cuda':
    #     # try:
    #     set_start_method('spawn')
    #     # except RuntimeError:
    #     #     pass
    run_t4_estimation(dt, freqs, stims, device)


if __name__ == '__main__':
    # logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
    #                     level=logging.INFO)
    main()

#                   Retina          L1 low              L1 High         L3              Mi1             Mi9             CT1 On          T4            K_Mi1 K_Mi9 K_CT1 K_T4
# params = np.array([cutoff_fastest, cutoff_fastest/10, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, 0.5,  0.5,  0.5,  0.1])   # Good guess
# device = 'cuda'
# stim_params = load_data('mcmc_stims.p')
# dt = stim_params['dt']
# freqs = stim_params['freqs']
# stims = stim_params['stims']
# stims[0] = stims[0].to(device)
# stims[1] = stims[1].to(device)
# start_time = time.time()
# cost = evaluate(params, dt, freqs, stims, device)
# print(time.time()- start_time)
# print(cost)
