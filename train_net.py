import pickle
import time
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
import pandas as pd
import argparse
from evotorch import Problem, Solution
import evotorch.algorithms as alg
from evotorch.logging import PandasLogger
from motion_vision_net import SNSMotionVisionMerged, SNSMotionVisionOn
from motion_data import ClipDataset
from datetime import datetime
import matplotlib.pyplot as plt

def vel_to_state(vel):
    return 1 - ((vel-.1)/0.4) * (1-0.1)

def state_to_vel(state):
    vel = ((1 - state) / (1 - 0.1)) * 0.4 + 0.1
    return vel


def process_args(args):
    params = {}
    params['dt'] = float(args.dt)
    if args.dtype == 'float64':
        params['dtype'] = torch.float64
    elif args.dtype == 'float32':
        params['dtype'] = torch.float32
    else:
        params['dtype'] = torch.float16
    params['boundFile'] = args.bound_file
    params['numGenerations'] = int(args.num_generations)
    params['popSize'] = int(args.pop_size)
    params['algorithm'] = args.algorithm
    if args.num_workers == 'max':
        params['numWorkers'] = 'max'
    else:
        params['numWorkers'] = int(args.num_workers)
    params['std'] = float(args.std)
    params['lr'] = float(args.lr)
    params['parenthoodRatio'] = float(args.parenthood_ratio)
    params['tol'] = float(args.tol)
    params['batchSize'] = int(args.batch_size)
    params['shapeField'] = int(args.shape_field)
    params['trainData'] = ClipDataset('FlyWheelTrain')
    params['testData'] = ClipDataset('FlyWheelTest')
    now = datetime.now()
    params['dateTime'] = now.strftime('%Y-%m-%d_%H-%M-%S')
    params['dir'] = args.dir
    params['debug'] = args.debug
    bounds = pd.read_csv(params['boundFile'])
    params['boundsLower'] = torch.as_tensor(bounds['Lower Bound'], dtype=params['dtype'])
    params['boundsUpper'] = torch.as_tensor(bounds['Upper Bound'], dtype=params['dtype'])
    params['centerInit'] = torch.as_tensor(bounds['Start'], dtype=params['dtype'])
    params['compile'] = bool(args.compile)
    params['max'] = 2.0*params['batchSize']#torch.finfo(params['dtype']).max
    if args.batch_validate == 'full':
        params['batchValidate'] = args.batch_validate
    else:
        params['batchValidate'] = int(args.batch_validate)
    params['network'] = args.network

    return params

def vec_to_dict(x, opt_params):
    freq_fast = 1000/(2*3*torch.pi)
    if opt_params['network'] == 'On':
        params = nn.ParameterDict({
            'reversalEx': nn.Parameter(torch.tensor([5.0])), 'reversalIn': nn.Parameter(torch.tensor([-2.0])),
            'reversalMod': nn.Parameter(torch.tensor([-0.1])), 'freqFast': nn.Parameter(torch.tensor([freq_fast])),
            'ampCenBO': nn.Parameter(x[0]), 'stdCenBO': nn.Parameter(x[1]), 'ampSurBO': nn.Parameter(x[2]),
            'stdSurBO': nn.Parameter(x[3]), 'freqBOFast': nn.Parameter(x[4]), 'freqBOSlow': nn.Parameter(x[5]),
            'ampCenLO': nn.Parameter(x[6]), 'stdCenLO': nn.Parameter(x[7]), 'ampSurLO': nn.Parameter(x[8]),
            'stdSurLO': nn.Parameter(x[9]), 'freqLO': nn.Parameter(x[10]), 'conductanceLOEO': nn.Parameter(x[11]),
            'freqEO': nn.Parameter(x[12]), 'biasEO': nn.Parameter(x[13]), 'conductanceBODO': nn.Parameter(x[14]),
            'freqDO': nn.Parameter(x[15]), 'biasDO': nn.Parameter(x[16]), 'conductanceDOSO': nn.Parameter(x[17]),
            'freqSO': nn.Parameter(x[18]), 'biasSO': nn.Parameter(x[19]), 'conductanceEOOn': nn.Parameter(x[20]),
            'conductanceDOOn': nn.Parameter(x[21]), 'conductanceSOOn': nn.Parameter(x[22]), 'freqOn': nn.Parameter(x[23]),
            'biasOn': nn.Parameter(x[24]),  'gainHorizontal': nn.Parameter(x[25])
        })
    return params

def run_net(solution, opt_params, data, state_labels, vel_labels):
    x = solution.values.clone().detach()
    x = torch.clamp(x, opt_params['boundsLower'], opt_params['boundsUpper'])
    with torch.no_grad():
        if opt_params['debug'] == 'True':
            print('Convert parameter vector')
        # Convert parameter vector to dictionary
        params = vec_to_dict(x, opt_params)

        # Build model
        if opt_params['debug'] == 'True':
            print('Building net')
        if opt_params['network'] == 'On':
            model = SNSMotionVisionOn(opt_params['dt'], [24, 64], opt_params['shapeField'],
                                      params=params, dtype=opt_params['dtype'], device='cpu')
        if opt_params['compile']:
            if opt_params['debug'] == 'True':
                print('Compiling net')
            model = torch.jit.script(model)
            model.eval()
            model = torch.jit.freeze(model)
            model = torch.compile(model)

        # Training loop
        num_trials = data.shape[0]
        num_frames = data.shape[1]
        num_steps_inner = 13
        num_steps = num_steps_inner * num_frames

        out_state = torch.zeros([num_steps, 2])
        error = torch.zeros(num_trials)
        prediction = torch.zeros(num_trials)
        if opt_params['debug'] == 'True':
            print('Start loop')
        for trial in range(num_trials):
            for frame in range(num_frames):
                image = data[trial, frame, :, :]
                for step in range(num_steps_inner):
                    if opt_params['debug'] == 'True':
                        print('Trial ' + str(trial) + ' Frame ' + str(frame) + ' Step ' + str(
                            frame * num_steps_inner + step))
                    if frame == 0 and step == 0:
                        out_state[frame * num_steps_inner + step, :] = model(image, True)
                    else:
                        out_state[frame * num_steps_inner + step, :] = model(image, True)
            if opt_params['debug'] == 'True':
                print('Trial error')
            avg_val = torch.mean(out_state[:, 0])
            error[trial] = mse_loss(avg_val, state_labels[trial])
            prediction[trial] = 0.05 * torch.round(state_to_vel(avg_val)/0.05)
        accuracy = torch.sum(prediction == vel_labels)/num_trials

        return torch.sum(error), accuracy

class OptimizeMotionVision(Problem):
    def __init__(self, opt_params):
        if opt_params['debug'] == 'True':
            print('Initializing problem')
        super().__init__(objective_sense='min',
                         solution_length=len(opt_params['boundsLower']),
                         initial_bounds=(opt_params['boundsLower'], opt_params['boundsUpper']),
                         num_actors=opt_params['numWorkers'])
        self.opt_params = opt_params

    def _evaluate(self, solution: Solution):
        if self.opt_params['debug']:
            print('Evaluating')


        # Load training data
        if self.opt_params['debug']:
            print('Loading training data')
        training_dataloader = DataLoader(self.opt_params['trainData'], batch_size=self.opt_params['batchSize'], shuffle=True)
        training_data, training_labels = next(iter(training_dataloader))
        state_labels = vel_to_state(training_labels)

        error, _ = run_net(solution, self.opt_params, training_data, state_labels, training_labels)
        # if torch.isnan(error):
        #     error = torch.tensor([torch.finfo(self.opt_params['dtype']).max])
        error = torch.nan_to_num(error, nan=self.opt_params['max'])

        if self.opt_params['debug']:
            print('Individual error')
        solution.set_evals(error)

def get_solver(opt_params, problem):
    if opt_params['algorithm'] == 'CES':
        solver = alg.CEM(problem, popsize=opt_params['popSize'], parenthood_ratio=opt_params['parenthoodRatio'],
                         stdev_init=opt_params['std'], center_init=opt_params['centerInit'])
    elif opt_params['algorithm'] == 'CMA-ES':
        solver = alg.CMAES(problem, stdev_init=opt_params['std'], popsize=opt_params['popSize'], center_init=opt_params['centerInit'])
    elif opt_params['algorithm'] == 'PGPE':
        solver = alg.PGPE(problem, popsize=opt_params['popSize'], center_learning_rate=opt_params['lr'],
                          stdev_learning_rate=opt_params['lr'], stdev_init=opt_params['std'], center_init=opt_params['centerInit'])
    elif opt_params['algorithm'] == 'SNES':
        solver = alg.SNES(problem, stdev_init=opt_params['std'], popsize=opt_params['popSize'], center_init=opt_params['centerInit'])
    else:
        solver = alg.XNES(problem, stdev_init=opt_params['std'], popsize=opt_params['popSize'], center_init=opt_params['centerInit'])

    if opt_params['debug'] == 'True':
        print('Solver created: '+opt_params['algorithm'])
    return solver

def validate(solution, opt_params):
    # Load training data
    if opt_params['debug'] == 'True':
        print('Loading training data')
    if opt_params['batchValidate'] == 'full':
        num_val = len(opt_params['testData'])
    else:
        num_val = opt_params['batchValidate']
    test_dataloader = DataLoader(opt_params['testData'], batch_size=num_val, shuffle=True)
    test_data, test_labels = next(iter(test_dataloader))
    state_labels = vel_to_state(test_labels)

    _, accuracy = run_net(solution, opt_params, test_data, state_labels, test_labels)
    return accuracy

def main(opt_params):
    if opt_params['debug'] == 'True':
        print('Generate Problem')
    problem = OptimizeMotionVision(opt_params)
    if opt_params['debug'] == 'True':
        print('Generate Solver')
    solver = get_solver(opt_params, problem)
    if opt_params['debug'] == 'True':
        print('Generate Logger')
    logger = PandasLogger(solver)
    log_filename = opt_params['dir'] + 'log_' + opt_params['dateTime'] + '_' + opt_params['algorithm'] + '.p'
    param_filename = opt_params['dir'] + 'params_' + opt_params['dateTime'] + '_' + opt_params['algorithm'] + '.p'
    history_filename = opt_params['dir'] + 'history_' + opt_params['dateTime'] + '_' + opt_params['algorithm'] + '.p'
    i = 0
    best = 10000000
    accuracy = []
    print('Starting Optimization:')
    #start = time.time()
    running_history = None
    while i < opt_params['numGenerations'] and best > opt_params['tol']:
        # print(func + ' ' + algorithm + ' ' + str(i))
        print('Algorithm: ' + opt_params['algorithm'] + ' Generation: ' + str(i))
        start = time.time()
        solver.step()
        #end = time.time()
        #print('Eval time: %0.2f s'&(end-start))
        population = solver.population.values
        evals = solver.population.evals
        history = torch.cat((population, evals), dim=1)
        data = logger.to_dataframe()
        data.to_pickle(log_filename)
        solution = problem.status['best'].values
        pickle.dump(solution, open(param_filename, 'wb'))
        best = data['pop_best_eval'].iloc[-1]
        print('Fitness: ' + str(best))
        if i == 0:
            running_history = history
            # accuracy = [validate(problem.status['best'], opt_params)]
        else:
            running_history = torch.cat((running_history, history), dim=0)
            # accuracy.append(validate(problem.status['best'], opt_params))
        pickle.dump(running_history, open(history_filename, 'wb'))
        i += 1
        problem.kill_actors()
        end = time.time()
        print('Time: ' + str(end-start))
        #problem.kill_actors()
    #print('Time:')
    #print(time.time()-start)

    plt.figure()
    # plt.subplot(1,2,1)
    plt.plot(data['pop_best_eval'])
    plt.title('Error')
    plt.xlabel('Trial')
    # plt.ylabel('Error')
    # plt.subplot(1,2,2)
    # plt.plot(accuracy)
    # plt.title('Accuracy')
    # plt.xlabel('Trial')
    # plt.ylabel('Accuracy (%)')




if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Train the Network")
    parser.add_argument('--dt', nargs='?', default='2.56')
    parser.add_argument('--dtype', nargs='?', choices=['float64', 'float32', 'float16'], default='float32')
    parser.add_argument('--bound_file', nargs='?', default='bounds_on_20240215.csv')
    parser.add_argument('--num_generations', nargs='?', default='3')
    parser.add_argument('--pop_size', nargs='?', default='16')
    parser.add_argument('--algorithm', nargs='?', default='CMA-ES', choices=['CES', 'CMA-ES', 'PGPE', 'SNES', 'XNES'])
    parser.add_argument('--num_workers', nargs='?', default='max')
    parser.add_argument('--std', nargs='?', default='5')
    parser.add_argument('--lr', nargs='?', default='0.01')
    parser.add_argument('--parenthood_ratio', nargs='?', default='0.1')
    parser.add_argument('--tol', nargs='?', default='1e-3')
    parser.add_argument('--batch_size', nargs='?', default='4')
    parser.add_argument('--shape_field', nargs='?', default='5')
    parser.add_argument('--dir', nargs='?', default='Runs/On/')
    parser.add_argument('--debug', nargs='?', default='False')
    parser.add_argument('--compile', nargs='?', default='True')
    parser.add_argument('--batch_validate', nargs='?', default='1')
    parser.add_argument('--network', nargs='?', default='On', choices=['On'])
    args = parser.parse_args()
    opt_params = process_args(args)

    main(opt_params)

    plt.show()
