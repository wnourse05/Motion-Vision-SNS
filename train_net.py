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
from motion_vision_net import SNSMotionVisionMerged
from motion_data import ClipDataset
from datetime import datetime
import matplotlib.pyplot as plt

def vel_to_state(vel):
    return 1 - ((vel-.1)/0.4) * (1-0.1)

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
    params['debug'] = bool(args.debug)
    bounds = pd.read_csv(params['boundFile'])
    params['boundsLower'] = torch.as_tensor(bounds['Lower Bound'], dtype=params['dtype'])
    params['boundsUpper'] = torch.as_tensor(bounds['Upper Bound'], dtype=params['dtype'])
    params['compile'] = bool(args.compile)
    params['max'] = 2.0*params['batchSize']#torch.finfo(params['dtype']).max
    if args.batch_validate == 'full':
        params['batchValidate'] = args.batch_validate
    else:
        params['batchValidate'] = int(args.batch_validate)

    return params

def vec_to_dict(x):
    params = nn.ParameterDict({
        'reversalEx': nn.Parameter(x[0]), 'reversalIn': nn.Parameter(x[1]), 'reversalMod': nn.Parameter(x[2]),
        'freqFast': nn.Parameter(x[3]), 'ampCenBO': nn.Parameter(x[4]), 'stdCenBO': nn.Parameter(x[5]),
        'ampSurBO': nn.Parameter(x[6]), 'stdSurBO': nn.Parameter(x[7]), 'freqBOFast': nn.Parameter(x[8]),
        'freqBOSlow': nn.Parameter(x[9]), 'ampCenL': nn.Parameter(x[10]), 'stdCenL': nn.Parameter(x[11]),
        'ampSurL': nn.Parameter(x[12]), 'stdSurL': nn.Parameter(x[13]), 'freqL': nn.Parameter(x[14]),
        'ampCenBF': nn.Parameter(x[15]), 'stdCenBF': nn.Parameter(x[16]), 'ampSurBF': nn.Parameter(x[17]),
        'stdSurBF': nn.Parameter(x[18]), 'freqBFFast': nn.Parameter(x[19]), 'freqBFSlow': nn.Parameter(x[20]),
        'conductanceLEO': nn.Parameter(x[21]), 'freqEO': nn.Parameter(x[22]), 'biasEO': nn.Parameter(x[23]),
        'conductanceBODO': nn.Parameter(x[24]), 'freqDO': nn.Parameter(x[25]), 'biasDO': nn.Parameter(x[26]),
        'conductanceDOSO': nn.Parameter(x[27]), 'freqSO': nn.Parameter(x[28]), 'biasSO': nn.Parameter(x[29]),
        'conductanceLEF': nn.Parameter(x[30]), 'freqEF': nn.Parameter(x[31]), 'biasEF': nn.Parameter(x[32]),
        'conductanceBFDF': nn.Parameter(x[33]), 'freqDF': nn.Parameter(x[34]), 'biasDF': nn.Parameter(x[35]),
        'conductanceDFSF': nn.Parameter(x[36]), 'freqSF': nn.Parameter(x[37]), 'biasSF': nn.Parameter(x[38]),
        'conductanceEOOn': nn.Parameter(x[39]), 'conductanceDOOn': nn.Parameter(x[40]),
        'conductanceSOOn': nn.Parameter(x[41]), 'conductanceEFOff': nn.Parameter(x[42]),
        'conductanceDFOff': nn.Parameter(x[43]), 'conductanceSFOff': nn.Parameter(x[44]), 'freqOn': nn.Parameter(x[45]),
        'biasOn': nn.Parameter(x[46]), 'freqOff': nn.Parameter(x[47]), 'biasOff': nn.Parameter(x[48]),
        'gainHorizontal': nn.Parameter(x[49]), 'freqHorizontal': nn.Parameter(x[50]),
        'biasHorizontal': nn.Parameter(x[51]),
    })
    return params

def run_net(solution, opt_params, data, labels):
    x = solution.values.clone().detach()
    x = torch.clamp(x, opt_params['boundsLower'], opt_params['boundsUpper'])
    with torch.no_grad():
        if opt_params['debug']:
            print('Convert parameter vector')
        # Convert parameter vector to dictionary
        params = vec_to_dict(x)

        # Build model
        if opt_params['debug']:
            print('Building net')
        model = SNSMotionVisionMerged(opt_params['dt'], [24, 64], opt_params['shapeField'],
                                      params=params, dtype=opt_params['dtype'], device='cpu')
        if opt_params['compile']:
            if opt_params['debug']:
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
        if opt_params['debug']:
            print('Start loop')
        for trial in range(num_trials):
            for frame in range(num_frames):
                image = data[trial, frame, :, :]
                for step in range(num_steps_inner):
                    if opt_params['debug']:
                        print('Trial ' + str(trial) + ' Frame ' + str(frame) + ' Step ' + str(
                            frame * num_steps_inner + step))
                    if frame == 0 and step == 0:
                        out_state[frame * num_steps_inner + step, :] = model(image, True)
                    else:
                        out_state[frame * num_steps_inner + step, :] = model(image, True)
            if opt_params['debug']:
                print('Trial error')
            avg_val = torch.mean(out_state[:, 0])
            error[trial] = mse_loss(avg_val, labels[trial])
        return torch.sum(error)

class OptimizeMotionVision(Problem):
    def __init__(self, opt_params):
        if opt_params['debug']:
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
        training_labels = vel_to_state(training_labels)

        error = run_net(solution, self.opt_params, training_data, training_labels)
        # if torch.isnan(error):
        #     error = torch.tensor([torch.finfo(self.opt_params['dtype']).max])
        error = torch.nan_to_num(error, nan=self.opt_params['max'])

        if self.opt_params['debug']:
            print('Individual error')
        solution.set_evals(error)

def get_solver(opt_params, problem):
    if opt_params['algorithm'] == 'CES':
        solver = alg.CEM(problem, popsize=opt_params['popSize'], parenthood_ratio=opt_params['parenthoodRatio'],
                         stdev_init=opt_params['std'])
    elif opt_params['algorithm'] == 'CMA-ES':
        solver = alg.CMAES(problem, stdev_init=opt_params['std'], popsize=opt_params['popSize'])
    elif opt_params['algorithm'] == 'PGPE':
        solver = alg.PGPE(problem, popsize=opt_params['popSize'], center_learning_rate=opt_params['lr'],
                          stdev_learning_rate=opt_params['lr'], stdev_init=opt_params['std']),
    elif opt_params['algorithm'] == 'SNES':
        solver = alg.SNES(problem, stdev_init=opt_params['std'], popsize=opt_params['popSize'])
    else:
        solver = alg.XNES(problem, stdev_init=opt_params['std'], popsize=opt_params['popSize'])

    if opt_params['debug']:
        print('Solver created: '+opt_params['algorithm'])
    return solver

def validate(solution, opt_params):
    # Load training data
    if opt_params['debug']:
        print('Loading training data')
    if opt_params['batchValidate'] == 'full':
        num_val = len(opt_params['testData'])
    else:
        num_val = opt_params['batchValidate']
    test_dataloader = DataLoader(opt_params['testData'], batch_size=num_val, shuffle=True)
    test_data, test_labels = next(iter(test_dataloader))
    test_labels = vel_to_state(test_labels)

    error = run_net(solution, opt_params, test_data, test_labels)
    return error

def main(opt_params):
    if opt_params['debug']:
        print('Generate Problem')
    problem = OptimizeMotionVision(opt_params)
    if opt_params['debug']:
        print('Generate Solver')
    solver = get_solver(opt_params, problem)
    if opt_params['debug']:
        print('Generate Logger')
    logger = PandasLogger(solver)
    log_filename = opt_params['dir'] + 'log_' + opt_params['dateTime'] + '_' + opt_params['algorithm'] + '.p'
    param_filename = opt_params['dir'] + 'params_' + opt_params['dateTime'] + '_' + opt_params['algorithm'] + '.p'
    i = 0
    best = 10000000
    print('Starting Optimization:')
    #start = time.time()
    while i < opt_params['numGenerations'] and best > opt_params['tol']:
        # print(func + ' ' + algorithm + ' ' + str(i))
        print('Algorithm: ' + opt_params['algorithm'] + ' Generation: ' + str(i))
        start = time.time()
        solver.step()
        #end = time.time()
        #print('Eval time: %0.2f s'&(end-start))
        data = logger.to_dataframe()
        data.to_pickle(log_filename)
        solution = problem.status['best'].values
        pickle.dump(solution, open(param_filename, 'wb'))
        best = data['pop_best_eval'].iloc[-1]
        print('Fitness: ' + str(best))
        i += 1
        end = time.time()
        print('Time: ' + str(end-start))
        problem.kill_actors()
    #print('Time:')
    #print(time.time()-start)
    test_error = validate(problem.status['best'], opt_params)
    plt.figure()
    plt.plot(data['pop_best_eval'])
    plt.title('Error: ' + str(test_error.item()))
    plt.xlabel('Trial')
    plt.ylabel('Error')




if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Train the Network")
    parser.add_argument('--dt', nargs='?', default='2.56')
    parser.add_argument('--dtype', nargs='?', choices=['float64', 'float32', 'float16'], default='float32')
    parser.add_argument('--bound_file', nargs='?', default='bounds.csv')
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
    parser.add_argument('--dir', nargs='?', default='Runs/')
    parser.add_argument('--debug', nargs='?', default='False')
    parser.add_argument('--compile', nargs='?', default='True')
    parser.add_argument('--batch_validate', nargs='?', default='1')
    args = parser.parse_args()
    opt_params = process_args(args)

    main(opt_params)

    plt.show()
