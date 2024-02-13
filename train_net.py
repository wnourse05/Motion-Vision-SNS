import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
import pandas as pd
import argparse
from evotorch import Problem, Solution
from motion_vision_net import SNSMotionVisionMerged

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
    params['parenthoodRatio'] = float(args.parentood_ratio)
    params['tol'] = float(args.tol)
    params['batchSize'] = int(args.batch_size)
    params['shapeField'] = int(args.shape_field)

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

class OptimizeMotionVision(Problem):
    def __init__(self, d, opt_params):
        bounds = pd.read_csv(opt_params['boundFile'])
        bounds_lower = torch.as_tensor(bounds['Lower Bound'], dtype=opt_params['dtype'])
        bounds_upper = torch.as_tensor(bounds['Upper Bound'], dtype=opt_params['dtype'])
        super().__init__(objective_sense='min',
                         solution_length=d,
                         initial_bounds=(bounds_lower, bounds_upper),
                         bounds=(bounds_lower, bounds_upper))
        self.opt_params = opt_params

    def _evaluate(self, solution: Solution):
        x = solution.values
        params = vec_to_dict(x)
        model = SNSMotionVisionMerged(self.opt_params['dt'], [24,64], self.opt_params['shapeField'])






if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Train the Network")
    parser.add_argument('--dt', nargs='?', default='2.56')
    parser.add_argument('--dtype', nargs='?', choices=['float64', 'float32', 'float16'], default='float32')
    parser.add_argument('--bound_file', nargs='?', default='bounds.csv')
    parser.add_argument('--num_generations', nargs='?', default='10000')
    parser.add_argument('--pop_size', nargs='?', default='100')
    parser.add_argument('--algorithm', nargs='?', default='CES', choices=['CES', 'CMA-ES', 'PGPE', 'SNES', 'XNES'])
    parser.add_argument('--num_workers', nargs='?', default='max')
    parser.add_argument('--std', nargs='?', default='5')
    parser.add_argument('--lr', nargs='?', default='0.01')
    parser.add_argument('--parenthood_ratio', nargs='?', default='0.1')
    parser.add_argument('--tol', nargs='?', default='1e-3')
    parser.add_argument('--batch_size', nargs='?', default='100')
    parser.add_argument('--shape_field', nargs='?', default='5')
    args = parser.parse_args()
    opt_params = process_args(args)

    print(opt_params)
