import pickle
import time
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
import pandas as pd
import argparse
from motion_vision_net import SNSMotionVisionMerged, SNSMotionVisionOn
from motion_data import ClipDataset
from datetime import datetime
import matplotlib.pyplot as plt

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
    params['numEpochs'] = int(args.num_epochs)
    params['lr'] = float(args.lr)
    params['shapeField'] = int(args.shape_field)
    params['logInterval'] = int(args.log_interval)
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
    params['device'] = str(args.device)

    return params

def vel_to_state(vel):
    return 1 - ((vel-.1)/0.4) * (1-0.1)

def state_to_vel(state):
    vel = ((1 - state) / (1 - 0.1)) * 0.4 + 0.1
    return vel

def run_sample(sample, net: nn.Module):
    (state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass, state_enhance_on,
     state_direct_on, state_suppress_on, state_cw_on, state_ccw_on, state_horizontal, avg) = net.init()

    net.zero_grad()

    for i in range(sample.shape[0]):
        (state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass, state_enhance_on,
         state_direct_on, state_suppress_on, state_cw_on, state_ccw_on, state_horizontal, avg) = net(sample[i,:,:],
                                                                                                     state_input,
                                                                                                     state_bo_input,
                                                                                                     state_bo_fast,
                                                                                                     state_bo_slow,
                                                                                                     state_bo_output,
                                                                                                     state_lowpass,
                                                                                                     state_enhance_on,
                                                                                                     state_direct_on,
                                                                                                     state_suppress_on,
                                                                                                     state_cw_on,
                                                                                                     state_ccw_on,
                                                                                                     state_horizontal,
                                                                                                     avg)

        return avg, net

def validate(net, loss_fn, testing_loader):
    with torch.no_grad():
        loss_history = torch.zeros(len(testing_loader))
        for i, data in enumerate(testing_loader):
            # Get data
            frames, target = data
            target = vel_to_state(target)

            # Simulate the network
            avg, net = run_sample(frames, net)

            # Compute loss and gradients
            loss = loss_fn(avg, target)
            loss_history[i] = loss.item()
        mean_loss = torch.mean(loss_history)
    return mean_loss

def run_epoch(index, net, loss_fn, optimizer, training_loader, testing_loader, log_interval):
    loss_history = torch.zeros(len(training_loader))
    for i, data in enumerate(training_loader):
        # Get data
        frames, target = data
        target = vel_to_state(target)

        # Reset gradients
        net.zero_grad()

        # Simulate the network
        avg, net = run_sample(frames, net)

        # Compute loss and gradients
        loss = loss_fn(avg, target)
        loss.backward()

        # Adjust parameters
        optimizer.step()
        if i % log_interval == 0:
            print('Epoch: %i Sample %i/%i Loss: %.4f'%(index, i+1, len(training_loader), loss.item()))
        loss_history[i] = loss.item()
    epoch_mean_loss = torch.mean(loss_history)
    test_loss = validate(net, loss_fn, testing_loader)
    print('Epoch: %i Mean Loss: %.4f'%(index, epoch_mean_loss.item()))
    print('Validation Loss: %.4f'%test_loss.item())

    torch.save({'epoch': index,
                'netStateDict': net.state_dict(),
                'optimizerStateDict': optimizer.state_dict(),
                }, 'checkpoints')
    return loss_history, epoch_mean_loss, test_loss, net, optimizer

def train(params):
    now = datetime.now()
    t = now.strftime('%Y-%m-%d_%H-%M-%S')
    data_training = ClipDataset('FlyWheelTrain')
    loader_training = DataLoader(data_training, shuffle=True)
    data_test = ClipDataset('FlyWheelTest')
    loader_testing = DataLoader(data_test, shuffle=True)

    loss_fn = nn.MSELoss()

    net = SNSMotionVisionOn(params['dt'], [24, 64], params['shapeField'], dtype=params['dtype'],
                            device=params['device'])

    optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'])

    epoch_loss_history = torch.zeros(params['numEpochs'])
    test_loss_history = torch.zeros(params['numEpochs'])

    for i in range(params['numEpochs']):
        epoch_loss, epoch_mean_loss, test_loss, net, optimizer = run_epoch(i, net, loss_fn, optimizer, loader_training,
                                                                           loader_testing, params['logInterval'])
        if i == 0:
            loss_history = epoch_loss
        else:
            loss_history = torch.cat((loss_history, epoch_loss))
        epoch_loss_history[i] = epoch_mean_loss
        test_loss_history[i] = test_loss


        filename = 'checkpoints/' + t + '_' + str(i) + '.pt'
        torch.save({'epoch': i,
                    'netStateDict': net.state_dict(),
                    'optimizerStateDict': optimizer.state_dict(),
                    'lossHistory': loss_history,
                    'epochLossHistory': epoch_loss_history,
                    'testLossHistory': test_loss_history
                    }, filename)
    return loss_history, epoch_loss_history, test_loss_history

def main(args):
    params = process_args(args)

    loss_history, epoch_loss_history, test_loss_history = train(params)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(loss_history)
    plt.title('Raw Loss')
    plt.subplot(3, 1, 2)
    plt.plot(epoch_loss_history)
    plt.title('Mean Loss Per Epoch')
    plt.subplot(3, 1, 3)
    plt.plot(test_loss_history)
    plt.title('Validation Loss')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Network")
    parser.add_argument('--dt', nargs='?', default='2.56')
    parser.add_argument('--dtype', nargs='?', choices=['float64', 'float32', 'float16'], default='float32')
    parser.add_argument('--bound_file', nargs='?', default='bounds_on_20240215.csv')
    parser.add_argument('--num_epochs', nargs='?', default='100')
    parser.add_argument('--lr', nargs='?', default='0.001')
    parser.add_argument('--shape_field', nargs='?', default='5')
    parser.add_argument('--log_interval', nargs='?', default='100')
    parser.add_argument('--compile', nargs='?', default='True')
    parser.add_argument('--network', nargs='?', default='On', choices=['On'])
    parser.add_argument('--device', nargs='?', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    main(args)