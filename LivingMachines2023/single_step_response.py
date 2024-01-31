import numpy as np

from utilities import load_data, save_data
from motion_vision_networks import gen_single_column
from sns_toolbox.renderer import render

def get_step_response(t, model, data, inputs, filename, neg=True):
    model.reset()
    found = False
    for i in range(len(t)):
        if neg and inputs[i,:] == 0.0 and not found:
            model.V_last[6] = 0.0
            model.V[6] = 0.0 # TODO: Fix this
            found = True
        data[i, :] = model(inputs[i, :])
    data = data.transpose()

    inp = data[0,:]
    bp = data[1,:]
    lp = data[2,:]
    e = data[3,:]
    d_on = data[4,:]
    d_off = data[5,:]
    s_on = data[6,:]
    s_off = data[7,:]

    output = {'t': t, 'in': inp, 'bp': bp, 'lp': lp, 'e': e, 'd_on': d_on, 'd_off': d_off, 's_on': s_on, 's_off': s_off, 'input': inputs}

    save_data(output, 'Step Responses/%s.pc'%filename)

params = load_data('params_net_20230327.pc')

model, net = gen_single_column(params)
# render(net)

dt = params['dt']
t = np.arange(0,100, dt)

inputs_pos = np.ones([len(t), net.get_num_inputs()])
inputs_neg = np.ones([len(t), net.get_num_inputs()])
for i in range(len(t)):
    if t[i] > 50:
        inputs_neg[i] = 0.0

data_pos = np.zeros([len(t), net.get_num_outputs_actual()])
data_neg = np.zeros([len(t), net.get_num_outputs_actual()])

get_step_response(t, model, data_pos, inputs_pos, 'positive', neg=False)
get_step_response(t, model, data_neg, inputs_neg, 'negative')
