import numpy as np
from sns_toolbox.networks import Network
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse
from utilities import activity_range, reversal_in, reversal_ex, dt, backend
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# c_slow = 10 * c_fast
t = np.arange(0, 1000, dt)
def create_bandpass_w_input(conductance_retina, reversal_retina, c_slow, c_fast):
    net = Network()
    neuron_fast = NonSpikingNeuron(membrane_capacitance=c_fast, membrane_conductance=1.0, bias=activity_range)
    neuron_slow = NonSpikingNeuron(membrane_capacitance=c_slow, membrane_conductance=1.0, bias=activity_range)
    neuron_retina = NonSpikingNeuron(membrane_capacitance=c_fast, membrane_conductance=1.0)

    conductance_in = -activity_range/reversal_in
    conductance_ex = conductance_in*(reversal_in-activity_range)/(activity_range-reversal_ex)
    synapse_in = NonSpikingSynapse(max_conductance=conductance_in, reversal_potential=reversal_in, e_lo=0.0, e_hi=activity_range)
    synapse_ex = NonSpikingSynapse(max_conductance=conductance_ex, reversal_potential=reversal_ex, e_lo=0.0, e_hi=activity_range)

    synapse_retina = NonSpikingSynapse(max_conductance=conductance_retina, reversal_potential=reversal_retina, e_lo=0.0, e_hi=activity_range)

    net.add_neuron(neuron_retina, name='Retina')
    net.add_neuron(neuron_fast, name='a', initial_value=activity_range)
    net.add_neuron(neuron_fast, name='b')
    net.add_neuron(neuron_slow, name='c')
    net.add_neuron(neuron_fast, name='d', initial_value=activity_range)

    # net.add_connection(synapse_retina, 'Retina', 'a')
    net.add_connection(synapse_in, 'a', 'b')
    net.add_connection(synapse_in, 'a', 'c')
    net.add_connection(synapse_in, 'b', 'd')
    net.add_connection(synapse_ex, 'c', 'd')

    net.add_input('a')
    net.add_output('d')

    model = net.compile(dt=dt, backend=backend)

    return model

def run_net(model, stim):
    data = np.zeros_like(t)
    model.reset()
    for i in range(len(t)):
        data[i] = model([stim[i]])

    return data

def step_response(model):
    stim = np.ones_like(t)-2
    data = run_net(model, stim)
    return data

def impulse_response(model):
    stim = np.zeros_like(t)
    stim[0] = -1.0
    data = run_net(model, stim)
    return data

def cutoff_calc(capacitance):
    cond_scaled = 1.0*1e-6
    cap_scaled = capacitance*1e-9
    # print(cond_scaled)
    # print(cap_scaled)
    cutoff = cond_scaled/(cap_scaled*2*np.pi)
    # print(cutoff)
    return cutoff

# def cutoff_calc_z(capacitance):
#     alpha = dt / capacitance
#     cutoff = 1000 / (2 * np.pi * dt) * np.arccos(1 - (alpha ** 2 / (2 * (1 - alpha))))
#     return cutoff
# def cutoff_calc_wiki(capacitance):
#     cond_scaled = 1.0 * 1e-6
#     cap_scaled = capacitance * 1e-9
#     dt_scaled = dt/1000
#
#     alpha = dt_scaled/(cond_scaled*cap_scaled+dt_scaled)
#     cutoff = alpha/((1-alpha)*2*np.pi*dt_scaled)
#     return cutoff/(10**12)
#
# def cutoff_calc_comb(c_slow, c_fast):
#     g_scaled = 1.0 * 1e-6
#     c_slow_scaled = c_slow * 1e-9
#     c_fast_scaled = c_fast * 1e-9
#
#     a = c_slow_scaled*g_scaled
#     b = c_fast_scaled*g_scaled
#
#     num0 = np.sqrt(a**2 - 6*a*b + b**2) - a + b
#     num1 = np.sqrt(a**2 - 6*a*b + b**2) + a - b
#     den = 2*a*b
#     return num0/den, num1/den

def freq_response(data, c_slow, c_fast):
    N = len(data)   # number of sample pts
    T = dt/1000

    yf = rfft(data)
    xf = rfftfreq(N, T)#[:N//2]
    plt.plot(xf, 20*np.log10(np.abs(yf/np.max(yf))), label='Frequency Response')
    plt.axhline(20*np.log10(1/np.sqrt(2)), color='black', linestyle='--', label='-3 dB')
    plt.axvline(cutoff_calc(c_fast), color='orange', label='C_fast')
    plt.axvline(cutoff_calc(c_slow), color='green', label='C_slow')

    plt.xscale('log')
    # plt.title('Frequency Response')
    plt.legend()
    # plt.yscale('log')
    plt.grid()

c_slow = np.linspace(5,100, num=3)
c_fast = np.linspace(1,50, num=3)

for slow in c_slow:
    for fast in c_fast:
        if slow > fast:
            title = 'C_fast: {}, C_slow: {}'.format(fast, slow)
            file_prefix = '{}_{}_'.format(fast, slow)
            print(title)

            test_model = create_bandpass_w_input(1.0, reversal_in, slow, fast)
            data_step = step_response(test_model)
            data_imp = impulse_response(test_model)

            fig = plt.figure()
            plt.plot(t, data_step)
            plt.title(title)
            plt.xlabel('t (ms)')
            plt.ylabel('Step Response')
            fig.savefig('Step Responses/png/'+file_prefix+'step.png')
            fig.savefig('Step Responses/svg/' + file_prefix + 'step.svg')
            plt.close(fig)

            imp_cond = -(data_imp-1)

            fig1 = plt.figure()
            plt.plot(t, data_imp)
            plt.title(title)
            plt.xlabel('t (ms)')
            plt.ylabel('Impulse Response')
            fig1.savefig('Impulse Responses/png/' + file_prefix + 'imp.png')
            fig1.savefig('Impulse Responses/svg/' + file_prefix + 'imp.svg')
            plt.close(fig1)

            fig2 = plt.figure()
            plt.xlim([1,1000])
            freq_response(imp_cond, slow, fast)
            plt.title(title)
            plt.xlabel('Hz')
            plt.ylabel('Frequency Response')
            fig2.savefig('Frequency Responses/png/' + file_prefix + 'freq.png')
            fig2.savefig('Frequency Responses/svg/' + file_prefix + 'freq.svg')
            # plt.close(fig2)
            # plt.xlim([10,1000])

plt.show()
