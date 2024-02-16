from snstorch import modules as m
import torch
import torch.nn as nn
import numpy as np
import torch.autograd.profiler as profiler
from typing import List
import torch.jit as jit
import math

def __calc_cap_from_cutoff__(cutoff):
    cap = 1000/(2*np.pi*cutoff)
    return cap

def __calc_2d_field__(amp_cen, amp_sur, std_cen, std_sur, shape_field, reversal_ex, reversal_in):
    axis = np.arange(-(5*(shape_field-1)/2), 5*((shape_field-1)/2+1), 5)
    conductance = torch.zeros([shape_field, shape_field])
    reversal = torch.zeros([shape_field, shape_field])
    for i in range(5):
        for j in range(5):
            target = -1 * (amp_cen * torch.exp(-(axis[i] ** 2 + axis[j] ** 2) / (2 * std_cen)) - amp_sur * torch.exp(
                -(axis[i] ** 2 + axis[j] ** 2) / (2 * std_sur)))
            if target >= 0:
                reversal[i,j] = reversal_ex
            else:
                reversal[i,j] = reversal_in
            conductance[i,j] = torch.clamp(target/(reversal[i,j]-target),0)
    return conductance, reversal



class SNSBandpass(nn.Module):
    def __init__(self, shape, params=None, device=None, dtype=torch.float32, generator=None):
        super().__init__()
        if device is None:
            device = 'cpu'
        self.params = nn.ParameterDict({
            'input_tau': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'input_leak': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'input_rest': nn.Parameter(torch.zeros(shape, dtype=dtype).to(device)),
            'input_bias': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'input_init': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'fast_tau': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'fast_leak': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'fast_rest': nn.Parameter(torch.zeros(shape, dtype=dtype).to(device)),
            'fast_bias': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'fast_init': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'slow_tau': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'slow_leak': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'slow_rest': nn.Parameter(torch.zeros(shape, dtype=dtype).to(device)),
            'slow_bias': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'slow_init': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'output_tau': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'output_leak': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'output_rest': nn.Parameter(torch.zeros(shape, dtype=dtype).to(device)),
            'output_bias': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'output_init': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'reversalIn': nn.Parameter((torch.tensor([-2.0], dtype=dtype)).to(device)),
            'reversalEx': nn.Parameter((torch.tensor([2.0], dtype=dtype)).to(device)),
        })
        if params is not None:
            self.params.update(params)
        self.shape = shape
        self.dtype = dtype
        self.device = device
        k = 1.0
        # k = 1.141306594552181
        activity_range = 1.0
        g_in = (-activity_range) / self.params['reversalIn']
        g_bd = (-k * activity_range) / (self.params['reversalIn'] + k * activity_range)
        g_cd = (g_bd * (self.params['reversalIn'] - activity_range)) / (activity_range - self.params['reversalEx'])
        params_input = nn.ParameterDict({
            'tau': self.params['input_tau'],
            'leak': self.params['input_leak'],
            'rest': self.params['input_rest'],
            'bias': self.params['input_bias'],
            'init': self.params['input_init']
        })
        self.input = m.NonSpikingLayer(shape, params=params_input, device=device, dtype=dtype)
        # self.state_input = torch.zeros(shape, dtype=dtype, device=device)+self.params['input_init']
        params_input_syn = nn.ParameterDict({
            'conductance': g_in,
            'reversal': self.params['reversalIn']
        })
        self.syn_input_fast = m.NonSpikingChemicalSynapseElementwise(params=params_input_syn, device=device, dtype=dtype)
        params_fast = nn.ParameterDict({
            'tau': self.params['fast_tau'],
            'leak': self.params['fast_leak'],
            'rest': self.params['fast_rest'],
            'bias': self.params['fast_bias'],
            'init': self.params['fast_init']
        })
        self.fast = m.NonSpikingLayer(shape, params=params_fast, device=device, dtype=dtype)
        # self.state_fast = torch.zeros(shape, dtype=dtype, device=device) + self.params['fast_init']
        self.syn_input_slow = m.NonSpikingChemicalSynapseElementwise(params=params_input_syn, device=device, dtype=dtype)
        params_slow = nn.ParameterDict({
            'tau': self.params['slow_tau'],
            'leak': self.params['slow_leak'],
            'rest': self.params['slow_rest'],
            'bias': self.params['slow_bias'],
            'init': self.params['slow_init']
        })
        self.slow = m.NonSpikingLayer(shape, params=params_slow, device=device, dtype=dtype)
        # self.state_slow = torch.zeros(shape, dtype=dtype, device=device) + self.params['slow_init']
        params_fast_syn_output = nn.ParameterDict({
            'conductance': g_bd,
            'reversal': self.params['reversalIn']
        })
        self.syn_fast_output = m.NonSpikingChemicalSynapseElementwise(params=params_fast_syn_output, device=device, dtype=dtype)
        params_slow_syn_output = nn.ParameterDict({
            'conductance': g_cd,
            'reversal': self.params['reversalEx']
        })
        self.syn_slow_output = m.NonSpikingChemicalSynapseElementwise(params=params_slow_syn_output, device=device, dtype=dtype)
        params_output = nn.ParameterDict({
            'tau': self.params['output_tau'],
            'leak': self.params['output_leak'],
            'rest': self.params['output_rest'],
            'bias': self.params['output_bias'],
            'init': self.params['output_init']
        })
        self.output = m.NonSpikingLayer(shape, params=params_output, device=device, dtype=dtype)
        # self.state_output = torch.zeros(shape, dtype=dtype, device=device) + self.params['output_init']

    def forward(self, x, state_input, state_fast, state_slow, state_output):
        input2fast = self.syn_input_fast(state_input, state_fast)
        input2slow = self.syn_input_slow(state_input,  state_slow)
        fast2out = self.syn_fast_output(state_fast, state_output)
        slow2out = self.syn_slow_output(state_slow, state_output)

        state_input = self.input(x, state_input)
        state_fast = self.fast(input2fast, state_fast)
        state_slow = self.slow(input2slow, state_slow)
        state_output = self.output(fast2out+slow2out, state_output)

        return state_input, state_fast, state_slow, state_output

    @jit.export
    def init(self):
        state_input =  torch.zeros(self.shape, dtype=self.dtype, device=self.device) + self.params['input_init']
        state_fast =   torch.zeros(self.shape, dtype=self.dtype, device=self.device) + self.params['fast_init']
        state_slow =   torch.zeros(self.shape, dtype=self.dtype, device=self.device) + self.params['slow_init']
        state_output = torch.zeros(self.shape, dtype=self.dtype, device=self.device) + self.params['output_init']
        return state_input, state_fast, state_slow, state_output

class OnPathway(nn.Module):
    def __init__(self, dt, shape_input, shape_field, params=None, device=None, dtype=torch.float32, generator=None):
        super().__init__()
        if device is None:
            device = 'cpu'

        self.params = nn.ParameterDict({
            'reversalEx': nn.Parameter(torch.tensor([2.0], dtype=dtype).to(device)),
            'reversalIn': nn.Parameter(torch.tensor([-2.0], dtype=dtype).to(device)),
            'reversalMod': nn.Parameter(torch.tensor([0.0], dtype=dtype).to(device)),
            'freqFast': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'ampCenBO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdCenBO': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ampSurBO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdSurBO': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqBOFast': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'freqBOSlow': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'ampCenLO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdCenLO': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ampSurLO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdSurLO': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqLO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'conductanceLOEO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqEO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasEO': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceBODO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqDO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasDO': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceDOSO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqSO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasSO': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceEOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceDOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceSOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasOn': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
        })
        if params is not None:
            self.params.update(params)
        # Network

        # Retina
        tau_fast = dt/__calc_cap_from_cutoff__(self.params['freqFast'].data)
        nrn_input_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_fast.data + torch.zeros(shape_input, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_input, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_input, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(torch.zeros(shape_input, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_input, dtype=dtype).to(device), requires_grad=False)
        })
        self.input = m.NonSpikingLayer(shape_input, params=nrn_input_params, device=device, dtype=dtype)
        # self.state_input = torch.zeros(shape_input, dtype=dtype, device=device)+nrn_input_params['init']

        # Lamina
        shape_post_conv = [x - (shape_field - 1) for x in shape_input]
        # Bo
        conductance, reversal = __calc_2d_field__(self.params['ampCenBO'], self.params['ampSurBO'],
                                                  self.params['stdCenBO'], self.params['stdSurBO'], shape_field,
                                                  self.params['reversalEx'], self.params['reversalIn'])
        syn_in_bo_params = nn.ParameterDict({
            'conductance': nn.Parameter(conductance, requires_grad=False),
            'reversal': nn.Parameter(reversal, requires_grad=False)
        })
        self.syn_input_bandpass_on = m.NonSpikingChemicalSynapseConv(1,1,shape_field, conv_dim=2,
                                                                     params=syn_in_bo_params, device=device, dtype=dtype)
        tau_bo_fast = dt / __calc_cap_from_cutoff__(self.params['freqBOFast'].data)
        tau_bo_slow = dt / __calc_cap_from_cutoff__(self.params['freqBOSlow'].data)
        nrn_bo_params = nn.ParameterDict({
            'input_tau': nn.Parameter((tau_fast + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                      requires_grad=False),
            'input_leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'input_rest': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'input_bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'input_init': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'fast_tau': nn.Parameter((tau_bo_fast + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                     requires_grad=False),
            'fast_leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'fast_rest': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'fast_bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'fast_init': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'slow_tau': nn.Parameter((tau_bo_slow + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                     requires_grad=False),
            'slow_leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'slow_rest': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'slow_bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'slow_init': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'output_tau': nn.Parameter((tau_fast + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                       requires_grad=False),
            'output_leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'output_rest': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'output_bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'output_init': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'reversalIn': nn.Parameter((self.params['reversalIn'].clone().detach()).to(device), requires_grad=False),
            'reversalEx': nn.Parameter((self.params['reversalEx'].clone().detach()).to(device), requires_grad=False),
        })
        self.bandpass_on = SNSBandpass(shape_post_conv, params=nrn_bo_params, device=device, dtype=dtype)
        # L
        conductance, reversal = __calc_2d_field__(self.params['ampCenLO'], self.params['ampSurLO'],
                                                  self.params['stdCenLO'], self.params['stdSurLO'], shape_field,
                                                  self.params['reversalEx'], self.params['reversalIn'])
        syn_in_l_params = nn.ParameterDict({
            'conductance': nn.Parameter(conductance, requires_grad=False),
            'reversal': nn.Parameter(reversal, requires_grad=False)
        })
        self.syn_input_lowpass = m.NonSpikingChemicalSynapseConv(1, 1, shape_field, conv_dim=2,
                                                                 params=syn_in_l_params, device=device, dtype=dtype)
        tau_l = dt / __calc_cap_from_cutoff__(self.params['freqLO'].data)
        nrn_l_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_l + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'init': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
        })
        self.lowpass = m.NonSpikingLayer(shape_post_conv, params=nrn_l_params, device=device, dtype=dtype)
        # self.state_lowpass = torch.zeros(shape_post_conv, dtype=dtype, device=device)+nrn_l_params['init']

        # Medulla
        # On
        # EO
        syn_l_eo_params = nn.ParameterDict({
            'conductance': self.params['conductanceLOEO'],
            'reversal': self.params['reversalEx']
        })
        self.syn_lowpass_enhance_on = m.NonSpikingChemicalSynapseElementwise(params=syn_l_eo_params, device=device,
                                                                             dtype=dtype)
        tau_eo = dt / __calc_cap_from_cutoff__(self.params['freqEO'].data)
        nrn_eo_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_eo + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype, device=device).to(device),
                                 requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasEO'] + torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
        })
        self.enhance_on = m.NonSpikingLayer(shape_post_conv, params=nrn_eo_params, device=device, dtype=dtype)
        # self.state_enhance_on = torch.zeros(shape_post_conv, dtype=dtype, device=device)+nrn_eo_params['init']
        # DO
        syn_bo_do_params = nn.ParameterDict({
            'conductance': self.params['conductanceBODO'],
            'reversal': self.params['reversalIn']
        })
        self.syn_bandpass_on_direct_on = m.NonSpikingChemicalSynapseElementwise(params=syn_bo_do_params, device=device,
                                                                                dtype=dtype)
        tau_do = dt / __calc_cap_from_cutoff__(self.params['freqDO'].data)
        nrn_do_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_do + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            # 'bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasDO'] + torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
        })
        self.direct_on = m.NonSpikingLayer(shape_post_conv, params=nrn_do_params, device=device, dtype=dtype)
        # self.state_direct_on = torch.zeros(shape_post_conv, dtype=dtype, device=device)+nrn_do_params['init']
        # SO
        syn_do_so_params = nn.ParameterDict({
            'conductance': self.params['conductanceDOSO'],
            'reversal': self.params['reversalEx']
        })
        self.syn_direct_on_suppress_on = m.NonSpikingChemicalSynapseElementwise(params=syn_do_so_params, device=device,
                                                                                dtype=dtype)
        tau_so = dt / __calc_cap_from_cutoff__(self.params['freqSO'].data)
        nrn_so_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_so + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasSO'] + torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
        })
        self.suppress_on = m.NonSpikingLayer(shape_post_conv, params=nrn_so_params, device=device, dtype=dtype)
        # self.state_suppress_on = torch.zeros(shape_post_conv, dtype=dtype, device=device)+nrn_so_params['init']

        # Lobula
        shape_emd = [x - 2 for x in shape_post_conv]
        # On
        syn_do_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, self.params['reversalEx'], 0], [0, 0, 0]],
                                                  dtype=dtype, device=device), requires_grad=False),
        })
        self.syn_direct_on_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2,
                                                                params=syn_do_on_params, device=device, dtype=dtype)
        # CCW
        syn_eo_ccw_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['conductanceEOOn'], 0, 0], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['reversalMod'], 0, 0], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
        })
        self.syn_enhance_on_ccw_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2,
                                                                      params=syn_eo_ccw_on_params, device=device, dtype=dtype)
        syn_so_ccw_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['conductanceSOOn']], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['reversalIn']], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
        })
        self.syn_suppress_on_ccw_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2,
                                                                      params=syn_so_ccw_on_params, device=device, dtype=dtype)
        nrn_ccw_on_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_fast + torch.zeros(shape_emd, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasOn'] + torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
        })
        self.ccw_on = m.NonSpikingLayer(shape_emd, params=nrn_ccw_on_params, device=device, dtype=dtype)
        # self.state_ccw_on = torch.zeros(shape_emd, dtype=dtype, device=device)+nrn_ccw_on_params['init']
        # CW
        syn_eo_cw_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['conductanceEOOn']], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['reversalMod']], [0, 0, 0]],
                                                  dtype=dtype, device=device), requires_grad=False),
        })
        self.syn_enhance_on_cw_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2,
                                                                     params=syn_eo_cw_on_params, device=device, dtype=dtype)
        syn_so_cw_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['conductanceSOOn'], 0, 0], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['reversalIn'], 0, 0], [0, 0, 0]],
                                                  dtype=dtype, device=device), requires_grad=False),
        })
        self.syn_suppress_on_cw_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2,
                                                                      params=syn_so_cw_on_params, device=device, dtype=dtype)
        nrn_cw_on_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_fast + torch.zeros(shape_emd, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasOn'] + torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
        })
        self.cw_on = m.NonSpikingLayer(shape_emd, params=nrn_cw_on_params, device=device, dtype=dtype)
        # self.state_cw_on = torch.zeros(shape_emd, dtype=dtype, device=device)+nrn_cw_on_params['init']



    def forward(self, x, state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass,
                state_enhance_on, state_direct_on, state_suppress_on, state_cw_on, state_ccw_on):
        # Synaptic updates
        # with profiler.record_function("EYE SYNAPTIC UPDATES"):
        syn_input_bandpass_on = self.syn_input_bandpass_on(state_input, state_bo_input)
        syn_input_lowpass = self.syn_input_lowpass(state_input, state_lowpass)
        syn_lowpass_enhance_on = self.syn_lowpass_enhance_on(state_lowpass, state_enhance_on)
        syn_bandpass_on_direct_on = self.syn_bandpass_on_direct_on(state_bo_output, state_direct_on)
        syn_direct_on_suppress_on = self.syn_direct_on_suppress_on(state_direct_on, state_suppress_on)
        syn_enhance_on_ccw_on = self.syn_enhance_on_ccw_on(state_enhance_on, state_ccw_on)
        syn_direct_on_ccw_on = self.syn_direct_on_on(state_direct_on, state_ccw_on)
        syn_suppress_on_ccw_on = self.syn_suppress_on_ccw_on(state_suppress_on, state_ccw_on)
        syn_enhance_on_cw_on = self.syn_enhance_on_cw_on(state_enhance_on, state_cw_on)
        syn_direct_on_cw_on = self.syn_direct_on_on(state_direct_on, state_cw_on)
        syn_suppress_on_cw_on = self.syn_suppress_on_cw_on(state_suppress_on, state_cw_on)

        # Neural updates
        # print(x)
        # print(state_input)
        # with profiler.record_function("EYE NEURAL UPDATES"):
        state_input = self.input(x.squeeze(), state_input)
        state_bo_input, state_bo_fast, state_bo_slow, state_bo_output = self.bandpass_on(syn_input_bandpass_on.squeeze(), state_bo_input, state_bo_fast, state_bo_slow, state_bo_output)
        state_lowpass = self.lowpass(torch.squeeze(syn_input_lowpass), state_lowpass)
        state_enhance_on = self.enhance_on(syn_lowpass_enhance_on, state_enhance_on)
        state_direct_on = self.direct_on(syn_bandpass_on_direct_on, state_direct_on)
        state_suppress_on = self.suppress_on(syn_direct_on_suppress_on, state_suppress_on)
        state_ccw_on = self.ccw_on(torch.squeeze(syn_enhance_on_ccw_on+syn_direct_on_ccw_on+syn_suppress_on_ccw_on), state_ccw_on)
        state_cw_on = self.cw_on(torch.squeeze(syn_enhance_on_cw_on+syn_direct_on_cw_on+syn_suppress_on_cw_on), state_cw_on)

        # with profiler.record_function("EYE OUTPUT FORMAT"):


        return state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass, state_enhance_on, state_direct_on, state_suppress_on, state_cw_on, state_ccw_on

    @jit.export
    def init(self):
        state_input = self.input.params['init']
        state_bo_input, state_bo_fast, state_bo_slow, state_bo_output = self.bandpass_on.init()
        state_lowpass = self.lowpass.params['init']
        state_enhance_on = self.enhance_on.params['init']
        state_direct_on = self.direct_on.params['init']
        state_suppress_on = self.suppress_on.params['init']
        state_ccw_on = self.ccw_on.params['init']
        state_cw_on = self.cw_on.params['init']


        return state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass, state_enhance_on, state_direct_on, state_suppress_on, state_cw_on, state_ccw_on

class SNSMotionVisionEye(nn.Module):
    def __init__(self, dt, shape_input, shape_field, params=None, device=None, dtype=torch.float32, generator=None):
        super().__init__()
        if device is None:
            device = 'cpu'

        self.params = nn.ParameterDict({
            'reversalEx': nn.Parameter(torch.tensor([2.0], dtype=dtype).to(device)),
            'reversalIn': nn.Parameter(torch.tensor([-2.0], dtype=dtype).to(device)),
            'reversalMod': nn.Parameter(torch.tensor([0.0], dtype=dtype).to(device)),
            'freqFast': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'ampCenBO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdCenBO': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ampSurBO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdSurBO': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqBOFast': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'freqBOSlow': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'ampCenL': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdCenL': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ampSurL': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdSurL': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqL': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'ampCenBF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdCenBF': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ampSurBF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdSurBF': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqBFFast': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'freqBFSlow': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'conductanceLEO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqEO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasEO': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceBODO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqDO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasDO': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceDOSO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqSO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasSO': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceLEF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqEF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasEF': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceBFDF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqDF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasDF': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceDFSF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqSF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasSF': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceEOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceDOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceSOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceEFOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceDFOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceSFOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasOn': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'freqOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasOff': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1)
        })
        if params is not None:
            self.params.update(params)
        # Network

        # Retina
        tau_fast = dt/__calc_cap_from_cutoff__(self.params['freqFast'].data)
        nrn_input_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_fast.data + torch.zeros(shape_input, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_input, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_input, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(torch.zeros(shape_input, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_input, dtype=dtype).to(device), requires_grad=False)
        })
        self.input = m.NonSpikingLayer(shape_input, params=nrn_input_params, device=device, dtype=dtype)
        self.state_input = torch.zeros(shape_input, dtype=dtype, device=device)+nrn_input_params['init']

        # Lamina
        shape_post_conv = [x - (shape_field - 1) for x in shape_input]
        # Bo
        conductance, reversal = __calc_2d_field__(self.params['ampCenBO'], self.params['ampSurBO'],
                                                  self.params['stdCenBO'], self.params['stdSurBO'], shape_field,
                                                  self.params['reversalEx'], self.params['reversalIn'])
        syn_in_bo_params = nn.ParameterDict({
            'conductance': nn.Parameter(conductance, requires_grad=False),
            'reversal': nn.Parameter(reversal, requires_grad=False)
        })
        self.syn_input_bandpass_on = m.NonSpikingChemicalSynapseConv(1,1,shape_field, conv_dim=2,
                                                                     params=syn_in_bo_params, device=device, dtype=dtype)
        tau_bo_fast = dt / __calc_cap_from_cutoff__(self.params['freqBOFast'].data)
        tau_bo_slow = dt / __calc_cap_from_cutoff__(self.params['freqBOSlow'].data)
        nrn_bo_params = nn.ParameterDict({
            'input_tau': nn.Parameter((tau_fast + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                      requires_grad=False),
            'input_leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'input_rest': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'input_bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'input_init': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'fast_tau': nn.Parameter((tau_bo_fast + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                     requires_grad=False),
            'fast_leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'fast_rest': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'fast_bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'fast_init': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'slow_tau': nn.Parameter((tau_bo_slow + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                     requires_grad=False),
            'slow_leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'slow_rest': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'slow_bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'slow_init': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'output_tau': nn.Parameter((tau_fast + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                       requires_grad=False),
            'output_leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'output_rest': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'output_bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'output_init': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'reversalIn': nn.Parameter((self.params['reversalIn'].clone().detach()).to(device), requires_grad=False),
            'reversalEx': nn.Parameter((self.params['reversalEx'].clone().detach()).to(device), requires_grad=False),
        })
        self.bandpass_on = SNSBandpass(shape_post_conv, params=nrn_bo_params, device=device, dtype=dtype)
        # L
        conductance, reversal = __calc_2d_field__(self.params['ampCenL'], self.params['ampSurL'],
                                                  self.params['stdCenL'], self.params['stdSurL'], shape_field,
                                                  self.params['reversalEx'], self.params['reversalIn'])
        syn_in_l_params = nn.ParameterDict({
            'conductance': nn.Parameter(conductance, requires_grad=False),
            'reversal': nn.Parameter(reversal, requires_grad=False)
        })
        self.syn_input_lowpass = m.NonSpikingChemicalSynapseConv(1, 1, shape_field, conv_dim=2,
                                                                 params=syn_in_l_params, device=device, dtype=dtype)
        tau_l = dt / __calc_cap_from_cutoff__(self.params['freqL'].data)
        nrn_l_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_l + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
            'init': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device),requires_grad=False),
        })
        self.lowpass = m.NonSpikingLayer(shape_post_conv, params=nrn_l_params, device=device, dtype=dtype)
        self.state_lowpass = torch.zeros(shape_post_conv, dtype=dtype, device=device)+nrn_l_params['init']
        # Bf
        conductance, reversal = __calc_2d_field__(self.params['ampCenBF'], self.params['ampSurBF'],
                                                  self.params['stdCenBF'], self.params['stdSurBF'], shape_field,
                                                  self.params['reversalEx'], self.params['reversalIn'])
        syn_in_bf_params = nn.ParameterDict({
            'conductance': nn.Parameter(conductance, requires_grad=False),
            'reversal': nn.Parameter(reversal, requires_grad=False)
        })
        self.syn_input_bandpass_off = m.NonSpikingChemicalSynapseConv(1, 1, shape_field, conv_dim=2,
                                                                      params=syn_in_bf_params, device=device, dtype=dtype)
        tau_bf_fast = dt / __calc_cap_from_cutoff__(self.params['freqBFFast'].data)
        tau_bf_slow = dt / __calc_cap_from_cutoff__(self.params['freqBFSlow'].data)
        nrn_bf_params = nn.ParameterDict({
            'input_tau': nn.Parameter((tau_fast + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                      requires_grad=False),
            'input_leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'input_rest': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'input_bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'input_init': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'fast_tau': nn.Parameter((tau_bf_fast + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                     requires_grad=False),
            'fast_leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'fast_rest': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'fast_bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'fast_init': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'slow_tau': nn.Parameter((tau_bf_slow + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                     requires_grad=False),
            'slow_leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'slow_rest': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'slow_bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'slow_init': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'output_tau': nn.Parameter((tau_fast + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                       requires_grad=False),
            'output_leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'output_rest': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'output_bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'output_init': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'reversalIn': nn.Parameter((self.params['reversalIn'].clone().detach()).to(device), requires_grad=False),
            'reversalEx': nn.Parameter((self.params['reversalEx'].clone().detach()).to(device), requires_grad=False),
        })
        self.bandpass_off = SNSBandpass(shape_post_conv, params=nrn_bf_params, device=device, dtype=dtype)

        # Medulla
        # On
        # EO
        syn_l_eo_params = nn.ParameterDict({
            'conductance': self.params['conductanceLEO'],
            'reversal': self.params['reversalEx']
        })
        self.syn_lowpass_enhance_on = m.NonSpikingChemicalSynapseElementwise(params=syn_l_eo_params, device=device,
                                                                             dtype=dtype)
        tau_eo = dt / __calc_cap_from_cutoff__(self.params['freqEO'].data)
        nrn_eo_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_eo + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype, device=device).to(device),
                                 requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasEO'] + torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
        })
        self.enhance_on = m.NonSpikingLayer(shape_post_conv, params=nrn_eo_params, device=device, dtype=dtype)
        self.state_enhance_on = torch.zeros(shape_post_conv, dtype=dtype, device=device)+nrn_eo_params['init']
        # DO
        syn_bo_do_params = nn.ParameterDict({
            'conductance': self.params['conductanceBODO'],
            'reversal': self.params['reversalIn']
        })
        self.syn_bandpass_on_direct_on = m.NonSpikingChemicalSynapseElementwise(params=syn_bo_do_params, device=device,
                                                                                dtype=dtype)
        tau_do = dt / __calc_cap_from_cutoff__(self.params['freqDO'].data)
        nrn_do_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_do + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            # 'bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasDO'] + torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
        })
        self.direct_on = m.NonSpikingLayer(shape_post_conv, params=nrn_do_params, device=device, dtype=dtype)
        self.state_direct_on = torch.zeros(shape_post_conv, dtype=dtype, device=device)+nrn_do_params['init']
        # SO
        syn_do_so_params = nn.ParameterDict({
            'conductance': self.params['conductanceDOSO'],
            'reversal': self.params['reversalEx']
        })
        self.syn_direct_on_suppress_on = m.NonSpikingChemicalSynapseElementwise(params=syn_do_so_params, device=device,
                                                                                dtype=dtype)
        tau_so = dt / __calc_cap_from_cutoff__(self.params['freqSO'].data)
        nrn_so_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_so + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasSO'] + torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
        })
        self.suppress_on = m.NonSpikingLayer(shape_post_conv, params=nrn_so_params, device=device, dtype=dtype)
        self.state_suppress_on = torch.zeros(shape_post_conv, dtype=dtype, device=device)+nrn_so_params['init']
        # Off
        # EF
        syn_l_ef_params = nn.ParameterDict({
            'conductance': self.params['conductanceLEF'],
            'reversal': self.params['reversalEx']
        })
        self.syn_lowpass_enhance_off = m.NonSpikingChemicalSynapseElementwise(params=syn_l_ef_params, device=device,
                                                                              dtype=dtype)
        tau_ef = dt / __calc_cap_from_cutoff__(self.params['freqEF'].data)
        nrn_ef_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_ef + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasEF'] + torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
        })
        self.enhance_off = m.NonSpikingLayer(shape_post_conv, params=nrn_ef_params, device=device, dtype=dtype)
        self.state_enhance_off = torch.zeros(shape_post_conv, dtype=dtype, device=device)+nrn_ef_params['init']
        # DF
        syn_bf_df_params = nn.ParameterDict({
            'conductance': self.params['conductanceBFDF'],
            'reversal': self.params['reversalEx']
        })
        offset_activation = m.PiecewiseActivation(min_val=1.0, max_val=2.0)
        self.syn_bandpass_off_direct_off = m.NonSpikingChemicalSynapseElementwise(params=syn_bf_df_params, device=device,
                                                                                  dtype=dtype, activation=offset_activation)
        tau_df = dt / __calc_cap_from_cutoff__(self.params['freqDF'].data)
        nrn_df_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_df + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasDF'] + torch.zeros(shape_post_conv, dtype=dtype).to(device),
                                 requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
        })
        self.direct_off = m.NonSpikingLayer(shape_post_conv, params=nrn_df_params, device=device, dtype=dtype)
        self.state_direct_off = torch.zeros(shape_post_conv, dtype=dtype, device=device)+nrn_df_params['init']
        # SF
        syn_df_sf_params = nn.ParameterDict({
            'conductance': self.params['conductanceDFSF'],
            'reversal': self.params['reversalEx']
        })
        self.syn_direct_off_suppress_off = m.NonSpikingChemicalSynapseElementwise(params=syn_df_sf_params, device=device,
                                                                                dtype=dtype)
        tau_sf = dt / __calc_cap_from_cutoff__(self.params['freqSF'].data)
        nrn_sf_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_sf + torch.zeros(shape_post_conv, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasSF'] + torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
        })
        self.suppress_off = m.NonSpikingLayer(shape_post_conv, params=nrn_sf_params, device=device, dtype=dtype)
        self.state_suppress_off = torch.zeros(shape_post_conv, dtype=dtype, device=device)+nrn_sf_params['init']
        # Lobula
        shape_emd = [x - 2 for x in shape_post_conv]
        # On
        syn_do_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, self.params['reversalEx'], 0], [0, 0, 0]],
                                                  dtype=dtype, device=device), requires_grad=False),
        })
        self.syn_direct_on_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2,
                                                                params=syn_do_on_params, device=device, dtype=dtype)
        # CCW
        syn_eo_ccw_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['conductanceEOOn'], 0, 0], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['reversalMod'], 0, 0], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
        })
        self.syn_enhance_on_ccw_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2,
                                                                      params=syn_eo_ccw_on_params, device=device, dtype=dtype)
        syn_so_ccw_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['conductanceSOOn']], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['reversalIn']], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
        })
        self.syn_suppress_on_ccw_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2,
                                                                      params=syn_so_ccw_on_params, device=device, dtype=dtype)
        nrn_ccw_on_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_fast + torch.zeros(shape_emd, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasOn'] + torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
        })
        self.ccw_on = m.NonSpikingLayer(shape_emd, params=nrn_ccw_on_params, device=device, dtype=dtype)
        self.state_ccw_on = torch.zeros(shape_emd, dtype=dtype, device=device)+nrn_ccw_on_params['init']
        # CW
        syn_eo_cw_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['conductanceEOOn']], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['reversalMod']], [0, 0, 0]],
                                                  dtype=dtype, device=device), requires_grad=False),
        })
        self.syn_enhance_on_cw_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2,
                                                                     params=syn_eo_cw_on_params, device=device, dtype=dtype)
        syn_so_cw_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['conductanceSOOn'], 0, 0], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['reversalIn'], 0, 0], [0, 0, 0]],
                                                  dtype=dtype, device=device), requires_grad=False),
        })
        self.syn_suppress_on_cw_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2,
                                                                      params=syn_so_cw_on_params, device=device, dtype=dtype)
        nrn_cw_on_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_fast + torch.zeros(shape_emd, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasOn'] + torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
        })
        self.cw_on = m.NonSpikingLayer(shape_emd, params=nrn_cw_on_params, device=device, dtype=dtype)
        self.state_cw_on = torch.zeros(shape_emd, dtype=dtype, device=device)+nrn_cw_on_params['init']

        # Off
        syn_df_off_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, self.params['reversalEx'], 0], [0, 0, 0]],
                                                  dtype=dtype, device=device), requires_grad=False),
        })
        self.syn_direct_off_off = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2,
                                                                params=syn_df_off_params, device=device, dtype=dtype)
        # CCW
        syn_ef_ccw_off_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['conductanceEFOff'], 0, 0], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['reversalEx'], 0, 0], [0, 0, 0]],
                                                  dtype=dtype, device=device), requires_grad=False),
        })
        self.syn_enhance_off_ccw_off = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2,
                                                                     params=syn_ef_ccw_off_params, device=device, dtype=dtype)
        syn_sf_ccw_off_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['conductanceSFOff']], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['reversalIn']], [0, 0, 0]],
                                                  dtype=dtype, device=device), requires_grad=False),
        })
        self.syn_suppress_off_ccw_off = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2,
                                                                      params=syn_sf_ccw_off_params, device=device, dtype=dtype)
        nrn_ccw_off_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_fast + torch.zeros(shape_emd, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasOff'] + torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
        })
        self.ccw_off = m.NonSpikingLayer(shape_emd, params=nrn_ccw_off_params, device=device, dtype=dtype)
        self.state_ccw_off = torch.zeros(shape_emd, dtype=dtype, device=device)+nrn_ccw_off_params['init']
        # CW
        syn_ef_cw_off_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['conductanceEFOff']], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['reversalEx']], [0, 0, 0]],
                                                  dtype=dtype, device=device), requires_grad=False),
        })
        self.syn_enhance_off_cw_off = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2,
                                                                    params=syn_ef_cw_off_params, device=device, dtype=dtype)
        syn_sf_cw_off_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['conductanceSFOff'], 0, 0], [0, 0, 0]],
                                                     dtype=dtype, device=device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['reversalIn'], 0, 0], [0, 0, 0]],
                                                  dtype=dtype, device=device), requires_grad=False),
        })
        self.syn_suppress_off_cw_off = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2,
                                                                     params=syn_sf_cw_off_params, device=device, dtype=dtype)
        nrn_cw_off_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_fast + torch.zeros(shape_emd, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasOff'] + torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
        })
        self.cw_off = m.NonSpikingLayer(shape_emd, params=nrn_cw_off_params, device=device, dtype=dtype)
        self.state_cw_off = torch.zeros(shape_emd, dtype=dtype, device=device)+nrn_cw_off_params['init']

    def forward(self, x, reset: bool=False):#, states: List[torch.Tensor]):
        if reset:
            self.reset()
        # with profiler.record_function("EYE STATE COPYING"):
        # state_input = states[0]
        # state_bp_on_input = states[1]
        # state_bp_on_fast = states[2]
        # state_bp_on_slow = states[3]
        # state_bp_on_output = states[4]
        # state_lowpass = states[5]
        # state_bp_off_input = states[6]
        # state_bp_off_fast = states[7]
        # state_bp_off_slow = states[8]
        # state_bp_off_output = states[9]
        # state_enhance_on = states[10]
        # state_direct_on = states[11]
        # state_suppress_on = states[12]
        # state_enhance_off = states[13]
        # state_direct_off = states[14]
        # state_suppress_off = states[15]
        # state_ccw_on = states[16]
        # state_cw_on = states[17]
        # state_ccw_off = states[18]
        # state_cw_off = states[19]

        # Synaptic updates
        # with profiler.record_function("EYE SYNAPTIC UPDATES"):
        syn_input_bandpass_on = self.syn_input_bandpass_on(self.state_input, self.bandpass_on.state_input)
        syn_input_lowpass = self.syn_input_lowpass(self.state_input, self.state_lowpass)
        syn_input_bandpass_off = self.syn_input_bandpass_off(self.state_input, self.bandpass_off.state_input)
        syn_lowpass_enhance_on = self.syn_lowpass_enhance_on(self.state_lowpass, self.state_enhance_on)
        syn_bandpass_on_direct_on = self.syn_bandpass_on_direct_on(self.bandpass_on.state_output, self.state_direct_on)
        syn_direct_on_suppress_on = self.syn_direct_on_suppress_on(self.state_direct_on, self.state_suppress_on)
        syn_lowpass_enhance_off = self.syn_lowpass_enhance_off(self.state_lowpass, self.state_enhance_off)
        syn_bandpass_off_direct_off = self.syn_bandpass_off_direct_off(self.bandpass_off.state_output, self.state_direct_off)
        syn_direct_off_suppress_off = self.syn_direct_off_suppress_off(self.state_direct_off, self.state_suppress_off)
        syn_enhance_on_ccw_on = self.syn_enhance_on_ccw_on(self.state_enhance_on, self.state_ccw_on)
        syn_direct_on_ccw_on = self.syn_direct_on_on(self.state_direct_on, self.state_ccw_on)
        syn_suppress_on_ccw_on = self.syn_suppress_on_ccw_on(self.state_suppress_on, self.state_ccw_on)
        syn_enhance_on_cw_on = self.syn_enhance_on_cw_on(self.state_enhance_on, self.state_cw_on)
        syn_direct_on_cw_on = self.syn_direct_on_on(self.state_direct_on, self.state_cw_on)
        syn_suppress_on_cw_on = self.syn_suppress_on_cw_on(self.state_suppress_on, self.state_cw_on)
        syn_enhance_off_ccw_off = self.syn_enhance_off_ccw_off(self.state_enhance_off, self.state_ccw_off)
        syn_direct_off_ccw_off = self.syn_direct_off_off(self.state_direct_off, self.state_ccw_off)
        syn_suppress_off_ccw_off = self.syn_suppress_off_ccw_off(self.state_suppress_off, self.state_ccw_off)
        syn_enhance_off_cw_off = self.syn_enhance_off_cw_off(self.state_enhance_off, self.state_cw_off)
        syn_direct_off_cw_off = self.syn_direct_off_off(self.state_direct_off, self.state_cw_off)
        syn_suppress_off_cw_off = self.syn_suppress_off_cw_off(self.state_suppress_off, self.state_cw_off)

        # Neural updates
        # print(x)
        # print(state_input)
        # with profiler.record_function("EYE NEURAL UPDATES"):
        self.state_input = self.input(x.squeeze(), self.state_input)
        _ = self.bandpass_on(syn_input_bandpass_on.squeeze(), reset=reset)
        self.state_lowpass = self.lowpass(torch.squeeze(syn_input_lowpass), self.state_lowpass)
        _ = self.bandpass_off(syn_input_bandpass_off.squeeze(), reset=reset)
        self.state_enhance_on = self.enhance_on(syn_lowpass_enhance_on, self.state_enhance_on)
        self.state_direct_on = self.direct_on(syn_bandpass_on_direct_on, self.state_direct_on)
        self.state_suppress_on = self.suppress_on(syn_direct_on_suppress_on, self.state_suppress_on)
        self.state_enhance_off = self.enhance_off(syn_lowpass_enhance_off, self.state_enhance_off)
        self.state_direct_off = self.direct_off(syn_bandpass_off_direct_off, self.state_direct_off)
        self.state_suppress_off = self.suppress_off(syn_direct_off_suppress_off, self.state_suppress_off)
        self.state_ccw_on = self.ccw_on(torch.squeeze(syn_enhance_on_ccw_on+syn_direct_on_ccw_on+syn_suppress_on_ccw_on), self.state_ccw_on)
        self.state_cw_on = self.cw_on(torch.squeeze(syn_enhance_on_cw_on+syn_direct_on_cw_on+syn_suppress_on_cw_on), self.state_cw_on)
        self.state_ccw_off = self.ccw_off(torch.squeeze(syn_enhance_off_ccw_off+syn_direct_off_ccw_off+syn_suppress_off_ccw_off), self.state_ccw_off)
        self.state_cw_off = self.cw_off(torch.squeeze(syn_enhance_off_cw_off+syn_direct_off_cw_off+syn_suppress_off_cw_off), self.state_cw_off)

        # with profiler.record_function("EYE OUTPUT FORMAT"):


        return self.state_ccw_on, self.state_cw_on, self.state_ccw_off, self.state_cw_off

    @jit.export
    def reset(self):
        self.state_input = self.input.params['init']
        self.state_lowpass = self.lowpass.params['init']
        self.state_enhance_on = self.enhance_on.params['init']
        self.state_direct_on = self.direct_on.params['init']
        self.state_suppress_on = self.suppress_on.params['init']
        self.state_enhance_off = self.enhance_off.params['init']
        self.state_direct_off = self.direct_off.params['init']
        self.state_suppress_off = self.suppress_off.params['init']
        self.state_ccw_on = self.ccw_on.params['init']
        self.state_cw_on = self.cw_on.params['init']
        self.state_ccw_off = self.ccw_off.params['init']
        self.state_cw_off = self.cw_off.params['init']

class SNSMotionVisionOn(nn.Module):
    def __init__(self, dt, shape_input, shape_field, params=None, device=None, dtype=torch.float32, generator=None):
        super().__init__()
        if device is None:
            device = 'cpu'

        freq_fast = 1000/(2*math.ceil(dt)*torch.pi)
        self.params = nn.ParameterDict({
            'reversalEx': nn.Parameter(torch.tensor([5.0], dtype=dtype).to(device), requires_grad=False),
            'reversalIn': nn.Parameter(torch.tensor([-2.0], dtype=dtype).to(device), requires_grad=False),
            'reversalMod': nn.Parameter(torch.tensor([-0.1], dtype=dtype).to(device), requires_grad=False),
            'freqFast': nn.Parameter(torch.tensor(freq_fast, dtype=dtype).to(device), requires_grad=False),
            'ampCenBO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdCenBO': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ampSurBO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdSurBO': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqBOFast': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'freqBOSlow': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'ampCenLO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdCenLO': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ampSurLO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdSurLO': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqLO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'conductanceLOEO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqEO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasEO': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceBODO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqDO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasDO': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceDOSO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqSO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasSO': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceEOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceDOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceSOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasOn': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'gainHorizontal': nn.Parameter(torch.tensor([1.0], dtype=dtype).to(device)),
        })
        if params is not None:
            self.params.update(params)
        self.dt = dt
        self.shape_field = shape_field
        self.shape_input = shape_input
        shape_post_conv = [x - (self.shape_field - 1) for x in self.shape_input]
        shape_emd = [x - 2 for x in shape_post_conv]
        flat_shape_emd = shape_emd[0] * shape_emd[1]
        self.device = device
        self.generator = generator
        self.dtype = dtype
        self.eye = OnPathway(dt, shape_input, shape_field, params=self.params, device=device, dtype=dtype,
                                      generator=None)
        self.syn_cw_ex_ccw_in = m.NonSpikingChemicalSynapseLinear(flat_shape_emd, 2, device=self.device,
                                                                  dtype=self.dtype, generator=self.generator)
        self.setup()

    def forward(self, x, state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass,
                state_enhance_on, state_direct_on, state_suppress_on, state_cw_on, state_ccw_on, state_horizontal, avg,
                niter):

        # with profiler.record_function("MERGED SYNAPTIC UPDATES"):
        syn_cw_on = self.syn_cw_ex_ccw_in(torch.flatten(state_cw_on), state_horizontal)
        syn_ccw_on = self.syn_cw_in_ccw_ex(torch.flatten(state_ccw_on), state_horizontal)

        # with profiler.record_function("EYE PROCESSING"):
        (state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass, state_enhance_on,
         state_direct_on, state_suppress_on, state_cw_on, state_ccw_on) = self.eye(x, state_input, state_bo_input,
                                                                                   state_bo_fast, state_bo_slow,
                                                                                   state_bo_output, state_lowpass,
                                                                                   state_enhance_on, state_direct_on,
                                                                                   state_suppress_on, state_cw_on,
                                                                                   state_ccw_on)

        # with profiler.record_function("MERGED NEURAL UPDATE"):
        state_horizontal = self.horizontal(syn_ccw_on+syn_cw_on, state_horizontal)

        avg += state_horizontal[0]/niter

        return (state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass,
                state_enhance_on, state_direct_on, state_suppress_on, state_cw_on, state_ccw_on, state_horizontal, avg)

    @jit.export
    def init(self):
        (state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass, state_enhance_on,
         state_direct_on, state_suppress_on, state_cw_on, state_ccw_on) = self.eye.init()
        state_horizontal = self.horizontal.params['init']
        avg = torch.zeros(1, dtype=self.dtype, device=self.device)
        return (state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass,
                state_enhance_on, state_direct_on, state_suppress_on, state_cw_on, state_ccw_on, state_horizontal, avg)

    @jit.export
    def setup(self):
        self.eye.setup()
        shape_post_conv = [x - (self.shape_field - 1) for x in self.shape_input]
        shape_emd = [x - 2 for x in shape_post_conv]
        flat_shape_emd = shape_emd[0] * shape_emd[1]
        num_zero = 6 * shape_emd[0]
        if shape_emd[1] > 6:
            flat_shape_emd_corrected = flat_shape_emd - num_zero
        else:
            flat_shape_emd_corrected = flat_shape_emd

        gain_horizontal = torch.clamp(self.params['gainHorizontal'], 0, 1)
        g_ex_full = gain_horizontal / (self.params['reversalEx'] - gain_horizontal)
        g_in_full = (-gain_horizontal * self.params['reversalEx']) / (
                    self.params['reversalIn'] * (self.params['reversalEx'] - gain_horizontal))
        g_ex_tensor = torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) + g_ex_full / flat_shape_emd_corrected
        g_in_tensor = torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) + g_in_full / flat_shape_emd_corrected
        if shape_emd[1] > 6:
            g_ex_tensor[(int(shape_emd[1] / 2) - 3):(int(shape_emd[1] / 2) + 3)] = 0.0
            g_in_tensor[(int(shape_emd[1] / 2) - 3):(int(shape_emd[1] / 2) + 3)] = 0.0
        reversal_ex_tensor = torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) + self.params['reversalEx']
        reversal_in_tensor = torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) + self.params['reversalIn']

        # Horizontal Cells
        syn_cw_ex_ccw_in_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.vstack((g_ex_tensor, g_in_tensor)).to(self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.vstack((reversal_ex_tensor, reversal_in_tensor)).to(self.device),
                                     requires_grad=False)
        })
        syn_cw_in_ccw_ex_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.vstack((g_in_tensor, g_ex_tensor)).to(self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.vstack((reversal_in_tensor, reversal_ex_tensor)).to(self.device),
                                     requires_grad=False)
        })
        self.syn_cw_ex_ccw_in.params.update(syn_cw_ex_ccw_in_params)

        self.syn_cw_in_ccw_ex = m.NonSpikingChemicalSynapseLinear(flat_shape_emd, 2,
                                                                  params=syn_cw_in_ccw_ex_params, device=self.device,
                                                                  dtype=self.dtype, generator=self.generator)
        tau_horizontal = self.dt / __calc_cap_from_cutoff__(self.params['freqFast'].data)
        nrn_hc_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_horizontal.data + torch.zeros(2, dtype=self.dtype, device=self.device)).to(self.device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(2, dtype=self.dtype).to(self.device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(2, dtype=self.dtype).to(self.device), requires_grad=False),
            'bias': nn.Parameter(torch.zeros(2, dtype=self.dtype).to(self.device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(2, dtype=self.dtype).to(self.device), requires_grad=False)
        })
        self.horizontal = m.NonSpikingLayer(2, params=nrn_hc_params, device=self.device, dtype=self.dtype)
        self.state_horizontal = torch.zeros(2, dtype=self.dtype, device=self.device) + nrn_hc_params['init']

class SNSMotionVisionMerged(nn.Module):
    def __init__(self, dt, shape_input, shape_field, params=None, device=None, dtype=torch.float32, generator=None):
        super().__init__()
        if device is None:
            device = 'cpu'

        self.params = nn.ParameterDict({
            'reversalEx': nn.Parameter(torch.tensor([2.0], dtype=dtype).to(device)),
            'reversalIn': nn.Parameter(torch.tensor([-2.0], dtype=dtype).to(device)),
            'reversalMod': nn.Parameter(torch.tensor([0.0], dtype=dtype).to(device)),
            'freqFast': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'ampCenBO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdCenBO': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ampSurBO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdSurBO': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqBOFast': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'freqBOSlow': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'ampCenL': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdCenL': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ampSurL': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdSurL': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqL': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'ampCenBF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdCenBF': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ampSurBF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdSurBF': nn.Parameter(1+99*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqBFFast': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'freqBFSlow': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'conductanceLEO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqEO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasEO': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceBODO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqDO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasDO': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceDOSO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqSO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasSO': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceLEF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqEF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasEF': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceBFDF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqDF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasDF': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceDFSF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqSF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasSF': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'conductanceEOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceDOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceSOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceEFOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceDFOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceSFOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasOn': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'freqOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasOff': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
            'gainHorizontal': nn.Parameter(torch.tensor([1.0], dtype=dtype).to(device)),
            'freqHorizontal': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'biasHorizontal': nn.Parameter(2*torch.rand(1, dtype=dtype, generator=generator).to(device)-1),
        })
        if params is not None:
            self.params.update(params)
        self.dt = dt
        self.device = device
        self.dtype = dtype
        self.eye = SNSMotionVisionEye(dt, shape_input, shape_field, params=self.params, device=device, dtype=dtype,
                                      generator=None)
        shape_post_conv = [x - (shape_field - 1) for x in shape_input]
        shape_emd = [x - 2 for x in shape_post_conv]
        flat_shape_emd = shape_emd[0]*shape_emd[1]
        num_zero = 6*shape_emd[0]
        if shape_emd[1] > 6:
            flat_shape_emd_corrected = flat_shape_emd-num_zero
        else:
            flat_shape_emd_corrected = flat_shape_emd

        g_ex_full = self.params['gainHorizontal']/(self.params['reversalEx']-self.params['gainHorizontal'])
        g_in_full = (-self.params['gainHorizontal']*self.params['reversalEx'])/(self.params['reversalIn']*(self.params['reversalEx']-self.params['gainHorizontal']))
        g_ex_tensor = torch.zeros(flat_shape_emd, dtype=dtype, device=device) + g_ex_full/flat_shape_emd_corrected
        g_in_tensor = torch.zeros(flat_shape_emd, dtype=dtype, device=device) + g_in_full / flat_shape_emd_corrected
        if shape_emd[1] > 6:
            g_ex_tensor[(int(shape_emd[1]/2)-3):(int(shape_emd[1]/2)+3)] = 0.0
            g_in_tensor[(int(shape_emd[1]/2)-3):(int(shape_emd[1]/2)+3)] = 0.0
        reversal_ex_tensor = torch.zeros(flat_shape_emd, dtype=dtype, device=device) + self.params['reversalEx']
        reversal_in_tensor = torch.zeros(flat_shape_emd, dtype=dtype, device=device) + self.params['reversalIn']

        # Horizontal Cells
        syn_cw_ex_ccw_in_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.vstack((g_ex_tensor,g_in_tensor)).to(device), requires_grad=False),
            'reversal': nn.Parameter(torch.vstack((reversal_ex_tensor,reversal_in_tensor)).to(device),
                                     requires_grad=False)
        })
        syn_cw_in_ccw_ex_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.vstack((g_in_tensor, g_ex_tensor)).to(device), requires_grad=False),
            'reversal': nn.Parameter(torch.vstack((reversal_in_tensor, reversal_ex_tensor)).to(device),
                                     requires_grad=False)
        })
        self.syn_cw_ex_ccw_in = m.NonSpikingChemicalSynapseLinear(flat_shape_emd, 2,
                                                                  params=syn_cw_ex_ccw_in_params, device=device,
                                                                  dtype=dtype, generator=generator)
        self.syn_cw_in_ccw_ex = m.NonSpikingChemicalSynapseLinear(flat_shape_emd, 2,
                                                                  params=syn_cw_in_ccw_ex_params, device=device,
                                                                  dtype=dtype, generator=generator)
        tau_horizontal = dt / __calc_cap_from_cutoff__(self.params['freqHorizontal'].data)
        nrn_hc_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_horizontal.data + torch.zeros(2, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(2, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(2, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasHorizontal'] + torch.zeros(2, dtype=dtype).to(device),
                                 requires_grad=False),
            'init': nn.Parameter(torch.zeros(2, dtype=dtype).to(device), requires_grad=False)
        })
        self.horizontal = m.NonSpikingLayer(2, params=nrn_hc_params, device=device, dtype=dtype)
        self.state_horizontal = torch.zeros(2, dtype=dtype, device=device)+nrn_hc_params['init']

    def forward(self, x, reset: bool=False):#, states):
        if reset:
            self.reset()
        # with profiler.record_function("MERGED STATE INPUT"):
        # state_ccw_on = states[16]
        # state_cw_on = states[17]
        # state_ccw_off = states[18]
        # state_cw_off = states[19]
        # state_horizontal = states[20]

        # with profiler.record_function("MERGED SYNAPTIC UPDATES"):
        syn_cw_on = self.syn_cw_ex_ccw_in(torch.flatten(self.eye.state_cw_on), self.state_horizontal)
        syn_ccw_on = self.syn_cw_in_ccw_ex(torch.flatten(self.eye.state_ccw_on), self.state_horizontal)
        syn_cw_off = self.syn_cw_ex_ccw_in(torch.flatten(self.eye.state_cw_off), self.state_horizontal)
        syn_ccw_off = self.syn_cw_in_ccw_ex(torch.flatten(self.eye.state_ccw_off), self.state_horizontal)

        # with profiler.record_function("EYE PROCESSING"):
        _ = self.eye(x, reset=reset)

        # with profiler.record_function("MERGED NEURAL UPDATE"):
        self.state_horizontal = self.horizontal(syn_ccw_on+syn_cw_on+syn_ccw_off+syn_cw_off, self.state_horizontal)

        # with profiler.record_function("MERGED STATE OUTPUT"):
        # states = [state_input, state_bp_on_input, state_bp_on_fast, state_bp_on_slow, state_bp_on_output,
        #  state_lowpass, state_bp_off_input, state_bp_off_fast, state_bp_off_slow, state_bp_off_output,
        #  state_enhance_on, state_direct_on, state_suppress_on, state_enhance_off, state_direct_off,
        #  state_suppress_off, state_ccw_on, state_cw_on, state_ccw_off, state_cw_off, state_horizontal]

        return self.state_horizontal

    @jit.export
    def reset(self):
        self.state_horizontal = torch.zeros(2, dtype=self.dtype, device=self.device)


if __name__ == "__main__":
    img_size = [24,64]
    # print(SNSBandpass(img_size))
    # print(SNSMotionVisionEye(img_size,5))
    print(SNSMotionVisionMerged(2.56, img_size, 5))