from snstorch import modules as m
import torch
import torch.nn as nn
import numpy as np

def __calc_cap_from_cutoff__(cutoff):
    cap = 1000/(2*np.pi*cutoff)
    return cap

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

        k = 1.0
        k = 1.141306594552181
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
        self.syn_input_slow = m.NonSpikingChemicalSynapseElementwise(params=params_input_syn, device=device, dtype=dtype)
        params_slow = nn.ParameterDict({
            'tau': self.params['slow_tau'],
            'leak': self.params['slow_leak'],
            'rest': self.params['slow_rest'],
            'bias': self.params['slow_bias'],
            'init': self.params['slow_init']
        })
        self.slow = m.NonSpikingLayer(shape, params=params_slow, device=device, dtype=dtype)
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

    def forward(self, x, states):
        state_input = states[0]
        state_fast = states[1]
        state_slow = states[2]
        state_output = states[3]

        input2fast = self.syn_input_fast(state_input, state_fast)
        input2slow = self.syn_input_slow(state_input, state_slow)
        fast2out = self.syn_fast_output(state_fast, state_output)
        slow2out = self.syn_slow_output(state_slow, state_output)

        state_input = self.input(x, state_input)
        state_fast = self.fast(input2fast, state_fast)
        state_slow = self.slow(input2slow, state_slow)
        state_output = self.output(fast2out+slow2out, state_output)
        return [state_input, state_fast, state_slow, state_output]

class SNSMotionVisionEye(nn.Module):
    def __init__(self, dt, shape_input, shape_field, params=None, device=None, dtype=torch.float32, generator=None):
        super().__init__()
        if device is None:
            device = 'cpu'

        self.params = nn.ParameterDict({
            'reversalEx': nn.Parameter(torch.tensor([2.0], dtype=dtype).to(device)),
            'reversalIn': nn.Parameter(torch.tensor([-2.0], dtype=dtype).to(device)),
            'reversalMod': nn.Parameter(torch.tensor([0.0], dtype=dtype).to(device)),
            'freqFast': nn.Parameter(torch.rand(1,dtype=dtype, generator=generator).to(device)*1000),
            'kernelConductanceInBO': nn.Parameter(torch.rand(shape_field, dtype=dtype, generator=generator).to(device)),
            'kernelReversalInBO': nn.Parameter((2*torch.rand(shape_field, generator=generator)-1).to(device)),
            'freqBOFast': nn.Parameter(torch.rand(1,dtype=dtype, generator=generator).to(device)*1000),
            'freqBOSlow': nn.Parameter(torch.rand(1,dtype=dtype, generator=generator).to(device)*1000),
            'kernelConductanceInL': nn.Parameter(torch.rand(shape_field, dtype=dtype, generator=generator).to(device)),
            'kernelReversalInL': nn.Parameter((2*torch.rand(shape_field, generator=generator)-1).to(device)),
            'freqL': nn.Parameter(torch.rand(1,dtype=dtype, generator=generator).to(device)*1000),
            'kernelConductanceInBF': nn.Parameter(torch.rand(shape_field, dtype=dtype, generator=generator).to(device)),
            'kernelReversalInBF': nn.Parameter((2*torch.rand(shape_field, generator=generator)-1).to(device)),
            'freqBFFast': nn.Parameter(torch.rand(1,dtype=dtype, generator=generator).to(device)*1000),
            'freqBFSlow': nn.Parameter(torch.rand(1,dtype=dtype, generator=generator).to(device)*1000),
            'conductanceLEO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqEO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'conductanceBODO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqDO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'conductanceDOSO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'freqSO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device) * 1000),
            'conductanceEOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceDOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceSOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceEFOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceDFOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceSFOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
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

        # Lamina
        shape_post_conv = [x - (shape_field - 1) for x in shape_input]
        # Bo
        syn_in_bo_params = nn.ParameterDict({
            'conductance': nn.Parameter(self.params['kernelConductanceInBO'].data, requires_grad=False),
            'reversal': nn.Parameter(self.params['kernelReversalInBO'].data, requires_grad=False)
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
        syn_in_l_params = nn.ParameterDict({
            'conductance': nn.Parameter(self.params['kernelConductanceInBO'].data, requires_grad=False),
            'reversal': nn.Parameter(self.params['kernelReversalInBO'].data, requires_grad=False)
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
        # Bf
        syn_in_bf_params = nn.ParameterDict({
            'conductance': nn.Parameter(self.params['kernelConductanceInBF'].data, requires_grad=False),
            'reversal': nn.Parameter(self.params['kernelReversalInBF'].data, requires_grad=False)
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
            'bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
        })
        self.enhance_on = m.NonSpikingLayer(shape_post_conv, params=nrn_eo_params, device=device, dtype=dtype)
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
            'bias': nn.Parameter(1.0921092205690466 + torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
        })
        self.direct_on = m.NonSpikingLayer(shape_post_conv, params=nrn_do_params, device=device, dtype=dtype)
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
            'bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
        })
        self.suppress_on = m.NonSpikingLayer(shape_post_conv, params=nrn_so_params, device=device, dtype=dtype)
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
            'bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.ones(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
        })
        self.enhance_off = m.NonSpikingLayer(shape_post_conv, params=nrn_ef_params, device=device, dtype=dtype)
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
            'bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device),
                                 requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
        })
        self.direct_off = m.NonSpikingLayer(shape_post_conv, params=nrn_df_params, device=device, dtype=dtype)
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
            'bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
        })
        self.suppress_off = m.NonSpikingLayer(shape_post_conv, params=nrn_sf_params, device=device, dtype=dtype)
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
            'bias': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
        })
        self.ccw_on = m.NonSpikingLayer(shape_emd, params=nrn_ccw_on_params, device=device, dtype=dtype)
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
            'bias': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
        })
        self.cw_on = m.NonSpikingLayer(shape_emd, params=nrn_cw_on_params, device=device, dtype=dtype)

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
            'bias': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
        })
        self.ccw_off = m.NonSpikingLayer(shape_emd, params=nrn_ccw_off_params, device=device, dtype=dtype)
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
            'bias': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_emd, dtype=dtype).to(device), requires_grad=False),
        })
        self.cw_off = m.NonSpikingLayer(shape_emd, params=nrn_cw_off_params, device=device, dtype=dtype)

    def forward(self, x, states):
        state_input = states[0]
        state_bp_on_input = states[1]
        state_bp_on_fast = states[2]
        state_bp_on_slow = states[3]
        state_bp_on_output = states[4]
        state_lowpass = states[5]
        state_bp_off_input = states[6]
        state_bp_off_fast = states[7]
        state_bp_off_slow = states[8]
        state_bp_off_output = states[9]
        state_enhance_on = states[10]
        state_direct_on = states[11]
        state_suppress_on = states[12]
        state_enhance_off = states[13]
        state_direct_off = states[14]
        state_suppress_off = states[15]
        state_ccw_on = states[16]
        state_cw_on = states[17]
        state_ccw_off = states[18]
        state_cw_off = states[19]

        # Synaptic updates
        syn_input_bandpass_on = self.syn_input_bandpass_on(state_input, state_bp_on_input)
        syn_input_lowpass = self.syn_input_lowpass(state_input, state_lowpass)
        syn_input_bandpass_off = self.syn_input_bandpass_off(state_input, state_bp_off_input)
        syn_lowpass_enhance_on = self.syn_lowpass_enhance_on(state_lowpass, state_enhance_on)
        syn_bandpass_on_direct_on = self.syn_bandpass_on_direct_on(state_bp_on_output, state_direct_on)
        syn_direct_on_suppress_on = self.syn_direct_on_suppress_on(state_direct_on, state_suppress_on)
        syn_lowpass_enhance_off = self.syn_lowpass_enhance_off(state_lowpass, state_enhance_off)
        syn_bandpass_off_direct_off = self.syn_bandpass_off_direct_off(state_bp_off_output, state_direct_off)
        syn_direct_off_suppress_off = self.syn_direct_off_suppress_off(state_direct_off, state_suppress_off)
        syn_enhance_on_ccw_on = self.syn_enhance_on_ccw_on(state_enhance_on, state_ccw_on)
        syn_direct_on_ccw_on = self.syn_direct_on_on(state_direct_on, state_ccw_on)
        syn_suppress_on_ccw_on = self.syn_suppress_on_ccw_on(state_suppress_on, state_ccw_on)
        syn_enhance_on_cw_on = self.syn_enhance_on_cw_on(state_enhance_on, state_cw_on)
        syn_direct_on_cw_on = self.syn_direct_on_on(state_direct_on, state_cw_on)
        syn_suppress_on_cw_on = self.syn_suppress_on_cw_on(state_suppress_on, state_cw_on)
        syn_enhance_off_ccw_off = self.syn_enhance_off_ccw_off(state_enhance_off, state_ccw_off)
        syn_direct_off_ccw_off = self.syn_direct_off_off(state_direct_off, state_ccw_off)
        syn_suppress_off_ccw_off = self.syn_suppress_off_ccw_off(state_suppress_off, state_ccw_off)
        syn_enhance_off_cw_off = self.syn_enhance_off_cw_off(state_enhance_off, state_cw_off)
        syn_direct_off_cw_off = self.syn_direct_off_off(state_direct_off, state_cw_off)
        syn_suppress_off_cw_off = self.syn_suppress_off_cw_off(state_suppress_off, state_cw_off)

        # Neural updates
        # print(x)
        # print(state_input)
        state_input = self.input(x.squeeze(), state_input)

        [state_bp_on_input, state_bp_on_fast, state_bp_on_slow, state_bp_on_output] = self.bandpass_on(
            syn_input_bandpass_on.squeeze(), [state_bp_on_input, state_bp_on_fast, state_bp_on_slow, state_bp_on_output])
        state_lowpass = self.lowpass(torch.squeeze(syn_input_lowpass), state_lowpass)
        [state_bp_off_input, state_bp_off_fast, state_bp_off_slow, state_bp_off_output] = self.bandpass_off(
            syn_input_bandpass_off.squeeze(), [state_bp_off_input, state_bp_off_fast, state_bp_off_slow, state_bp_off_output])
        state_enhance_on = self.enhance_on(syn_lowpass_enhance_on, state_enhance_on)
        state_direct_on = self.direct_on(syn_bandpass_on_direct_on, state_direct_on)
        state_suppress_on = self.suppress_on(syn_direct_on_suppress_on, state_suppress_on)
        state_enhance_off = self.enhance_off(syn_lowpass_enhance_off, state_enhance_off)
        state_direct_off = self.direct_off(syn_bandpass_off_direct_off, state_direct_off)
        state_suppress_off = self.suppress_off(syn_direct_off_suppress_off, state_suppress_off)
        state_ccw_on = self.ccw_on(torch.squeeze(syn_enhance_on_ccw_on+syn_direct_on_ccw_on+syn_suppress_on_ccw_on), state_ccw_on)
        state_cw_on = self.cw_on(torch.squeeze(syn_enhance_on_cw_on+syn_direct_on_cw_on+syn_suppress_on_cw_on), state_cw_on)
        state_ccw_off = self.ccw_off(torch.squeeze(syn_enhance_off_ccw_off+syn_direct_off_ccw_off+syn_suppress_off_ccw_off), state_ccw_off)
        state_cw_off = self.cw_off(torch.squeeze(syn_enhance_off_cw_off+syn_direct_off_cw_off+syn_suppress_off_cw_off), state_cw_off)

        return [state_input, state_bp_on_input, state_bp_on_fast, state_bp_on_slow, state_bp_on_output,
                state_lowpass, state_bp_off_input, state_bp_off_fast, state_bp_off_slow, state_bp_off_output,
                state_enhance_on, state_direct_on, state_suppress_on, state_enhance_off, state_direct_off,
                state_suppress_off, state_ccw_on, state_cw_on, state_ccw_off, state_cw_off]

        # # Medulla (On)
        # # Do
        # self.syn_bandpass_on_direct_on = m.ChemicalSynapseElementwise(device=device, dtype=dtype)
        # self.direct_on = m.NonSpikingLayer(shape_post_conv, device=device, dtype=dtype)
        # # Eo
        # self.syn_lowpass_enhance_on = m.ChemicalSynapseElementwise(device=device, dtype=dtype)
        # self.enhance_on = m.NonSpikingLayer(shape_post_conv, device=device, dtype=dtype)
        # # So
        # self.syn_direct_on_suppress_on = m.ChemicalSynapseElementwise(device=device, dtype=dtype)
        # self.suppress_on = m.NonSpikingLayer(shape_post_conv, device=device, dtype=dtype)
        #
        # # Medulla (Off)
        # # Df
        # self.syn_bandpass_off_direct_off = m.ChemicalSynapseElementwise(device=device, dtype=dtype)
        # self.syn_suppress_off_direct_off = m.ChemicalSynapseElementwise(device=device, dtype=dtype)
        # self.direct_off = m.NonSpikingLayer(shape_post_conv, device=device, dtype=dtype)
        # # Ef
        # self.syn_lowpass_enhance_off = m.ChemicalSynapseElementwise(device=device, dtype=dtype)
        # self.syn_suppress_off_enhance_off = m.ChemicalSynapseElementwise(device=device, dtype=dtype)
        # self.enhance_off = m.NonSpikingLayer(shape_post_conv, device=device, dtype=dtype)
        # # Sf
        # self.syn_direct_off_suppress_off = m.ChemicalSynapseElementwise(device=device, dtype=dtype)
        # self.syn_enhance_off_suppress_off = m.ChemicalSynapseElementwise(device=device, dtype=dtype)
        # self.suppress_off = m.NonSpikingLayer(shape_post_conv, device=device, dtype=dtype)
        #
        # # Detectors
        # shape_emd = [x - 2 for x in shape_post_conv]
        # # On EMD
        # cond_enhance_on_emd = nn.Parameter(torch.tensor([9.0],device=device,dtype=dtype),requires_grad=True)
        # cond_direct_on_emd = nn.Parameter(torch.tensor([0.262],device=device,dtype=dtype),requires_grad=True)
        # cond_suppress_on_emd = nn.Parameter(torch.tensor([0.5],device=device,dtype=dtype),requires_grad=True)
        #
        # # CW
        # kernel_cond_enhance_on_cw_on = torch.tensor([[0,0,0],[0,0,cond_enhance_on_emd.data_sns_toolbox],[0,0,0]],device=device,dtype=dtype)
        # kernel_cond_direct_on_cw_on = torch.tensor([[0,0,0],[0,cond_direct_on_emd.data_sns_toolbox,0,0],[0,0,0]],device=device,dtype=dtype)
        # kernel_cond_suppress_on_cw_on = torch.tensor([[0,0,0],[cond_suppress_on_emd.data_sns_toolbox,0,0],[0,0,0]],device=device,dtype=dtype)
        #
        # kernel_reversal_enhance_on_cw_on = torch.tensor([[0, 0, 0], [0, 0, reversal_mod.data_sns_toolbox], [0, 0, 0]],
        #                                             device=device, dtype=dtype)
        # kernel_reversal_direct_on_cw_on = torch.tensor([[0, 0, 0], [0, reversal_ex.data_sns_toolbox, 0, 0], [0, 0, 0]],
        #                                            device=device, dtype=dtype)
        # kernel_reversal_suppress_on_cw_on = torch.tensor([[0, 0, 0], [reversal_in.data_sns_toolbox, 0, 0], [0, 0, 0]],
        #                                          device=device, dtype=dtype)
        #
        # self.syn_enhance_on_emd_cw_on = m.ChemicalSynapseConv2d(1,1,3, kernel_conductance=ke, device=device, dtype=dtype)
        # self.syn_direct_on_emd_cw_on = m.ChemicalSynapseConv2d(1, 1, 3, device=device, dtype=dtype)
        # self.syn_suppress_on_emd_cw_on = m.ChemicalSynapseConv2d(1, 1, 3, device=device, dtype=dtype)
        # self.emd_cw_on = m.NonSpikingLayer(shape_emd, device=device, dtype=dtype)
        #
        # # CCW
        # kernel_cond_enhance_on_ccw_on = torch.tensor([[0, 0, 0], [cond_enhance_on_emd.data_sns_toolbox, 0, 0], [0, 0, 0]],
        #                                              device=device, dtype=dtype)
        # kernel_cond_direct_on_ccw_on = torch.tensor([[0, 0, 0], [0, cond_direct_on_emd.data_sns_toolbox, 0, 0], [0, 0, 0]],
        #                                             device=device, dtype=dtype)
        # kernel_cond_suppress_on_ccw_on = torch.tensor([[0, 0, 0], [0, 0, cond_enhance_on_emd.data_sns_toolbox], [0, 0, 0]],
        #                                           device=device, dtype=dtype)
        #
        # kernel_reversal_enhance_on_ccw_on = torch.tensor([[0, 0, 0], [reversal_mod.data_sns_toolbox, 0, 0], [0, 0, 0]],
        #                                              device=device, dtype=dtype)
        # kernel_reversal_direct_on_ccw_on = torch.tensor([[0, 0, 0], [0, reversal_ex.data_sns_toolbox, 0, 0], [0, 0, 0]],
        #                                             device=device, dtype=dtype)
        # kernel_reversal_suppress_on_ccw_on = torch.tensor([[0, 0, 0], [0, 0, reversal_in.data_sns_toolbox], [0, 0, 0]],
        #                                           device=device, dtype=dtype)

if __name__ == "__main__":
    img_size = [24,32]
    print(SNSBandpass(img_size))
    print(SNSMotionVisionEye(img_size,5))