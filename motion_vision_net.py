from snstorch import modules as m
import torch
import torch.nn as nn

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
            'input_fast_conductance': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'input_fast_reversal': nn.Parameter(
                (torch.rand(1, dtype=dtype, generator=generator) * 2.0 - 1.0).to(device)),
            'input_slow_conductance': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'input_slow_reversal': nn.Parameter(
                (torch.rand(1, dtype=dtype, generator=generator) * 2.0 - 1.0).to(device)),
            'fast_output_conductance': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'fast_output_reversal': nn.Parameter(
                (torch.rand(1, dtype=dtype, generator=generator) * 2.0 - 1.0).to(device)),
            'slow_output_conductance': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'slow_output_reversal': nn.Parameter(
                (torch.rand(1, dtype=dtype, generator=generator) * 2.0 - 1.0).to(device))
        })
        if params is not None:
            self.params.update(params)
        self.input = m.NonSpikingLayer(shape, device=device, dtype=dtype)
        self.syn_input_fast = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.fast = m.NonSpikingLayer(shape, device=device, dtype=dtype)
        self.syn_input_slow = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.slow = m.NonSpikingLayer(shape, device=device, dtype=dtype)
        self.syn_fast_output = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.syn_slow_output = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.output = m.NonSpikingLayer(shape, device=device, dtype=dtype)

    def forward(self, x, states):
        state_input = states[0]
        state_fast = states[1]
        state_slow = states[2]
        state_output = states[3]

        input2fast = self.syn_input_fast(state_input, state_fast)
        input2slow = self.syn_input_slow(state_input, state_slow)
        fast2out = self.syn_fast_output(state_fast, state_output)
        slow2out = self.syn_slow_output(state_slow, state_output)

        state_input = (x, state_input)
        state_fast = self.fast(input2fast, state_fast)
        state_slow = self.slow(input2slow, state_slow)
        state_output = self.output(fast2out+slow2out, state_output)
        return [state_input, state_fast, state_slow, state_output]

class SNSMotionVisionEye(nn.Module):
    def __init__(self, shape_input, shape_field, reversal_ex=torch.tensor([2.0]), reversal_in=torch.tensor([-2.0]),
                 reversal_mod=torch.tensor([0.0]), device=None, dtype=torch.float32):
        super().__init__()
        if device is None:
            device = 'cpu'

        # Network
        self.reversal_ex = nn.Parameter(reversal_ex.to(device))
        self.reversal_in = nn.Parameter(reversal_in.to(device))
        self.reversal_mod = nn.Parameter(reversal_mod.to(device))

        # Retina
        self.input = m.NonSpikingLayer(shape_input, device=device, dtype=dtype)

        # Lamina
        shape_post_conv = [x - (shape_field - 1) for x in shape_input]
        # Bo
        self.syn_input_bandpass_on = m.NonSpikingChemicalSynapseConv(1,1,shape_field)
        self.bandpass_on = SNSBandpass(shape_post_conv, device=device, dtype=dtype)
        # L
        self.syn_input_lowpass = m.NonSpikingChemicalSynapseConv(1, 1, shape_field)
        self.lowpass = m.NonSpikingLayer(shape_post_conv, device=device, dtype=dtype)
        # Bf
        self.syn_input_bandpass_off = m.NonSpikingChemicalSynapseConv(1, 1, shape_field)
        self.bandpass_off = SNSBandpass(shape_post_conv, device=device, dtype=dtype)

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

        # Synaptic updates
        syn_input_bandpass_on = self.syn_input_bandpass_on(state_input, state_bp_on_input)
        syn_input_lowpass = self.syn_input_lowpass(state_input, state_lowpass)
        syn_input_bandpass_off = self.syn_input_bandpass_off(state_input, state_bp_off_input)

        # Neural updates
        state_input = self.input(x, state_input)

        [state_bp_on_input, state_bp_on_fast, state_bp_on_slow, state_bp_on_output] = self.bandpass_on(
            syn_input_bandpass_on, [state_bp_on_input, state_bp_on_fast, state_bp_on_slow, state_bp_on_output])
        state_lowpass = self.lowpass(syn_input_lowpass, state_lowpass)
        [state_bp_off_input, state_bp_off_fast, state_bp_off_slow, state_bp_off_output] = self.bandpass_off(
            syn_input_bandpass_off, [state_bp_off_input, state_bp_off_fast, state_bp_off_slow, state_bp_off_output])

        return [state_input, state_bp_on_input, state_bp_on_fast, state_bp_on_slow, state_bp_on_output,
                state_lowpass, state_bp_off_input, state_bp_off_fast, state_bp_off_slow, state_bp_off_output]

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