from motion_vision_net import __calc_2d_field__
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import snstorch.modules as m

class ConvTest(nn.Module):
    def __init__(self, dt, amp_rel, std_cen, std_sur, shape_field, reversal_ex, reversal_in):
        super().__init__()
        self.dt = dt

        self.tau_fast = self.dt / (10 * self.dt)

        conductance, reversal, target = __calc_2d_field__(amp_rel, std_cen,std_sur, shape_field, reversal_ex, reversal_in, 'cpu')
        print(torch.sum(conductance*reversal))
        print(torch.sum(conductance))
        print(torch.sum(target))
        syn_in_bo_params = nn.ParameterDict({
            'conductance': nn.Parameter(conductance, requires_grad=False),
            'reversal': nn.Parameter(reversal, requires_grad=False)
        })
        self.target = target
        self.syn_input_bandpass_on = m.NonSpikingChemicalSynapseConv(1, 1, shape_field, conv_dim=2)
        self.syn_input_bandpass_on.params.update(syn_in_bo_params)
        self.syn_input_bandpass_on.setup()
        nrn_input_params = nn.ParameterDict({
            'tau': nn.Parameter((self.tau_fast + torch.zeros([1])), requires_grad=False),
            'leak': nn.Parameter(torch.ones([1]), requires_grad=False),
            'rest': nn.Parameter(torch.ones([1]), requires_grad=False),
            'bias': nn.Parameter(torch.zeros([1]), requires_grad=False),
            'init': nn.Parameter(torch.ones([1]), requires_grad=False)
        })
        self.input = m.NonSpikingLayer([1], params=nrn_input_params)

    def forward(self, x, state_input):
        syn = self.syn_input_bandpass_on(x, state_input)
        state_input = self.input(syn, state_input)
        return state_input

dt = 1/(30*13)*1000
num_steps = 100
shape_field = 5
net = ConvTest(dt, 0.0, 1,17.5, shape_field, 5, -2)

state_input = torch.ones([1])
x = torch.ones([shape_field,shape_field])
data = torch.zeros([2,num_steps])

for i in range(num_steps):
    state_input = net(x, state_input)
    data[0,i] = state_input

# print(torch.sum(net.target))
plt.figure()
plt.suptitle('Receptive Field')
plt.subplot(1,3,1)
plt.imshow(net.target, cmap='coolwarm', vmin=-0.5, vmax=0.5)
plt.title('Target')
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(net.syn_input_bandpass_on.params['conductance'])
plt.title('Conductance')
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(net.syn_input_bandpass_on.params['reversal'], cmap='coolwarm', vmin=-5, vmax=5)
plt.title('Reversal')
plt.colorbar()

plt.figure()
plt.plot(data[0,:].detach().numpy())
# plt.plot(data[1,:].detach().numpy())

plt.show()