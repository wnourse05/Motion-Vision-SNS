import torch
import matplotlib.pyplot as plt
import numpy as np

def draw_gauss(shape_field, std_cen, amp_rel, std_sur):
    axis = torch.tensor(np.arange(-(5 * (shape_field - 1) / 2), 5 * ((shape_field - 1) / 2 + 1), 5))
    coeff_center = 1 / (std_cen * torch.sqrt(torch.tensor([2 * torch.pi])))
    coeff_surround = amp_rel / (std_sur * torch.sqrt(torch.tensor([2 * torch.pi])))
    target = torch.zeros([shape_field, shape_field])
    for i in range(shape_field):
        for j in range(shape_field):
            target[i,j] = -1 * coeff_center * torch.exp(-(axis[i] ** 2 + axis[j] ** 2) / (2 * std_cen**2)) + coeff_surround * torch.exp(
                -(axis[i] ** 2 + axis[j] ** 2) / (2 * std_sur**2))
    return target

shape_field = 5
bo_std_cen = 2.7
bo_amp_rel = 0.012
bo_std_sur = 17.5
l_std_cen = 2.5
l_amp_rel = 0.2
l_std_sur = 6.4
bf_std_cen = 2.9
bf_amp_rel = 0.013
bf_std_sur = 12.4

field_bo = draw_gauss(shape_field, bo_std_cen, bo_amp_rel, bo_std_sur)
field_l = draw_gauss(shape_field, l_std_cen, l_amp_rel, l_std_sur)
field_bf = draw_gauss(shape_field, bf_std_cen, bf_amp_rel, bf_std_sur)
max_val = torch.max(torch.tensor([torch.max(field_bo), torch.max(field_l), torch.max(field_bf)]))
min_val = torch.min(torch.tensor([torch.min(field_bo), torch.min(field_l), torch.min(field_bf)]))
true_max = torch.max(torch.tensor([torch.abs(max_val), torch.abs(min_val)]))

plt.figure()
plt.subplot(1,3,1)
plt.imshow(field_bo, cmap='bwr', vmin=-0.15, vmax=0.15)
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(field_l, cmap='bwr', vmin=-0.15, vmax=0.15)
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(field_bf, cmap='bwr', vmin=-0.15, vmax=0.15)
plt.colorbar()
plt.show()