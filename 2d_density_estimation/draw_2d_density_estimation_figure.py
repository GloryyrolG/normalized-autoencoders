# %% [markdown]
# # Visualize 2D synthetic density estimation result

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from models.modules import FCNet, IsotropicGaussian, FCResNet
from models.ae import AE, VAE
from models.nae import NAE
# from models.bnaf import BNAF
from loaders.synthetic import sample2d

# %%
device = 'cpu:0'

# %%
dset = '8gaussians'
if dset == '8gaussians':
    xmin, xmax, ymin, ymax = [-4, 4, -4, 4]
elif dset == '2spirals':
    xmin, xmax, ymin, ymax = [-4, 4, -4, 4]
elif dset == 'checkerboard':
    xmin, xmax, ymin, ymax = [-4, 4, -4, 4]


# %%
XX, YY = torch.meshgrid(torch.linspace(xmin, xmax, 100), torch.linspace(ymin,ymax, 100))
grid = torch.cat([XX.reshape(-1,1), YY.reshape(-1,1)], dim=1)
grid_gpu = grid
grid_gpu = grid.to(device)

# %% [markdown]
# # AE

# %%
zdim = 1
encoder = FCResNet(2, zdim, res_dim=256, n_res_hidden=1024, n_resblock=5, out_activation='spherical')
decoder = FCResNet(zdim, 2, res_dim=256, n_res_hidden=1024, n_resblock=5, out_activation='linear')
ae = AE(encoder, IsotropicGaussian(decoder, sigma=0.5, sigma_trainable=True, error_normalize=False))
ae.load_state_dict(torch.load('ae_1_8gaussians.pth'))
ae.to(device);

# %%
z_grid = ae.encoder(grid_gpu)
E_ae = - ae.decoder.log_likelihood(grid_gpu, z_grid).detach().cpu().reshape(100, 100)
Omega = ((8 / 100 * 8 / 100) * np.exp(-E_ae)).sum()
p_ae = np.exp(-E_ae.T)/Omega
print("> ae_1 is problematic")

# %%
zdim = 3
encoder = FCResNet(2, zdim, res_dim=256, n_res_hidden=1024, n_resblock=5, out_activation='tanh')
decoder = FCResNet(zdim, 2, res_dim=256, n_res_hidden=1024, n_resblock=5, out_activation='linear')
ae = AE(encoder, IsotropicGaussian(decoder, sigma=0.5, sigma_trainable=True, error_normalize=False))
ae.load_state_dict(torch.load('ae_3_8gaussians.pth'))
ae.to(device);

# %%
z_grid = ae.encoder(grid_gpu)
E_ae = - ae.decoder.log_likelihood(grid_gpu, z_grid).detach().cpu().reshape(100, 100)
Omega = ((8 / 100 * 8 / 100) * np.exp(-E_ae)).sum()
p_ae_3 = np.exp(-E_ae.T)/Omega

# %% [markdown]
# # VAE

# %%
zdim = 1
encoder = FCResNet(2, zdim * 2, res_dim=256, n_res_hidden=1024, n_resblock=5, out_activation='linear')
decoder = FCResNet(zdim, 2, res_dim=256, n_res_hidden=1024, n_resblock=5, out_activation='linear')
vae = VAE(encoder, decoder, sigma_trainable=True, use_mean=False)
vae.load_state_dict(torch.load('vae_1_8gaussians.pth'))
vae.to(device);

print(f"sigma {vae.decoder.sigma:.6f}")
print("vae", vae)

# Sampling VAEs.
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
with torch.no_grad():
    sample = vae.sample(1024, device)
axes[0].scatter(sample['sample_x'][:, 0], sample['sample_x'][:, 1], s=8)

# %%
vae.n_sample = 100
p_vae = np.exp(-vae.reconstruction_probability(grid_gpu).detach().cpu()).reshape((100,100))

# %%
gg = vae.marginal_likelihood(grid_gpu, n_sample=500).detach().cpu()
p_vae = torch.exp(gg.reshape(100, 100))

# %%
zdim = 3
encoder = FCResNet(2, zdim * 2, res_dim=256, n_res_hidden=1024, n_resblock=5, out_activation='linear')
decoder = FCResNet(zdim, 2, res_dim=256, n_res_hidden=1024, n_resblock=5, out_activation='linear')
vae = VAE(encoder, decoder, sigma_trainable=True, use_mean=False)
vae.load_state_dict(torch.load('vae_3_8gaussians.pth'))
vae.to(device);

# Sampling VAEs.
with torch.no_grad():
    sample = vae.sample(1024, device)
axes[1].scatter(sample['sample_x'][:, 0], sample['sample_x'][:, 1], s=8)
plt.savefig('../results/vae_sample_x.png')
plt.close()

# %%
vae.n_sample = 100
# p_vae_3 = np.exp(-vae.reconstruction_probability(grid_gpu).detach().cpu()).reshape((100,100))
gg = vae.marginal_likelihood(grid_gpu, n_sample=500).detach().cpu() #- np.log(1000)
p_vae_3 = torch.exp(gg.reshape(100, 100))

# %%
p_vae.flatten().numpy()

# %%
plt.hist(np.log(p_vae.flatten().numpy()))

# %% [markdown]
# # NAE

# %%
# zdim = 2
# res_dim = 200
# encoder = FCResNet(2, zdim, res_dim=res_dim, n_res_hidden=1024, n_resblock=5, out_activation='linear', use_spectral_norm=False)
# decoder = FCResNet(zdim, 2, res_dim=res_dim, n_res_hidden=1024, n_resblock=5, out_activation='linear', use_spectral_norm=False)
# nae = NAE(encoder, decoder, sampling='on_manifold',
#            x_step=30, x_stepsize=None, x_noise_std=0.1, x_bound=(-5, 5), x_clip_langevin_grad=None,
#            z_step=10, z_stepsize=None, z_noise_std=0.1, z_bound=None, z_clip_langevin_grad=None,
#            gamma=1, spherical=True,
#            temperature=0.5, temperature_trainable=True,
#            l2_norm_reg=None, l2_norm_reg_en=None, z_norm_reg=None,
#            initial_dist='gaussian', replay=True, replay_ratio=0.95, buffer_size=10000,
#            mh=True, mh_z=False, reject_boundary=True, reject_boundary_z=True)
# nae.load_state_dict(torch.load('nae_2_8gaussians.pth'))
# # nae.to(device);

# # %%
# E = nae.energy_T(grid_gpu).detach().cpu().reshape(100, 100)
# # E = nae.energy(grid_gpu).detach().cpu().reshape(100, 100)
# Omega = ((8 / 100 * 8 / 100) * np.exp(-E)).sum()
# p_nae = np.exp(-E.T)/Omega
p_nae = torch.zeros((100, 100))

# %%
zdim = 3
encoder = FCResNet(2, zdim, res_dim=256, n_res_hidden=1024, n_resblock=5, out_activation='linear', use_spectral_norm=False)
decoder = FCResNet(zdim, 2, res_dim=256, n_res_hidden=1024, n_resblock=5, out_activation='linear', use_spectral_norm=False)
nae = NAE(encoder, decoder, sampling='on_manifold',
           x_step=30, x_stepsize=None, x_noise_std=0.1, x_bound=(-5, 5), x_clip_langevin_grad=None,
           z_step=10, z_stepsize=None, z_noise_std=0.1, z_bound=None, z_clip_langevin_grad=None,
           gamma=1, spherical=False,
           temperature=0.1, temperature_trainable=True,
           l2_norm_reg=None, l2_norm_reg_en=None, z_norm_reg=0.01,
           initial_dist='gaussian', replay=True, replay_ratio=0.95, buffer_size=10000,
           mh=True, mh_z=False, reject_boundary=True, reject_boundary_z=True)
nae.load_state_dict(torch.load('nae_3_8gaussians.pth'))
# nae.to(device);

# %%
E = nae.energy_T(grid_gpu).detach().cpu().reshape(100, 100)
# E = nae.energy(grid_gpu).detach().cpu().reshape(100, 100)
Omega = ((8 / 100 * 8 / 100) * np.exp(-E)).sum()
p_nae_3 = np.exp(-E.T)/Omega

# %% [markdown]
# # BNAF
# 
# * 저장했던 파일을 불러옴

# %%
# bnaf = BNAF(1,3, 2, 50)
# bnaf.load_state_dict(torch.load('bnaf_8gaussians.pth'))
# bnaf.to(device)

# %%
# model = bnaf
# gg = model.log_likelihood(grid.to(device)).detach().cpu()
# prd = gg.reshape(100, 100)
# bnaf_pred = prd

# %%
# plt.imshow(np.exp(bnaf_pred), origin='upper', extent=(-4, 4, -4, 4))
# plt.colorbar()
# # plt.xticks(np.linspace(-4,4,100));

# %% [markdown]
# # Gaussian

# %%
from torch.distributions import Normal

# %%
g1 = Normal(torch.tensor([4/np.sqrt(2),0]), torch.tensor([0.5/np.sqrt(2), 0.5/np.sqrt(2)]))
g2 = Normal(torch.tensor([-4/np.sqrt(2),0]), torch.tensor([0.5/np.sqrt(2), 0.5/np.sqrt(2)]))
g3 = Normal(torch.tensor([0,4/np.sqrt(2)]), torch.tensor([0.5/np.sqrt(2), 0.5/np.sqrt(2)]))
g4 = Normal(torch.tensor([0,-4/np.sqrt(2)]), torch.tensor([0.5/np.sqrt(2), 0.5/np.sqrt(2)]))
g5 = Normal(torch.tensor([2,2]), torch.tensor([0.5/np.sqrt(2), 0.5/np.sqrt(2)]))
g6 = Normal(torch.tensor([-2,2]), torch.tensor([0.5/np.sqrt(2), 0.5/np.sqrt(2)]))
g7 = Normal(torch.tensor([2,-2]), torch.tensor([0.5/np.sqrt(2), 0.5/np.sqrt(2)]))
g8 = Normal(torch.tensor([-2,-2]), torch.tensor([0.5/np.sqrt(2), 0.5/np.sqrt(2)]))

# %%
p1 = torch.exp(g1.log_prob(grid).sum(dim=1))
p2 = torch.exp(g2.log_prob(grid).sum(dim=1))
p3 = torch.exp(g3.log_prob(grid).sum(dim=1))
p4 = torch.exp(g4.log_prob(grid).sum(dim=1))
p5 = torch.exp(g5.log_prob(grid).sum(dim=1))
p6 = torch.exp(g6.log_prob(grid).sum(dim=1))
p7 = torch.exp(g7.log_prob(grid).sum(dim=1))
p8 = torch.exp(g8.log_prob(grid).sum(dim=1))

# %%
p_8gaussian = (p1 + p2 + p3 + p4 + p5+ p6+ p7 + p8) / 8

# %%
plt.imsave('../results/p_8gaussian.png', p_8gaussian.reshape(100,100))

# %% [markdown]
# # Figure drawing

# %%
from mpl_toolkits.axes_grid1 import ImageGrid

# %%
p_vae.max()

# %%
p_vae_3.max()

# %%
p_nae_3.max()

# %%
p_8gaussian.max()

# %%
print(f"p_ae.max() {p_ae.max()}")

# %%
cat_p = np.concatenate([p_ae.flatten(), p_vae.flatten(), p_nae.flatten(), p_ae_3.flatten(), p_nae_3.flatten()]) # p_vae_3.flatten(),
p_max = cat_p.max()

# %%
p_max = p_8gaussian.max()

# %%
plt.rcParams.update({'font.size': 25})
fig = plt.figure(constrained_layout=True, figsize=(16,8))
spec = fig.add_gridspec(ncols=4, nrows=2,)
ax = fig.add_subplot(spec[:,0])
im = ax.imshow(p_8gaussian.reshape(100,100), origin='lower', extent=(-4, 4, -4, 4), cmap='Reds',)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('Data Distribution')
plt.colorbar(im, ax=ax, shrink=0.47)

ax = fig.add_subplot(spec[0,1])
ax.imshow(p_ae, origin='lower', extent=(-4, 4, -4, 4), cmap='Reds', vmin=0, vmax=p_max)
ax.set_title('AE ($D_Z$=1)')
ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(spec[0,2])
ax.imshow(p_vae, origin='lower', extent=(-4, 4, -4, 4), cmap='Reds', vmin=0, vmax=p_max)
ax.set_title('VAE ($D_Z$=1)')
ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(spec[0,3])
im = ax.imshow(p_nae, origin='lower', extent=(-4, 4, -4, 4), cmap='Reds', vmin=0, vmax=p_max)
ax.set_title('NAE ($D_Z$=2)')
ax.set_xticks([]); ax.set_yticks([])


ax = fig.add_subplot(spec[1,1])
ax.imshow(p_ae_3, origin='lower', extent=(-4, 4, -4, 4), cmap='Reds', vmin=0, vmax=p_max)
ax.set_title('AE ($D_Z$=3)')
ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(spec[1,2])
ax.imshow(p_vae_3, origin='lower', extent=(-4, 4, -4, 4), cmap='Reds', vmin=0, vmax=p_max)
ax.set_title('VAE ($D_Z$=3)')
ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(spec[1,3])
ax.imshow(p_nae_3, origin='lower', extent=(-4, 4, -4, 4), cmap='Reds', vmin=0, vmax=p_max)
ax.set_title('NAE ($D_Z$=3)')
ax.set_xticks([]); ax.set_yticks([])
# plt.savefig('../results/fig_2d_density_estimation.pdf', bbox_inches='tight')
plt.savefig('../results/fig_2d_density_estimation.png', bbox_inches='tight')


