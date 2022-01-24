# %% [markdown]
# # Normalized Autoencoders

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from tqdm import tqdm
import io
import PIL
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tensorboardX import SummaryWriter
from torchvision import transforms

from models.modules import FCNet, IsotropicGaussian, FCResNet
from models.ae import AE, VAE
from models.nae import NAE, FFEBM
from models.mmd import mmd

from loaders.synthetic import sample2d
from IPython.display import clear_output


def gen_plot():
    """Create a pyplot plot and save to buffer."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image)  # .unsqueeze(0)
    return image


tb_writer = SummaryWriter('../results/runs')
 

# %%
device = 'cuda:0'

# %%
dset = '8gaussians'
if dset == '8gaussians':
    xmin, xmax, ymin, ymax = [-4, 4, -4, 4]
elif dset == '2spirals':
    xmin, xmax, ymin, ymax = [-4, 4, -4, 4]
elif dset == 'checkerboard':
    xmin, xmax, ymin, ymax = [-4, 4, -4, 4]
batch_size = 200

# %%
XX, YY = torch.meshgrid(torch.linspace(xmin, xmax, 100), torch.linspace(ymin,ymax, 100))
grid = torch.cat([XX.reshape(-1,1), YY.reshape(-1,1)], dim=1)
grid_gpu = grid.to(device)

# %% [markdown]
# # On-Manifold Initialization

# %%
zdim = 2
encoder = FCResNet(2, zdim * 2, res_dim=256, n_res_hidden=1024, n_resblock=5, out_activation='linear')
decoder = FCResNet(zdim, 2, res_dim=256, n_res_hidden=1024, n_resblock=5, out_activation='linear')
vae = VAE(encoder, decoder, sigma_trainable=True, use_mean=False)

vae.to(device)

pretrained = None
if pretrained:
    print(f"> Loading pretrained from {pretrained}")
    ckpt = torch.load(pretrained, map_location=device)
    vae.load_state_dict(ckpt)
    warnings.warn("> Not resuming the optimizer")

opt = Adam([{'params': list(vae.encoder.parameters()) + list(vae.decoder.net.parameters())},
            {'params': vae.decoder.sigma}], lr=1e-3)
l_loss = []
l_kld = []
l_mmd = []
l_en_norm = []
l_T = []
l_sigma = []
l_temperature = []
l_pos = []; l_neg = []

# %%
n_iter = 5000
for i_iter in tqdm(range(n_iter)):
    batch_x = sample2d(dset, batch_size=batch_size)
    batch_x = torch.tensor(batch_x, dtype=torch.float, device=device)
    d_train = vae.train_step(batch_x, opt)
    l_loss.append(d_train['loss'])
    l_kld.append(d_train['vae/kl_loss_'].detach().cpu())
    l_sigma.append(vae.decoder.sigma.item())
    
    mmd_ = -1.  # mmd(batch_x, d_train['x_neg'].to(device)).item()
    l_mmd.append(mmd_)
    
    if i_iter % 10 == 0:
        batch_x = batch_x.cpu()
        with torch.no_grad():
            sample_x = vae.sample(128, device)['sample_x'].cpu()
        
        clear_output(wait=True)
        fig, axs = plt.subplots(ncols=6, figsize=(24,4))
        axs[0].plot(l_loss, label='loss'); axs[0].plot(l_kld, label='kld'); axs[0].set_title('loss')
        tb_writer.add_scalar('Loss_it/NLL', l_loss[-1], global_step=i_iter)
        tb_writer.add_scalar('Loss_it/KLD', l_kld[-1], global_step=i_iter)
        axs[0].legend()
        ax2 = axs[0].twinx()
        axs[1].plot(l_mmd); axs[1].set_title('mmd')
        axs[2].plot(l_sigma, label='sigma'); axs[2].set_title('sigma and T'); 
        ax2 = axs[2].twinx()
        axs[2].legend()
        axs[4].scatter(batch_x[:,0], batch_x[:,1]); axs[4].set_title('data')
        axs[4].scatter(sample_x[:,0], sample_x[:,1])
        E = -vae.marginal_likelihood(grid_gpu, n_sample=50).detach().cpu().reshape(100, 100)
        img = axs[5].imshow(np.exp(-E.T), origin='lower', extent=(-4, 4, -4, 4))
        fig.colorbar(img, ax=axs[5])
        plt.tight_layout()
        # plt.show()
        # plt.savefig('../results/vae_training_process.png')
        tb_writer.add_image('vae_training_process', gen_plot(), global_step=i_iter)
        plt.close()
        
torch.save(vae.state_dict(), f'../results/vae_{zdim}_{dset}.pth')
