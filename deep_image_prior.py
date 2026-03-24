import deepinv as dinv
import torch    
import numpy as np
import matplotlib.pyplot as plt
from utils import * 
from algos_dinv import MyPGD, NPN_PGD
import argparse
from deepinv.utils import plot
# Configs
from laplacians import (
    grid_laplacian,
    anisotropic_from_x0,
    laplacian_normalize,
    fractional_laplacian,
    learn_graph_from_batch,
    product_space_channel_laplacian,
    hypergraph_laplacian_from_patches,
    make_batch_from_x0,
)
from tqdm import tqdm

from other_models import UNet, UNetLeon
from utils_sr import set_seed, downsampling_matrix_torch,psnr_fun, null_projector
from pathlib import Path
from load_data import get_test_dataloader

DENOISER_LISTS = {'dncnn': dinv.models.DnCNN,
                  'restormer': dinv.models.Restormer,
                  'bm3d': dinv.models.BM3D,
                  'wavelet': dinv.models.WaveletDenoiser,
                  'scunet': dinv.models.SCUNet,
                  'swinir': dinv.models.SwinIR,
                  'drunet': dinv.models.DRUNet
                  }

DENOISER_ARGS = {
    'dncnn': {'in_channels': 3, 'out_channels': 3, 'pretrained': 'download_lipschitz'},
    'restormer': {'in_channels': 3, 'out_channels': 3, 'pretrained': 'denoising'},
    'bm3d': {},
    'wavelet': {'level': 4, 'wv': 'db8', 'non_linearity': 'soft'},
    'swinir': {},
    'scunet': {},
    'drunet': {},
    
}

def main(args):
    # Validate arguments
    
    
    c = 3 if not args.grayscale else 1
    n = args.n

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    testloader = get_test_dataloader(args)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    set_seed(seed=0)
   
    S_path = graph_matrix_path(args, args.variant, args.n, mode="sr", srf=args.srf)
    S = torch.load(S_path).to(device)
    H = downsampling_matrix_torch(n=n**2, s=args.srf, device=device)._to_dense()
    S = S[:int(S.shape[0]*args.p),]

    # Fix: Remove incorrect normalization that breaks compressed sensing theory
    m_h = H.shape[0]
    m_s = S.shape[0]

    spc_h = dinv.physics.CompressedSensing(m=m_h,img_size=(3,n,n),device=device,channelwise=True,fast=False)

    print('Registering H matrix...')
    _A_dagger = torch.linalg.pinv(H)
    spc_h.register_buffer("_A", H)
    spc_h.register_buffer("_A_dagger", _A_dagger)
    spc_h.register_buffer("_A_adjoint", spc_h._A.conj().T.type(spc_h.dtype).to(device))



    Pn = null_projector(H)

    if args.variant in {"grid", "grid4", "grid_4nn"}:
        L = grid_laplacian(args.n, device=device, dtype=torch.float32, kind="4nn")
    elif args.variant == "grid_8nn":
        L = grid_laplacian(args.n, device=device, dtype=torch.float32, kind="8nn")
    elif args.variant == 'identity':
        L = torch.eye(args.n**2, device=device, dtype=torch.float32)

    B = L.T@Pn@L
    B = B.type(torch.float32).to(device)
    Br = Pn@L

    exp_path = model_path(args, args.dataset, args.variant, args.p, 0.001, mode="sr", srf=args.srf)
    
    checkpoint = torch.load(exp_path, map_location=device)


    model = UNetLeon(n_channels=c, base_channel=64).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    dip_model = UNetLeon(n_channels=c,base_channel=32).to(device)
    dip_model.train()
        # Fix: Use correct measurement size for each sensor
    m_s = S.shape[0]  # Get actual measurement size from matrix
    spc_s = dinv.physics.CompressedSensing(m=m_s,img_size=(3,n,n),device=device,channelwise=True,fast=False)
    _A_dagger = torch.linalg.pinv(S)
    spc_s.register_buffer("_A", S)
    spc_s.register_buffer("_A_dagger", _A_dagger)
    spc_s.register_buffer("_A_adjoint", spc_s._A.conj().T.type(spc_s.dtype).to(device))
  
    x = next(iter(testloader)).to(device)


    y = spc_h(x)

    y_s = spc_s(model(spc_h.A_adjoint(y)) )

    y_s_gt = spc_s(x)

    # Verify closeness of predicted and true measurements

    print(f"Sensor  {psnr_fun(y_s, y_s_gt).item()}")


    optimizer_dip = torch.optim.Adam(dip_model.parameters(), lr=args.learning_rate)
    z = torch.randn_like(x).to(device)
    mse = torch.nn.MSELoss()
    loop = tqdm(range(args.num_epochs), desc="DIP Optimization") 
    psnrs = []
    mses = []
    x_hats = []
    for epoch in loop:
        optimizer_dip.zero_grad()
        x_hat = dip_model(z)
        # y_s_hat = spc_s(x_hat)
        loss = mse(y_s, spc_s(x_hat))*args.gamma_npn + mse(spc_h.A(x_hat), y) + torch.norm(x_hat.reshape(x_hat.shape[0],x_hat.shape[1],-1)@Br, dim=1).mean()*args.gamma_graph
        loss.backward(retain_graph=True)
        optimizer_dip.step()
        psnr = psnr_fun(x, x_hat).item()
        mse_r = mse(x, x_hat).item()
        loop.set_postfix(loss=loss.item(), psnr=psnr, mse=mse_r)
        if epoch %2 ==0:
            psnrs.append(psnr)
            mses.append(mse_r)
            x_hats.append(x_hat.detach().cpu().numpy())
        
    
    # save reconstructed images and PSNR values

    results_path = Path("RESULTS") / "PNP" / "SR" / f"metrics_DIP_gnpn_sr_{args.srf}_{args.variant}_p_{args.p}_gamma_{args.gamma_npn}_gramm_graph_{args.gamma_graph}.npz"
    np.savez(results_path, psnrs=np.array(psnrs), mses=np.array(mses), x_hats=np.array(x_hats))
    print(f"Saved results to {results_path} ")
        # if epoch % 1 == 0:} ")
        

    # plot_reconstructions(x, x_hat, x_hats, idx_imgs, psnrs, title="SPC Reconstruction using NPN-PGD",save_path=f'results/spc_npn_pgd_cr{cr_h}_sigma{args.sigma}.png')


parser = argparse.ArgumentParser(description="Single Pixel Camera Reconstruction")

# Physics parameters 
parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
parser.add_argument('--srf', type=int, default=4, help='Spatial resolution factor for downsampling matrix H')
parser.add_argument('--variant', type=str, default='grid_8nn', help='Variant of graph for S')
parser.add_argument('--p', type=float, default=0.1, help='Proportion of S to use')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for DIP optimization')
parser.add_argument('--num_epochs', type=int, default=5000, help='Number of epochs for DIP optimization')

# Data parameters
parser.add_argument('--debug', type=bool, default=True, help='Debug mode with limited data')
parser.add_argument('--n', type=int, default=128, help='Image size (n x n)')
parser.add_argument('--dataset', type=str, default='celeba', help='Dataset to use (mnist, fashionmnist, cifar10, BSDS500, CelebA, ct)')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loaders')
parser.add_argument('--grayscale', type=bool, default=False, help='Convert images to grayscale')

parser.add_argument('--gamma_npn', type=float, default=0.1, help='Regularization parameter for NPN-PGD') # set 0 to baseline DIP 
parser.add_argument('--gamma_graph', type=float, default=0.001, help='Graph regularization parameter for NPN-PGD')
parser.add_argument('--num_train_images', type=int, default=100, help='Number of training images to use in debug mode')
parser.add_argument('--base_channel', type=int, default=32, help='Base channel for UNet backbone')
parser.add_argument('--sigma_x', type=float, default=0.1, help='Noise level for data augmentation')
parser.add_argument('--use_test_data', dest='use_test_data', action='store_true', help='Load evaluation images from test_data/<dataset>.')
parser.add_argument('--no_test_data', dest='use_test_data', action='store_false', help='Load evaluation images from the full dataset test split.')
parser.set_defaults(use_test_data=True)

args = parser.parse_args()

main(args)
