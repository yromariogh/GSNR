import deepinv as dinv
import torch    
import numpy as np
import matplotlib.pyplot as plt
from utils import * 
from algos_dinv import MyPGD, NPN_PGD
import argparse
from deepinv.utils import plot
# Configs
from tqdm import tqdm
from deepinv.utils.tensorlist import randn_like, TensorList

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
# noise schedule of the algorithm


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
    # _A_dagger = torch.linalg.pinv(H)
    spc_h.register_buffer("_A", H)
    # spc_h.register_buffer("_A_dagger", _A_dagger)
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


    exp_path = model_path(args, args.dataset, args.variant, args.p, 0.001, mode="sr", srf=args.srf)
    
    checkpoint = torch.load(exp_path, map_location=device)


    model = UNetLeon(n_channels=c, base_channel=64).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

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

    data_fidelity = dinv.optim.L2()

    
    
    
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    
    max_iter = args.iters
    zeta = 0.999
    T = 1000

    def get_alphas(beta_start=0.01 / 1000, beta_end=20 / 1000, num_train_timesteps=T):
        betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        betas = torch.from_numpy(betas).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)  # This is \overline{\alpha}_t
        return torch.tensor(alphas_cumprod)

    alphas_cumprod = get_alphas()
    sigmas = torch.sqrt(1.0 - alphas_cumprod) / alphas_cumprod.sqrt()
    sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    reduced_alpha_cumprod = torch.div(
        sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod
    )  # equivalent noise sigma on image
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)

    def get_noise_schedule(sigma, lambda_=10.0, num_train_timesteps=1000, max_iter=max_iter):
        sigmas = []
        sigma_ks = []
        rhos = []
        for i in range(num_train_timesteps):
            sigmas.append(reduced_alpha_cumprod[num_train_timesteps - 1 - i])
            sigma_ks.append((sqrt_1m_alphas_cumprod[i] / sqrt_alphas_cumprod[i]))
            rhos.append(lambda_ * (sigma**2) / (sigma_ks[i] ** 2))
        rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)

        seq = np.sqrt(np.linspace(0, num_train_timesteps**2, max_iter))
        seq = [int(s) for s in list(seq)]
        seq[-1] = seq[-1] - 1

        return rhos, sigmas, seq
    
    sigma_noise = args.sigma_x
     # noise schedule   
    rhos, sigmas, seq = get_noise_schedule(sigma_noise)
    max_iter = args.iters
    


    # Initialization
    x_hat = 2 * spc_h.A_dagger(y) - 1  # Rescale
    x_hat = (
        x_hat + (sigmas[seq[0]] ** 2 - 4 * sigma_noise**2).sqrt() * torch.randn_like(x_hat)
    ) / sqrt_recip_alphas_cumprod[
        -1
    ]  # Add noise (simpler than the original code, may be suboptimal)
    model = dinv.models.DiffUNet(large_model=True).to(device)

    # Images to save for visualization
    list_denoised, list_prox, list_noisy = [], [], []
    save_steps = [0, 1, 2, 5, 10, 20, 29]

   
    with torch.no_grad():
        loop = tqdm(range(len(seq)),desc='DIFFPIR Reconstruction')
        for i in loop:

            sigma_cur = sigmas[seq[i]]

            # time step associated with the noise level sigmas[i]
            t_i = find_nearest(reduced_alpha_cumprod, sigma_cur.cpu().numpy())
            at = 1 / sqrt_recip_alphas_cumprod[t_i] ** 2

            # Denoising step
            x_aux = x_hat / (2 * at.sqrt()) + 0.5  # renormalize in [0, 1]
            out = model(x_aux, sigma_cur / 2)
            denoised = 2 * out - 1  # back to [-1, 1]
            x0 = denoised.clamp(-1,1)  # optional

            if not seq[i] == seq[-1]:
                # 2. Data fidelity step (augmented with auxiliary sensor fidelity)
                
                x0 = data_fidelity.prox(x0, y, spc_h, gamma=1/ (2 * rhos[t_i]))

                if args.use_graphS:
                    x0 = data_fidelity.prox(x0, y_s, spc_s, gamma=1e-6 / (2 * rhos[t_i]))


                # 3. Sampling step
                next_sigma = sigmas[T - 1 - seq[i + 1]].cpu().numpy()
                t_im1 = find_nearest(
                    sigmas.cpu().numpy(), next_sigma
                )  # time step associated with the next noise level

                eps = (x_hat - alphas_cumprod[t_i].sqrt() * x0) / torch.sqrt(
                    1.0 - alphas_cumprod[t_i]
                )  # effective noise

                x_hat = alphas_cumprod[t_im1].sqrt() * x0 + torch.sqrt(
                    1.0 - alphas_cumprod[t_im1]
                ) * (np.sqrt(1 - zeta) * eps + np.sqrt(zeta) * torch.randn_like(x_hat))
            # Print PSNR every 50 iterations
            x_print = (x_hat - x_hat.min()) / (x_hat.max() - x_hat.min())
            loop.set_postfix({"psnr": psnr_fun(x_print, x).item()})
            
            # print(f"Iteration {i}, PSNR: {psnr_fun(x_print, x).item()}")
            if i in save_steps:
                list_noisy.append(x_aux)
                list_denoised.append(denoised)
                list_prox.append(x0)

   
    from torchmetrics import StructuralSimilarityIndexMeasure
    x_hat = (x_hat - x_hat.min()) / (x_hat.max() - x_hat.min())
    psnr_v = psnr_fun(x_hat, x)
    ssim = StructuralSimilarityIndexMeasure().to(device)(x_hat, x).item()
    print(f"DIFFPIR PSNR: {psnr_v}, SSIM: {ssim}")

    psnr_val_lin = psnr_fun(spc_h.A_adjoint(y), x)
    ssim_lin = StructuralSimilarityIndexMeasure().to(device)(spc_h.A_adjoint(y), x).item()
    print(f"Linear PSNR: {psnr_val_lin}, SSIM: {ssim_lin}")
    # Plotting the results
    plot(
        {
            f"Measurement P": y.reshape(1,3,n//args.srf,n//args.srf),
            f"Model Output (PSNR {psnr_v:.2f}, SSIM {ssim:.3f})": x_hat,
            "Ground Truth": x,
        },


        save_dir=Path("RESULTS") / "PNP" / "SR" / f"visual_diffpir_sr_{args.srf}_{args.variant}_p_{args.p}_base.svg"
    )


parser = argparse.ArgumentParser(description="Single Pixel Camera Reconstruction")

# Physics parameters 
parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
parser.add_argument('--srf', type=int, default=4, help='Spatial resolution factor for downsampling matrix H')
parser.add_argument('--variant', type=str, default='identity', help='Variant of graph for S')
parser.add_argument('--p', type=float, default=0.1, help='Proportion of S to use')
parser.add_argument('--use_graphS', type=bool, default=True, help='Whether to use graph-based regularization in the algorithm')

# Data parameters
parser.add_argument('--debug', type=bool, default=True, help='Debug mode with limited data')
parser.add_argument('--n', type=int, default=128, help='Image size (n x n)')
parser.add_argument('--dataset', type=str, default='celeba', help='Dataset to use (mnist, fashionmnist, cifar10, BSDS500, CelebA, ct)')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loaders')
parser.add_argument('--grayscale', type=bool, default=False, help='Convert images to grayscale')

parser.add_argument('--denoiser', type=str, default='swinir', choices=list(DENOISER_LISTS.keys()), help='Denoiser to use in PnP-PGD')
parser.add_argument('--iters', type=int, default=50, help='Number of iterations for PGD methods')
parser.add_argument('--gamma_graph', type=float, default=0.00001, help='Graph regularization parameter for NPN-PGD')
parser.add_argument('--lambda_secondary', type=float, default=1.2, help='Regularization weight for auxiliary sensor fidelity.')
# Model parameters
parser.add_argument('--base_channel', type=int, default=32, help='Base channel for UNet backbone')
parser.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs')
parser.add_argument('--sigma_x', type=float, default=1e-4, help='Noise level for data augmentation')
# parser.add_argument('--debug', type=bool, default=True, help='Regularization parameter for graph-based regularization')
parser.add_argument('--num_train_images', default=100, action='store_true', help='Run demo mode')
parser.add_argument('--use_test_data', dest='use_test_data', action='store_true', help='Load evaluation images from test_data/<dataset>.')
parser.add_argument('--no_test_data', dest='use_test_data', action='store_false', help='Load evaluation images from the full dataset test split.')
parser.set_defaults(use_test_data=True)
args = parser.parse_args()
for variant in ['grid_8nn']:
    args.variant=variant
    main(args)









