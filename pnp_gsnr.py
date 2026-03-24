import argparse
from pathlib import Path

import deepinv as dinv
import matplotlib.pyplot as plt
import numpy as np
import torch
from deepinv.utils import plot

from algos_dinv import MyPGD, NPN_PGD
from laplacians import grid_laplacian, laplacian_normalize
from load_data import get_test_dataloader
from other_models import UNetLeon
from utils import (
    demosaicing_h_matrix_path,
    graph_matrix_path,
    model_path,
    normalize_task,
    task_results_dir,
)
from utils_sr import downsampling_matrix_torch, null_projector, psnr_fun, set_seed


DENOISER_LISTS = {
    "dncnn": dinv.models.DnCNN,
    "restormer": dinv.models.Restormer,
    "bm3d": dinv.models.BM3D,
    "wavelet": dinv.models.WaveletDenoiser,
    "scunet": dinv.models.SCUNet,
    "swinir": dinv.models.SwinIR,
    "drunet": dinv.models.DRUNet,
    "diffunet": dinv.models.DiffUNet,
}

DENOISER_ARGS = {
    "dncnn": {"in_channels": 3, "out_channels": 3, "pretrained": "download_lipschitz"},
    "restormer": {"in_channels": 3, "out_channels": 3, "pretrained": "denoising"},
    "bm3d": {},
    "wavelet": {"level": 4, "wv": "db8", "non_linearity": "soft"},
    "swinir": {},
    "scunet": {},
    "drunet": {},
    "diffunet": {"in_channels": 3, "out_channels": 3, "pretrained": "download"},
}

TASK_DEFAULTS = {
    "sr": {
        "dataset": "celeba",
        "batch_size": 100,
        "n": 128,
        "grayscale": False,
        "variant": "identity", # Select identity for base NPN, try grid_4nn and grid_8nn for GSNR variants
        "p": 0.1,
        "iters": 1000,
        "lambd": 0.0001,
        "gamma_npn": 0.001,
        "gamma_graph": 0.01,
        "sigma_x": 0.00,
        "num_train_images": 10,
    },
    "demo": {
        "dataset": "celeaba",
        "batch_size": 1,
        "n": 64,
        "grayscale": False,
        "variant": "identity",
        "p": 0.5,
        "iters": 3000,
        "lambd": 0.001,
        "gamma_npn": 0.001,
        "gamma_graph": 0.001,
        "sigma_x": 0.00,
        "num_train_images": 10,
    },
}


def apply_task_defaults(args):
    task = normalize_task(args.task)
    for key, value in TASK_DEFAULTS[task].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.task = task
    return args


def build_laplacian(variant: str, n: int, device, dtype):
    if variant in {"grid", "grid4", "grid_4nn"}:
        return grid_laplacian(n, device=device, dtype=dtype, kind="4nn")
    if variant == "grid_8nn":
        return grid_laplacian(n, device=device, dtype=dtype, kind="8nn")
    if variant == "identity":
        return torch.eye(n**2, device=device, dtype=dtype)
    if variant == "sym":
        return laplacian_normalize(
            grid_laplacian(n, device=device, dtype=dtype, kind="4nn"),
            mode="sym",
        )
    if variant == "rw":
        return laplacian_normalize(
            grid_laplacian(n, device=device, dtype=dtype, kind="4nn"),
            mode="rw",
        )
    raise ValueError(f"Unsupported variant '{variant}'.")


def build_primary_operator(args, device, channels):
    task = args.task
    if task == "sr":
        H = downsampling_matrix_torch(n=args.n**2, s=args.srf, device=device)._to_dense()
        m_h = H.shape[0]
        spc_h = dinv.physics.CompressedSensing(
            m=m_h,
            img_size=(channels, args.n, args.n),
            device=device,
            channelwise=True,
            fast=False,
        )
        spc_h.register_buffer("_A", H)
        spc_h.register_buffer("_A_dagger", torch.linalg.pinv(H))
        spc_h.register_buffer("_A_adjoint", spc_h._A.conj().T.type(spc_h.dtype).to(device))
        spc_h.noise_model = dinv.physics.GaussianNoise(sigma=args.sigma_x)
        return H, spc_h

    H = torch.load(demosaicing_h_matrix_path(args, args.n), map_location=device).to(device)
    spc_h = dinv.physics.Demosaicing(img_size=(channels, args.n, args.n), pattern="bayer").to(device)
    spc_h.noise_model = dinv.physics.GaussianNoise(sigma=args.sigma_x)
    return H, spc_h


def build_secondary_operator(args, device, channels, S):
    channelwise = args.task == "sr"
    spc_s = dinv.physics.CompressedSensing(
        m=S.shape[0],
        img_size=(channels, args.n, args.n),
        device=device,
        channelwise=channelwise,
        fast=False,
    )
    spc_s.register_buffer("_A", S)
    spc_s.register_buffer("_A_dagger", torch.linalg.pinv(S))
    spc_s.register_buffer("_A_adjoint", spc_s._A.conj().T.type(spc_s.dtype).to(device))
    return spc_s


def build_graph_term(args, device, channels, H):
    laplacian = build_laplacian(args.variant, args.n, device=device, dtype=torch.float32)
    if args.task == "demo":
        laplacian = torch.kron(torch.eye(channels, device=device, dtype=torch.float32), laplacian)
    projector = null_projector(H)
    if args.task == "sr":
        return (projector @ laplacian @ projector).type(torch.float32).to(device)
    return (laplacian.T @ projector @ laplacian).type(torch.float32).to(device)


def build_denoiser(args, device):
    denoiser = DENOISER_LISTS[args.denoiser](**DENOISER_ARGS[args.denoiser]).to(device)
    if args.equivariant:
        rotate = dinv.transform.Rotate(multiples=90, positive=True, n_trans=args.n_trans)
        transform = dinv.transform.Reflect(dim=[-1], n_trans=args.n_trans) * rotate
        denoiser = dinv.models.EquivariantDenoiser(denoiser, transform).to(device)
    return dinv.optim.PnP(denoiser=denoiser)


def main(args):
    args = apply_task_defaults(args)
    channels = 1 if args.grayscale else 3
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    testloader = get_test_dataloader(args)
    set_seed(seed=0)

    S = torch.load(
        graph_matrix_path(
            args,
            args.variant,
            args.n,
            mode=args.task,
            srf=args.srf if args.task == "sr" else None,
        ),
        map_location=device,
    ).to(device)
    S = S[: int(S.shape[0] * args.p), :]

    H, spc_h = build_primary_operator(args, device, channels)
    spc_s = build_secondary_operator(args, device, channels, S)
    graph_term = build_graph_term(args, device, channels, H)

    checkpoint = torch.load(
        model_path(
            args,
            args.dataset,
            args.variant,
            args.p,
            0.001,
            mode=args.task,
            srf=args.srf if args.task == "sr" else None,
        ),
        map_location=device,
    )

    predictor = UNetLeon(n_channels=channels, base_channel=64).to(device)
    predictor.load_state_dict(checkpoint)
    predictor.eval()

    x = next(iter(testloader))
    x = x[0].to(device) if isinstance(x, (list, tuple)) else x.to(device)
    if args.task == "sr":
        x = x[args.idx_imgs, :, :, :].unsqueeze(0)

    y = spc_h(x)
    y_s = spc_s(predictor(spc_h.A_adjoint(y)))
    y_s_gt = spc_s(x)

    data_fidelity = dinv.optim.L2()
    prior = build_denoiser(args, device)
    stepsize = args.alpha / spc_h.compute_norm(spc_h.A_adjoint(y), tol=1e-3).item()

    base_solver = MyPGD(
        data_fidelity=data_fidelity,
        prior=prior,
        stepsize=stepsize,
        lambd=args.lambd,
        max_iter=args.iters,
    )
    graph_solver = NPN_PGD(
        data_fidelity=data_fidelity,
        prior=prior,
        stepsize=stepsize,
        lambd=args.lambd,
        max_iter=args.iters,
        gamma=args.gamma_npn,
        gamma_decay=1.0,
        gamma_min=1e-4,
        gamma_graph=args.gamma_graph,
    )

    x0 = spc_h.A_dagger(y)
    x0_graph = x0 if args.task == "sr" else x0 + spc_s.A_dagger(y_s)
    x_hat_graph, x_hats_graph = graph_solver(
        x0_graph,
        y,
        y_s,
        spc_h,
        spc_s,
        xgt=x,
        B=graph_term,
        channel_wise=args.task == "demo",
    )
    x_hat_base, x_hats_base = base_solver(x0, y, spc_h, xgt=x)

    mse = torch.nn.MSELoss()
    psnrs_base = [psnr_fun(x_k, x).item() for x_k in x_hats_base]
    psnrs_graph = [psnr_fun(x_k, x).item() for x_k in x_hats_graph]
    mses_base = [mse(x_k, x).item() for x_k in x_hats_base]
    mses_graph = [mse(x_k, x).item() for x_k in x_hats_graph]

    print(f"Sensor PSNR: {psnr_fun(y_s, y_s_gt).item():.4f}")
    print(f"Final PGD MSE: {mses_base[-1]:.6f}")
    print(f"Final NPN-PGD MSE: {mses_graph[-1]:.6f}")
    print(f"Final PGD PSNR: {psnrs_base[-1]:.4f}")
    print(f"Final NPN-PGD PSNR: {psnrs_graph[-1]:.4f}")

    output_dir = Path("RESULTS") / "PNP" / task_results_dir(args.task)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(psnrs_base)
    plt.plot(psnrs_graph)
    plt.legend(["PnP-PGD", "NPN-PGD"])
    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.title("PSNR vs Iteration")
    plt.grid()
    curve_path = output_dir / (
        f"{args.dataset}_pnp_gnpn_{args.task}_{args.variant}_p_{args.p}"
        f"_denoiser_{args.denoiser}_lambda_{args.lambd}_alpha_{args.alpha}.svg"
    )
    plt.savefig(curve_path)
    plt.close()

    if args.task == "sr":
        titles = [
            r"$\mathbf{H}^\top y$",
            "Ground Truth",
            f"PnP-PGD PSNR={psnrs_base[-1]:.2f}",
            f"NPN-PGD PSNR={psnrs_graph[-1]:.2f}",
        ]
        images = [x0[0].cpu(), x[0].cpu(), x_hat_base[0].cpu(), x_hat_graph[0].cpu()]
    else:
        titles = [
            "Ground Truth",
            f"PnP-PGD PSNR={psnrs_base[-1]:.2f}",
            f"NPN-PGD PSNR={psnrs_graph[-1]:.2f}",
        ]
        images = [x[0].cpu(), x_hat_base[0].cpu(), x_hat_graph[0].cpu()]

    visual_path = output_dir / (
        f"{args.dataset}_visual_pnp_gnpn_{args.task}_{args.variant}_p_{args.p}"
        f"_denoiser_{args.denoiser}_lambda_{args.lambd}_alpha_{args.alpha}_{args.idx_imgs}_{args.gamma_npn}.svg"
    )
    plot(images, titles, suptitle=f"Reconstructions p={args.p}, variant={args.variant}", save_dir=visual_path, close=True, show=False)

    metrics_path = output_dir / (
        f"{args.dataset}_metrics_pnp_gnpn_{args.task}_{args.variant}_p_{args.p}"
        f"_denoiser_{args.denoiser}_lambda_{args.lambd}_alpha_{args.alpha}"
        f"_equiv_{args.equivariant}_gamma_{args.gamma_npn}_graph_{args.gamma_graph}_sigma_{args.sigma_x}.npz"
    )
    np.savez(
        metrics_path,
        psnrs_pnp=np.array(psnrs_base, dtype=float),
        psnrs_npn=np.array(psnrs_graph, dtype=float),
        mses_pnp=np.array(mses_base, dtype=float),
        mses_npn=np.array(mses_graph, dtype=float),
    )
    print(f"Saved results to {metrics_path}")


def build_parser(default_task="sr"):
    parser = argparse.ArgumentParser(description="PnP reconstruction for SR or demosaicing.")
    parser.add_argument("--task", type=str, default=default_task, choices=["sr", "demosaicing", "demo"])
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--srf", type=int, default=4, help="Spatial resolution factor for SR")
    parser.add_argument("--variant", type=str, default=None, help="Variant of graph for S")
    parser.add_argument("--p", type=float, default=None, help="Proportion of S to use")
    parser.add_argument("--debug", action="store_true", help="Debug mode with limited data")
    parser.add_argument("--n", type=int, default=None, help="Image size (n x n)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for data loaders")
    parser.add_argument("--grayscale", dest="grayscale", action="store_true", help="Convert images to grayscale")
    parser.add_argument("--color", dest="grayscale", action="store_false", help="Use RGB images")
    parser.set_defaults(grayscale=None)
    parser.add_argument("--denoiser", type=str, default="dncnn", choices=list(DENOISER_LISTS.keys()))
    parser.add_argument("--lambd", type=float, default=None, help="Regularization parameter for PnP-PGD")
    parser.add_argument("--gamma_npn", type=float, default=None, help="Regularization parameter for NPN-PGD")
    parser.add_argument("--alpha", type=float, default=1.9, help="Stepsize scaling factor for PGD methods")
    parser.add_argument("--iters", type=int, default=None, help="Number of iterations for PGD methods")
    parser.add_argument("--gamma_graph", type=float, default=None, help="Graph regularization parameter")
    parser.add_argument("--equivariant", default=True, action="store_true", help="Use equivariant denoiser")
    parser.add_argument("--n_trans", type=int, default=2, help="Number of transformations for equivariant denoiser")
    parser.add_argument("--sigma_x", type=float, default=None, help="Noise level for data augmentation")
    parser.add_argument("--num_train_images", type=int, default=None, help="Number of training images to use")
    parser.add_argument("--idx_imgs", type=int, default=1, help="Image index for SR visualization")
    parser.add_argument("--use_test_data", dest="use_test_data", action="store_true", help="Load evaluation images from test_data/<dataset>.")
    parser.add_argument("--no_test_data", dest="use_test_data", action="store_false", help="Load evaluation images from the full dataset test split.")
    parser.set_defaults(use_test_data=True)
    return parser


def run_cli(default_task="demo"):
    args = build_parser(default_task=default_task).parse_args()
    main(args)


if __name__ == "__main__":
    run_cli()
