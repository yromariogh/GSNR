import argparse
from pathlib import Path

import deepinv as dinv
import torch
from tqdm import tqdm

from algos_dinv import NPN_PGD
from load_data import get_dataloaders
from other_models import UNetLeon
from pnp_gsnr import (
    build_denoiser,
    build_graph_term,
    build_parser as build_base_parser,
    build_primary_operator,
    build_secondary_operator,
)
from utils import graph_matrix_path, model_path, normalize_task, task_results_dir
from utils_sr import AverageMeter, psnr_fun, set_seed


TASK_DEFAULTS = {
    "sr": {
        "dataset": "celeba",
        "batch_size": 1,
        "n": 128,
        "grayscale": False,
        "variant": "identity",
        "p": 0.1,
        "iters": 300,
        "lambd": 0.1,
        "gamma_npn": 0.0,
        "gamma_graph": 0.0,
        "sigma_x": 0.0,
        "num_train_images": 20,
        "num_test_images": 50,
    },
    "demo": {
        "dataset": "celeba",
        "batch_size": 1,
        "n": 64,
        "grayscale": False,
        "variant": "identity",
        "p": 0.5,
        "iters": 2000,
        "lambd": 0.1,
        "gamma_npn": 0.0,
        "gamma_graph": 0.0,
        "sigma_x": 0.0,
        "num_train_images": 20,
        "num_test_images": 20,
    },
}


def apply_test_defaults(args):
    args.task = normalize_task(args.task)
    for key, value in TASK_DEFAULTS[args.task].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def main(args):
    args = apply_test_defaults(args)
    channels = 1 if args.grayscale else 3
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    _, testloader = get_dataloaders(args)
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

    data_fidelity = dinv.optim.L2()
    prior = build_denoiser(args, device)

    gsnr_psnr = AverageMeter()
    sensor_psnr = AverageMeter()
    data_loop = tqdm(testloader, desc="Testing", unit="batch", colour="blue")

    for idx, batch in enumerate(data_loop):
        if idx >= args.num_test_images:
            print("Reached num_test_images limit.")
            break

        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        batch_size = x.shape[0]
        y = spc_h(x)
        stepsize = args.alpha / spc_h.compute_norm(spc_h.A_adjoint(y), tol=1e-3).item()

        solver = NPN_PGD(
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

        y_s = spc_s(predictor(spc_h.A_adjoint(y)))
        y_s_gt = spc_s(x)
        sensor_psnr.update(psnr_fun(y_s, y_s_gt).item(), batch_size)

        x0 = spc_h.A_dagger(y)
        x0_graph = x0 if args.task == "sr" else x0 + spc_s.A_dagger(y_s)
        x_hat_graph, _ = solver(
            x0_graph,
            y,
            y_s,
            spc_h,
            spc_s,
            xgt=x,
            B=graph_term,
            channel_wise=args.task == "demo",
        )

        gsnr_psnr.update(psnr_fun(x_hat_graph, x).item(), batch_size)
        data_loop.set_postfix({"GSNR_PSNR": gsnr_psnr.avg, "Sensor_PSNR": sensor_psnr.avg})

    print(f"Final GSNR PSNR: {gsnr_psnr.avg:.4f}")
    print(f"Final Sensor PSNR: {sensor_psnr.avg:.4f}")

    output_dir = Path("RESULTS") / "PNP" / task_results_dir(args.task)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / (
        f"test_{args.task}_variant_{args.variant}_denoiser_{args.denoiser}_p_{args.p}"
        f"_lambd_{args.lambd}_gamma_npn_{args.gamma_npn}_gamma_graph_{args.gamma_graph}"
        f"_alpha_{args.alpha}_iters_{args.iters}.txt"
    )
    with open(results_path, "w", encoding="utf-8") as handle:
        handle.write(f"Final GSNR PSNR: {gsnr_psnr.avg:.4f}\n")
        handle.write(f"Final Sensor PSNR: {sensor_psnr.avg:.4f}\n")


def build_parser(default_task="sr"):
    parser = build_base_parser(default_task=default_task)
    parser.description = "Batch GSNR evaluation for SR or demosaicing."
    parser.set_defaults(lambd=None, gamma_npn=None, gamma_graph=None, sigma_x=None)
    parser.add_argument("--num_test_images", type=int, default=None, help="Number of test batches to evaluate")
    return parser


def run_cli(default_task="sr"):
    args = build_parser(default_task=default_task).parse_args()
    main(args)


if __name__ == "__main__":
    run_cli()
