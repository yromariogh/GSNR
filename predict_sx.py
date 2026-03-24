import argparse

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from load_data import get_dataloaders
from other_models import UNetLeon
from utils import demosaicing_h_matrix_path, graph_matrix_path, model_dir, normalize_task
from utils_sr import AverageMeter, downsampling_matrix_torch, psnr_fun, set_matplotlib_latex

try:
    import wandb

    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


TASK_DEFAULTS = {
    "sr": {
        "dataset": "celeba",
        "batch_size": 40,
        "n": 128,
        "grayscale": True,
        "variant": "sym",
        "p": 0.1,
        "epochs": 100,
        "num_train_images": 10000,
    },
    "demo": {
        "dataset": "places",
        "batch_size": 12,
        "n": 64,
        "grayscale": False,
        "variant": "grid_4nn",
        "p": 0.7,
        "epochs": 50,
        "num_train_images": 10000,
    },
}


set_matplotlib_latex()


def apply_task_defaults(args):
    task = normalize_task(args.task)
    for key, value in TASK_DEFAULTS[task].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.task = task
    return args


def flatten_batch(x: torch.Tensor, channels: int, task: str) -> torch.Tensor:
    if task == "sr":
        return x.reshape(x.shape[0], channels, -1)
    return x.reshape(x.shape[0], -1)


def reshape_x0(y: torch.Tensor, H: torch.Tensor, batch_size: int, channels: int, n: int) -> torch.Tensor:
    x0 = y @ H
    return x0.reshape(batch_size, channels, n, n)


def reshape_sensor_reconstruction(ys_hat: torch.Tensor, S: torch.Tensor, batch_size: int, channels: int, n: int) -> torch.Tensor:
    x_rec = ys_hat @ S
    return x_rec.reshape(batch_size, channels, n, n)


def reshape_measurement(y: torch.Tensor, args, channels: int) -> torch.Tensor:
    if args.task == "sr":
        return y.reshape(y.shape[0], channels, args.n // args.srf, args.n // args.srf)
    return y.reshape(y.shape[0], 1, args.n, args.n)


def load_task_matrices(args, device):
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

    if args.task == "sr":
        H = downsampling_matrix_torch(n=args.n**2, s=args.srf, device=device)
    else:
        H = torch.load(demosaicing_h_matrix_path(args, args.n), map_location=device).to(device)

    return S, H


def main(args):
    args = apply_task_defaults(args)
    channels = 1 if args.grayscale else 3
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = UNetLeon(n_channels=channels, base_channel=64).to(device)
    if args.wandb and _HAS_WANDB:
        wandb.init(project=args.wandb_project, config=vars(args), reinit=True)
        wandb.watch(model, log="all", log_freq=100)

    trainloader, testloader = get_dataloaders(args)
    S, H = load_task_matrices(args, device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()
    exp_path = model_dir(
        args,
        args.dataset,
        args.variant,
        args.p,
        args.lr,
        mode=args.task,
        srf=args.srf if args.task == "sr" else None,
    )
    exp_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_psnr = AverageMeter()
        train_loss = AverageMeter()
        train_loop = tqdm(trainloader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch_idx, batch in enumerate(train_loop):
            x = batch.to(device)
            x_flat = flatten_batch(x, channels, args.task)
            y = x_flat @ H.t()
            ys = x_flat @ S.t()
            x0 = reshape_x0(y, H, x.shape[0], channels, args.n)
            x_hat = model(x0)
            ys_hat = flatten_batch(x_hat, channels, args.task) @ S.t()

            loss = mse(ys_hat, ys)
            train_loss.update(loss.item(), x.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_psnr.update(psnr_fun(ys_hat, ys).item(), x.shape[0])
            train_loop.set_postfix({"loss": train_loss.avg, "psnr_ys": train_psnr.avg})
            if args.wandb and _HAS_WANDB:
                step = epoch * len(trainloader) + batch_idx
                wandb.log({"train/loss": train_loss.avg, "train/psnr": train_psnr.avg}, step=step)

        val_psnr = AverageMeter()
        val_loss = AverageMeter()
        val_loop = tqdm(testloader, desc=f"Validation epoch {epoch + 1}/{args.epochs}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loop):
                x = batch.to(device)
                x_flat = flatten_batch(x, channels, args.task)
                y = x_flat @ H.t()
                ys = x_flat @ S.t()
                x0 = reshape_x0(y, H, x.shape[0], channels, args.n)
                x_hat = model(x0)
                ys_hat = flatten_batch(x_hat, channels, args.task) @ S.t()

                loss = mse(ys_hat, ys)
                val_loss.update(loss.item(), x.shape[0])
                val_psnr.update(psnr_fun(ys_hat, ys).item(), x.shape[0])
                val_loop.set_postfix({"val_loss": val_loss.avg, "val_psnr_ys": val_psnr.avg})

            if args.wandb and _HAS_WANDB:
                wandb.log({"val/loss": val_loss.avg, "val/psnr": val_psnr.avg}, step=epoch)

        torch.save(model.state_dict(), exp_path / "model.pt")
        if args.wandb and _HAS_WANDB:
            wandb.save(str(exp_path / "model.pt"))

        with torch.no_grad():
            batch = next(iter(testloader))
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            batch_size = x.shape[0]
            x_flat = flatten_batch(x, channels, args.task)
            y = x_flat @ H.t()
            x0 = reshape_x0(y, H, batch_size, channels, args.n)
            x_hat = model(x0)
            ys_hat = flatten_batch(x_hat, channels, args.task) @ S.t()
            ys = x_flat @ S.t()
            x_rec = reshape_sensor_reconstruction(ys_hat, S, batch_size, channels, args.n)
            y_img = reshape_measurement(y, args, channels)

            fig, axs = plt.subplots(3, 4, figsize=(9, 9))
            n_display = min(batch_size, axs.shape[0])
            for i in range(n_display):
                axs[i, 0].imshow(
                    x[i].cpu().permute(1, 2, 0).squeeze(),
                    cmap="gray" if args.grayscale else None,
                )
                axs[i, 0].set_title("Original x")
                axs[i, 0].set_axis_off()

                axs[i, 1].imshow(
                    x_rec[i].cpu().permute(1, 2, 0).squeeze(),
                    cmap="gray" if args.grayscale else None,
                )
                psnr_rec = psnr_fun(x_rec[i].reshape(1, -1), x_flat[i].reshape(1, -1)).item()
                axs[i, 1].set_title(r"$\mathbf{S}^T\,\mathrm{G}(\mathbf{x_0})$" + f"\nPSNR = {psnr_rec:.4f}")
                axs[i, 1].set_axis_off()

                ss_img = reshape_sensor_reconstruction(ys[i : i + 1], S, 1, channels, args.n)[0]
                psnr_ss = psnr_fun(ss_img.reshape(1, -1), x_flat[i].reshape(1, -1)).item()
                axs[i, 2].imshow(
                    ss_img.cpu().permute(1, 2, 0).squeeze(),
                    cmap="gray" if args.grayscale else None,
                )
                axs[i, 2].set_title(r"$\mathbf{S}^T \mathbf{S}\,\mathbf{x}^*$" + f"\nPSNR = {psnr_ss:.4f}")
                axs[i, 2].set_axis_off()

                axs[i, 3].imshow(
                    y_img[i].cpu().permute(1, 2, 0).squeeze(),
                    cmap="gray" if args.grayscale else None,
                )
                axs[i, 3].set_title(r"$\mathbf{H}\mathbf{x}$")
                axs[i, 3].set_axis_off()

            plt.tight_layout()
            figure_path = exp_path / f"reconstruction_epoch_{epoch + 1}.svg"
            plt.savefig(figure_path)
            if args.wandb and _HAS_WANDB:
                wandb.log({"reconstruction": wandb.Image(str(figure_path))}, step=epoch)
            plt.close()


def build_parser(default_task="sr"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=default_task, choices=["sr", "demosaicing", "demo"])
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to use.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for data loaders")
    parser.add_argument("--n", type=int, default=None, help="Image size (n x n)")
    parser.add_argument("--grayscale", dest="grayscale", action="store_true", help="Convert images to grayscale")
    parser.add_argument("--color", dest="grayscale", action="store_false", help="Use RGB images")
    parser.set_defaults(grayscale=None)
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cpu or cuda)")
    parser.add_argument("--srf", type=int, default=4, help="Super-resolution factor")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="gNPN", help="wandb project name")
    parser.add_argument("--debug", action="store_true", help="Debug mode with fewer images")
    parser.add_argument("--variant", type=str, default=None, help="Variant of graph for S")
    parser.add_argument("--p", type=float, default=None, help="Proportion of S to use")
    parser.add_argument("--num_train_images", type=int, default=None, help="Number of training images to use")
    return parser


def run_cli(default_task="sr"):
    args = build_parser(default_task=default_task).parse_args()
    main(args)


if __name__ == "__main__":
    run_cli()
