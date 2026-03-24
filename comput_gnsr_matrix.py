import argparse
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from laplacians import (
    anisotropic_from_x0,
    fractional_laplacian,
    grid_laplacian,
    hypergraph_laplacian_from_patches,
    laplacian_normalize,
    learn_graph_from_batch,
    make_batch_from_x0,
    product_space_channel_laplacian,
)
from utils import demosaicing_h_matrix_path, normalize_task, results_root_path
from utils_coverage import load_cifar10_split
from utils_sr import _is_sparse, _spmm, _t, build_nullspace_operator_sparse, downsampling_matrix_torch


SEED = 7
DEFAULT_NOISE_STD = 0
GRAPH_VARIANTS = [
    "grid_4nn",
    "grid_8nn",
    "identity",
    "rw",
    "sym",
]

TASK_DEFAULTS = {
    "sr": {
        "n": 64,
        "srf": 4,
        "channels": 1,
        "test_count": 10000,
        "separation": 1000,
    },
    "demo": {
        "n": 64,
        "srf": 4,
        "channels": 3,
        "test_count": 2000,
        "separation": 10,
    },
}


def str_to_bool(value: Union[str, bool]) -> bool:
    if isinstance(value, bool):
        return value
    text = value.strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    raise ValueError(f"Cannot interpret '{value}' as boolean.")


def parse_float_sequence(text: str) -> List[float]:
    if not text:
        return []
    items: List[float] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if chunk:
            try:
                items.append(float(chunk))
            except ValueError:
                continue
    return items


def apply_task_defaults(args: argparse.Namespace) -> argparse.Namespace:
    task = normalize_task(args.task)
    args.task = task
    for key, value in TASK_DEFAULTS[task].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def task_suffix(args: argparse.Namespace) -> str:
    if args.task == "sr":
        return f"SR_{args.srf}_H_{args.n}"
    return f"demo_H_{args.n}"


def build_measurement_matrix(args: argparse.Namespace, device: torch.device, dense: bool = False) -> torch.Tensor:
    if args.task == "sr":
        matrix = downsampling_matrix_torch(n=args.n * args.n, s=args.srf, device=device)
    else:
        matrix = torch.load(demosaicing_h_matrix_path(args, args.n), map_location=device)
        if hasattr(matrix, "to"):
            matrix = matrix.to(device)
    if dense and _is_sparse(matrix):
        matrix = matrix.to_dense()
    return matrix


def load_reference_split(args: argparse.Namespace, device: torch.device):
    return load_cifar10_split(
        args.n,
        args.n,
        0,
        0,
        args.test_count,
        device=device,
        grayscale=args.task == "sr",
    )


def safe_x0_img(
    n_side: int,
    H: Optional[torch.Tensor],
    x0_flat: Optional[torch.Tensor],
    device: torch.device,
    channels: int = 1,
) -> torch.Tensor:
    numel = n_side * n_side
    if x0_flat is not None:
        x0_flat = x0_flat.detach().to(device)
        if x0_flat.numel() == numel:
            return x0_flat.view(n_side, n_side)
        if channels > 1 and x0_flat.numel() == channels * numel:
            return x0_flat.view(channels, n_side, n_side).mean(dim=0)
    if H is not None:
        H = H.detach().to(device)
        n_cols = H.shape[1]
        if n_cols == numel:
            one = torch.ones(n_cols, device=device)
            y = H @ one
            reg = 1e-3 * torch.eye(n_cols, device=device)
            if _is_sparse(H):
                HtH = _spmm(_t(H), H)
                Hty = _spmm(_t(H), y)
                x_est = torch.linalg.solve(HtH + reg, Hty)
            else:
                x_est = torch.linalg.solve(H.T @ H + reg, H.T @ y)
            x_est = (x_est - x_est.min()) / (x_est.max() - x_est.min() + 1e-12)
            return x_est.view(n_side, n_side)
    t = torch.linspace(0, 1, steps=n_side, device=device)
    grid_x, grid_y = torch.meshgrid(t, t, indexing="ij")
    return 0.5 * (grid_x + grid_y)


def build_laplacian_variant(name: str, args: argparse.Namespace, n_side: int, x0_img: torch.Tensor) -> torch.Tensor:
    device = args.device
    size = n_side * n_side
    name = name.strip()
    if name in {"grid", "grid4", "grid_4nn"}:
        L = grid_laplacian(n_side, device=device, dtype=torch.float32, kind="4nn")
    elif name == "grid_8nn":
        L = grid_laplacian(n_side, device=device, dtype=torch.float32, kind="8nn")
    elif name == "identity":
        L = torch.eye(size, device=device, dtype=torch.float32)
    elif name == "shuffled":
        base = grid_laplacian(n_side, device=device, dtype=torch.float32, kind="4nn")
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        perm = torch.randperm(size, generator=generator, device=device)
        L = base.index_select(0, perm).index_select(1, perm)
    elif name == "anisotropic":
        L = anisotropic_from_x0(n_side, x0_img, sigma=args.ani_sigma, kind="8nn")
    elif name == "misaligned":
        rotated = torch.rot90(x0_img, k=1, dims=(0, 1))
        L = anisotropic_from_x0(n_side, rotated, sigma=args.ani_sigma, kind="8nn")
    elif name in {"learned", "knn_patches"}:
        batch = make_batch_from_x0(x0_img, T=args.learn_T, noise_std=args.learn_jitter)
        L = learn_graph_from_batch(batch, k=args.knn_k, l1=args.learn_l1, l2=args.learn_l2)
    elif name == "multiscale":
        base_local = grid_laplacian(n_side, device=device, dtype=torch.float32, kind="4nn")
        batch = make_batch_from_x0(x0_img, T=args.learn_T, noise_std=args.learn_jitter)
        nonlocal_L = learn_graph_from_batch(batch, k=args.knn_k, l1=args.learn_l1, l2=args.learn_l2)
        L = args.ms_w_local * base_local + args.ms_w_nonlocal * nonlocal_L
    elif name == "sym":
        base = grid_laplacian(n_side, device=device, dtype=torch.float32, kind="4nn")
        L = laplacian_normalize(base, mode="sym")
    elif name == "rw":
        base = grid_laplacian(n_side, device=device, dtype=torch.float32, kind="4nn")
        L = laplacian_normalize(base, mode="rw")
    elif name == "fractional":
        base = grid_laplacian(n_side, device=device, dtype=torch.float32, kind="4nn")
        L = fractional_laplacian(base, alpha=args.frac_alpha)
    elif name == "product":
        base = grid_laplacian(n_side, device=device, dtype=torch.float32, kind="4nn")
        L = product_space_channel_laplacian(base, C=args.channels, gamma_c=args.prod_gamma_c)
    elif name == "hyper":
        L = hypergraph_laplacian_from_patches(
            x0_img,
            n_clusters=args.hyper_k,
            patch=args.knn_patch,
            iters=args.learn_iters,
        )
    else:
        raise ValueError(f"Unknown Laplacian variant '{name}'.")
    L = 0.5 * (L + L.T)
    return L.to(device=device, dtype=torch.float32)


def unique_preserve_order(names: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def select_variants(args: argparse.Namespace) -> List[str]:
    variants: List[str] = []
    if args.graph_suite == "reviewer":
        variants.extend(GRAPH_VARIANTS)
    train_variant = args.train_L_variant.strip()
    if train_variant:
        variants.append(train_variant)
    else:
        variants.append(args.graph_training_variant)
    if args.sim_L and args.sim_L in {"grid", "shuffled", "identity"}:
        mapping = {"grid": "grid_4nn", "identity": "identity", "shuffled": "shuffled"}
        variants.append(mapping[args.sim_L])
    for family in args.L_families.split(","):
        family = family.strip()
        if family and family in GRAPH_VARIANTS:
            variants.append(family)
    return unique_preserve_order(variants)


def compute_p_list(q: int, separation: int, limit: Optional[int]) -> List[int]:
    target = min(q, limit) if limit is not None else q
    values = list(range(1, target + 1, separation))
    if values and values[-1] != target:
        values.append(target)
    elif not values:
        values = [target]
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate graph-constrained S matrices for SR or demosaicing.")
    parser.add_argument("--task", type=str, default="sr", choices=["sr", "demosaicing"])
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--separation", type=int, default=None, help="Stride when sweeping p.")
    parser.add_argument("--n", type=int, default=None, help="Spatial image size.")
    parser.add_argument("--srf", type=int, default=None, help="SR factor when task=sr.")
    parser.add_argument("--results_root_dir", type=str, default=None)

    parser.add_argument("--use_graphS", type=str, default="True")
    parser.add_argument("--graph_r", type=int, default=None)
    parser.add_argument("--graph_reg", type=float, default=1e-3)
    parser.add_argument("--Sg_project_each", type=str, default="epoch", choices=["epoch", "none"])

    parser.add_argument("--sim_graph", type=str, default="True")
    parser.add_argument("--sim_gamma", type=str, default="0,0.001,0.01,0.1,1.0")
    parser.add_argument("--sim_L", type=str, default="identity", choices=["grid", "shuffled", "identity"])

    parser.add_argument("--graph_suite", type=str, default="reviewer", choices=["none", "reviewer"])
    parser.add_argument(
        "--graph_training_variant",
        type=str,
        default="grid_4nn",
        choices=["grid_4nn", "grid_8nn", "identity", "anisotropic", "knn_patches", "fractional"],
    )
    parser.add_argument("--ani_sigma", type=float, default=0.1)
    parser.add_argument("--knn_k", type=int, default=8)
    parser.add_argument("--knn_patch", type=int, default=3)
    parser.add_argument("--frac_alpha", type=float, default=0.75)
    parser.add_argument("--ms_w_local", type=float, default=1.0)
    parser.add_argument("--ms_w_nonlocal", type=float, default=1.0)
    parser.add_argument("--prod_gamma_c", type=float, default=0.2)
    parser.add_argument("--hyper_k", type=int, default=16)
    parser.add_argument("--learn_l1", type=float, default=1e-3)
    parser.add_argument("--learn_l2", type=float, default=1e-3)
    parser.add_argument("--learn_iters", type=int, default=50)
    parser.add_argument("--learn_step", type=float, default=1e-1)
    parser.add_argument("--learn_T", type=int, default=16)
    parser.add_argument("--learn_jitter", type=float, default=0.02)
    parser.add_argument("--do_individual_plots", type=str, default="False")
    parser.add_argument("--train_L_variant", type=str, default="grid_4nn")
    parser.add_argument("--search_L", type=str, default="True")
    parser.add_argument("--only_search_L", type=str, default="True")
    parser.add_argument("--fail_on_scout_error", type=str, default="False")
    parser.add_argument(
        "--L_families",
        type=str,
        default="grid_4nn,grid_8nn,anisotropic,learned,identity,shuffled,fractional,sym,rw,random_knn,random_er",
    )
    parser.add_argument("--num_L_per_family", type=int, default=10000)
    parser.add_argument("--p_list", type=str, default="mB")
    parser.add_argument("--tau_budget", type=float, default=1.0)
    parser.add_argument("--save_all_L", type=str, default="True")
    parser.add_argument("--save_all_Sg", type=str, default="True")
    parser.add_argument("--channels", type=int, default=None, help="Channel count for product/demosaicing tasks.")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def run_experiment(args: argparse.Namespace) -> None:
    args = apply_task_defaults(args)
    args.sim_graph = str_to_bool(args.sim_graph)
    args.use_graphS = str_to_bool(args.use_graphS)
    args.do_individual_plots = str_to_bool(args.do_individual_plots)
    args.search_L = str_to_bool(args.search_L)
    args.only_search_L = str_to_bool(args.only_search_L)
    args.fail_on_scout_error = str_to_bool(args.fail_on_scout_error)
    args.save_all_L = str_to_bool(args.save_all_L)
    args.save_all_Sg = str_to_bool(args.save_all_Sg)

    if not args.sim_graph:
        print("sim_graph flag is False, skipping S generation.")
        return

    _ = parse_float_sequence(args.sim_gamma)

    seed = args.seed if args.seed is not None else SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    args.seed = seed

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.device = device

    folder_path = results_root_path(args) / "GRAPH_MATRICES"
    folder_path.mkdir(parents=True, exist_ok=True)

    Hmat = build_measurement_matrix(args, device, dense=False).detach()
    _, _, X_test = load_reference_split(args, device)
    if X_test.shape[0] == 0:
        raise ValueError("Test set must contain at least one image.")
    X_test_mean = X_test.mean(dim=0, keepdim=True)

    x0_img = safe_x0_img(
        args.n,
        H=Hmat.detach().cpu(),
        x0_flat=X_test[0].detach().cpu(),
        device=device,
        channels=args.channels,
    )

    for variant in select_variants(args):
        try:
            L = build_laplacian_variant(variant, args, args.n, x0_img)
        except Exception as exc:
            if args.fail_on_scout_error:
                raise
            print(f"[WARN] Skipping variant '{variant}': {exc}")
            continue

        if args.channels > 1 and L.shape[0] == args.n * args.n:
            eye_c = torch.eye(args.channels, device=L.device, dtype=L.dtype)
            L = torch.kron(eye_c, L)

        print(f"Processing task='{args.task}' variant='{variant}' with L shape {tuple(L.shape)}")
        S_full, _, evals = build_nullspace_operator_sparse(Hmat, L=L)
        suffix = task_suffix(args)
        torch.save(S_full, folder_path / f"S_full_{variant}_{suffix}.pt")
        torch.save(evals, folder_path / f"evals_{variant}_{suffix}.pt")

    print("S-matrix generation completed.")


if __name__ == "__main__":
    run_experiment(parse_args())
