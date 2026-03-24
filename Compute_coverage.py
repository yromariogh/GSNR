import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from comput_gnsr_matrix import (
    apply_task_defaults,
    build_measurement_matrix,
    build_parser as build_generator_parser,
    compute_p_list,
    load_reference_split,
    parse_float_sequence,
    select_variants,
    str_to_bool,
    task_suffix,
)
from utils import results_root_path
from utils_sr import normalized_coverage, null_projector, set_matplotlib_latex


set_matplotlib_latex(12)


def build_parser() -> argparse.ArgumentParser:
    parser = build_generator_parser()
    parser.description = "Compute coverage curves for SR or demosaicing graph operators."
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
        print("sim_graph flag is False, skipping coverage computation.")
        return

    sim_gamma_values = parse_float_sequence(args.sim_gamma)
    if sim_gamma_values:
        print(f"Using sim_gamma values: {sim_gamma_values}")

    if args.p_list not in {"mB", "auto", ""}:
        p_values = []
        for token in args.p_list.split(","):
            token = token.strip()
            if token:
                try:
                    p_values.append(max(1, int(token)))
                except ValueError:
                    continue
        args.p_values = sorted(set(p_values))
    else:
        args.p_values = []

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.device = device

    folder_path = results_root_path(args) / "GRAPH_MATRICES"
    folder_path.mkdir(parents=True, exist_ok=True)
    suffix = task_suffix(args)

    Hmat = build_measurement_matrix(args, device, dense=True).detach().to(device)
    Pn = null_projector(Hmat)

    _, _, X_test = load_reference_split(args, device)
    if X_test.shape[0] == 0:
        raise ValueError("Test set must contain at least one image.")
    X_test_mean = X_test.mean(dim=0, keepdim=True)

    coverage_curves = []
    for variant in select_variants(args):
        S_path = folder_path / f"S_full_{variant}_{suffix}.pt"
        if not S_path.exists():
            print(f"[WARN] Missing S matrix for variant '{variant}': {S_path}")
            continue

        S_full = torch.load(S_path, map_location=device)
        if isinstance(S_full, torch.Tensor):
            S_full = S_full.to(device=device, dtype=torch.float32)
        else:
            S_full = torch.as_tensor(S_full, dtype=torch.float32, device=device)

        q = int(S_full.shape[0])
        if q == 0:
            print(f"[WARN] Variant '{variant}' produced empty S.")
            continue

        p_limit = args.graph_r if args.graph_r is not None else None
        separation = max(1, args.separation)
        target_max = min(q, p_limit) if p_limit is not None else q
        if args.p_values:
            p_list = [p for p in args.p_values if p <= target_max]
            if not p_list or p_list[-1] != target_max:
                p_list.append(target_max)
        else:
            p_list = compute_p_list(q, separation, p_limit)

        cov = []
        for p in p_list:
            S_rows = S_full[:p, :]
            cov.append(normalized_coverage(S_rows, Pn, X_test, X_test_mean).cpu())
        cov = torch.stack(cov)
        coverage_curves.append((variant, p_list, cov))
        print(f"{variant}: average coverage = {cov.mean():.4f}")

        if args.do_individual_plots:
            plt.figure(figsize=(7, 4.5))
            p_axis = [0] + p_list
            cov_axis = np.concatenate(([0.0], cov.numpy()))
            plt.plot(p_axis, cov_axis, label=variant)
            plt.xlabel("$p$")
            plt.ylabel("$C(p)$")
            plt.legend()
            plt.tight_layout()
            plt.savefig(folder_path / f"coverage_vs_p_{variant}_{suffix}.png")
            plt.close()

    if not coverage_curves:
        print("No coverage curves were computed.")
        return

    save_dict = {}
    for variant, p_list, cov in coverage_curves:
        save_dict[f"{variant}_p"] = np.array(p_list, dtype=np.int64)
        save_dict[f"{variant}_cov"] = cov.numpy()
    np.savez(folder_path / f"coverage_curves_{suffix}.npz", **save_dict)

    labels = {
        "identity": r"$\mathbf{L} = \mathbf{I}$",
        "grid_4nn": r"$\mathbf{L} = \mathbf{L}_{4nn}$",
        "grid_8nn": r"$\mathbf{L} = \mathbf{L}_{8nn}$",
        "rw": r"$\mathbf{L} = \mathbf{L}_{rw}$",
        "sym": r"$\mathbf{L} = \mathbf{L}_{sym}$",
    }
    plt.figure(figsize=(7, 4.5))
    for variant, p_list, cov in coverage_curves:
        p_axis = [0] + p_list
        cov_axis = np.concatenate(([0.0], cov.numpy()))
        plt.plot(p_axis, cov_axis, label=labels.get(variant, variant), linewidth=2)
    plt.xlabel("$p$")
    plt.ylabel("$C(p)$")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder_path / f"coverage_vs_p_graph_suite_{suffix}.png")
    plt.close()


if __name__ == "__main__":
    run_experiment(parse_args())
