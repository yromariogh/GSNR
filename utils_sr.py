from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

import numpy as np
import scipy
import torch

# Optional imports (only needed for eigensolvers / conversions / plotting)
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# -----------------------------------------------------------------------------
# Helpers for sparse/dense ops
# -----------------------------------------------------------------------------

def _is_sparse(A: torch.Tensor) -> bool:
    return A.layout in (torch.sparse_coo, torch.sparse_csr)

def _t(A: torch.Tensor) -> torch.Tensor:
    return A.transpose(-2, -1)

def _spmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiply handling sparse/dense combos:
      - sparse @ dense  -> torch.sparse.mm
      - dense  @ sparse -> densify right then @ (PyTorch doesn't support directly)
      - sparse @ sparse -> densify right
      - dense  @ dense  -> @
    """
    if _is_sparse(A) and not _is_sparse(B):
        return torch.sparse.mm(A, B)
    elif (not _is_sparse(A)) and _is_sparse(B):
        return A @ B.to_dense()
    elif _is_sparse(A) and _is_sparse(B):
        return torch.sparse.mm(A, B.to_dense())
    else:
        return A @ B

def _to_dense_if_sparse(A: torch.Tensor) -> torch.Tensor:
    return A.to_dense() if _is_sparse(A) else A


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


from torch import Tensor
import matplotlib.pyplot as plt

def set_matplotlib_latex(fs=12):
    """
    Configure matplotlib to use LaTeX rendering with a given global font size.
    
    Parameters:
        fs (int): Global font size for text (default is 12).
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.size": fs,
        "font.family": "serif",
        "axes.titlesize": fs,
        "axes.labelsize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": fs,
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

def psnr_fun(preds: Tensor, target: Tensor, min_val=0) -> Tensor:
    N = preds.shape[0]
    flat_preds  = preds.reshape(N, -1)
    flat_target = target.reshape(N, -1)

    # 1) MSE por muestra
    mse_per_sample = torch.mean((flat_preds - flat_target) ** 2, dim=1)  # (N,)

    # 2) Pico por muestra
    max_val = flat_target.max(dim=1).values
    if min_val is None:
        min_val = flat_target.min(dim=1).values
    peak = max_val - min_val

    # 3) PSNR por muestra, sin eps para permitir ±∞
    psnr_per_sample = 10 * torch.log10(peak**2 / mse_per_sample)

    # 4) Media sobre el batch (ignorando infinities en la media de PyTorch)
    psnr_mean = psnr_per_sample.mean()

    return psnr_mean
# -----------------------------------------------------------------------------
# Downsampling matrix (sparse) and application
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
def downsampling_matrix_torch(
    n: int,
    s: int,
    mode: str = "average",          # 'average' or 'decimate'
    layout: str = "csr",            # 'csr' or 'coo'
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build a sparse downsampling matrix D so that y = D @ x downsamples
    a vectorized (row-major) square image of size HxW (H=W=√n) by factor s.

    Returns
    -------
    D : torch.sparse_*_tensor with shape [ (H//s * W//s), n ]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = W = int(math.isqrt(n))
    if H * W != n:
        raise ValueError("n must be a perfect square (image assumed square).")
    if (H % s) != 0:
        raise ValueError("√n must be divisible by s.")
    if mode not in ("average", "decimate"):
        raise ValueError("mode must be 'average' or 'decimate'.")
    if layout not in ("csr", "coo"):
        raise ValueError("layout must be 'csr' or 'coo'.")

    hL, wL = H // s, W // s
    m = hL * wL

    # Low-res grid (row-major)
    lr_r = torch.arange(hL, device=device).repeat_interleave(wL)   # (m,)
    lr_c = torch.arange(wL, device=device).repeat(hL)              # (m,)

    if mode == "decimate":
        # one nonzero per LR pixel
        hr_r = lr_r * s                                            # (m,)
        hr_c = lr_c * s                                            # (m,)
        hr_idx = hr_r * W + hr_c                                   # (m,)

        rows = torch.arange(m, device=device, dtype=torch.int64)   # (m,)
        cols = hr_idx.to(torch.int64)
        vals = torch.ones(m, device=device, dtype=dtype)

    else:
        # 'average' → s^2 nonzeros per LR pixel (weight = 1/s^2)
        br = torch.arange(s, device=device)
        bc = torch.arange(s, device=device)
        block_r = br.repeat_interleave(s)                          # (s^2,)
        block_c = bc.repeat(s)                                     # (s^2,)

        # Expand each LR pixel to its s×s HR block coordinates
        hr_r = (lr_r[:, None] * s + block_r[None, :]).reshape(-1)  # (m*s^2,)
        hr_c = (lr_c[:, None] * s + block_c[None, :]).reshape(-1)  # (m*s^2,)
        hr_idx = hr_r * W + hr_c                                   # (m*s^2,)

        rows = torch.arange(m, device=device, dtype=torch.int64).repeat_interleave(s * s)
        cols = hr_idx.to(torch.int64)
        vals = torch.full((m * s * s,), 1.0 / (s * s), device=device, dtype=dtype)

    if layout == "csr" and hasattr(torch, "sparse_csr_tensor"):
        # compute crow_indices (row pointer) directly
        if mode == "decimate":
            nnz_per_row = torch.ones(m, device=device, dtype=torch.int64)
        else:
            nnz_per_row = torch.full((m,), s * s, device=device, dtype=torch.int64)
        crow_indices = torch.zeros(m + 1, device=device, dtype=torch.int64)
        crow_indices[1:] = torch.cumsum(nnz_per_row, dim=0)

        D = torch.sparse_csr_tensor(crow_indices, cols, vals, size=(m, n), dtype=dtype, device=device)
    else:
        indices = torch.stack([rows, cols], dim=0)                 # (2, nnz)
        D = torch.sparse_coo_tensor(indices, vals, size=(m, n), dtype=dtype, device=device).coalesce()

    return D

def apply_downsampling_sparse(D: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Apply sparse downsampling matrix D to a single image vector x [n]
    or a batch of vectors x [B, n].
    """
    if x.ndim == 1:
        return torch.sparse.mv(D, x)
    elif x.ndim == 2:
        yTB = torch.sparse.mm(D, x.t())   # [m, B]
        return yTB.t()
    else:
        raise ValueError("x must be a 1-D vector [n] or a 2-D batch [B, n].")

# -----------------------------------------------------------------------------
# Sparse-aware analytics
# -----------------------------------------------------------------------------


def normalized_coverage(S_rows: torch.Tensor,
                        Pn: torch.Tensor,
                        X: torch.Tensor,
                        mean: torch.Tensor) -> torch.Tensor:
    """
    S_rows : [p, n] (dense or sparse) — rows spanning a subspace in R^n
    Pn     : [n, n] (dense or sparse) — projector onto the nullspace of H
    X      : [N, n] dense data matrix (each row a sample)
    mean   : [n] or [1, n] dense mean to subtract
    Returns scalar coverage in [0,1].
    """
    Xc = X - mean
    if _is_sparse(Pn):
        Xn = _spmm(Pn, Xc.T).T               # (N, n)
    else:
        Xn = Xc @ _t(Pn)                     # (N, n)

    Cb = (Xn.T @ Xn) / max(1, X.shape[0])    # (n, n) dense
    eps = torch.tensor(1e-12, dtype=Cb.dtype, device=Cb.device)
    trCb = torch.trace(Cb) + eps

    Qp = S_rows.T/S_rows.max()#orthonormal_basis_from_rows(S_rows) # (n, r)
    if S_rows.shape[1] == 0:
        return torch.as_tensor(0.0, dtype=Cb.dtype, device=Cb.device)

    B = Cb @ Qp                               # (n, r)
    cov = torch.sum(Qp @ B.T) / trCb            # scalar
    return cov

def generate_orthogonal_rows_qr(H: torch.Tensor, p: int) -> torch.Tensor:
    """
    Given H (m×n), generate p orthonormal rows S in the nullspace of H.
    If H is sparse, densify for QR (PyTorch lacks sparse QR).
    Returns S : [p_eff, n] dense with orthonormal rows (p_eff ≤ p).
    """
    Ht = _t(H).to_dense()          # (n, m) dense
    Q_full, _ = torch.linalg.qr(Ht, mode='complete')  # (n, n)
    m, n = H.shape
    nullspace_basis = Q_full[:, m:]                        # (n, k)

    if nullspace_basis.shape[1] == 0 or p <= 0:
        return torch.zeros((0, n), dtype=H.dtype, device=H.device)

    k = nullspace_basis.shape[1]
    p_eff = min(p, k)
    P = torch.randn(k, p_eff, device=H.device, dtype=H.dtype)
    U, _ = torch.linalg.qr(P, mode='reduced')             # (k, p_eff)
    S = _t(U) @ _t(nullspace_basis)                       # (p_eff, n)
    return S

def null_projector(H: torch.Tensor) -> torch.Tensor:
    """
    Projector onto ker(H): Pn = I - H^+ H.
    For sparse H, use H^+ = H^T (H H^T)^{-1} (assumes full row rank).
    Returns dense [n, n] projector.
    """
    m, n = H.shape
    I_n = torch.eye(n, dtype=H.dtype, device=H.device)

    if _is_sparse(H):
        G = _spmm(H, _t(H).to_dense())                    # (m, m) dense
        I_m = torch.eye(m, dtype=H.dtype, device=H.device)
        Ginv = torch.linalg.solve(G, I_m)                 # (m, m)
        H_pinv = _t(H).to_dense() @ Ginv                  # (n, m)
        Pn = I_n - H_pinv @ _to_dense_if_sparse(H)        # (n, n) dense
    else:
        H_pinv = torch.linalg.pinv(H)                     # (n, m)
        Pn = I_n - H_pinv @ H                             # (n, n)

    return Pn

def orthonormal_basis_from_rows(S: torch.Tensor) -> torch.Tensor:
    """
    Given S [p, n] (dense or sparse), return an orthonormal basis Q of its
    row-span as a dense matrix of shape [n, r], with Q^T Q = I_r.
    """
    if S.numel() == 0 or S.shape[0] == 0:
        return torch.zeros((S.shape[1], 0), dtype=S.dtype, device=S.device)
    St = _to_dense_if_sparse(_t(S))                       # (n, p)
    Q, _ = torch.linalg.qr(St, mode="reduced")            # (n, r)
    return Q

# -----------------------------------------------------------------------------
# Sparse-friendly nullspace operator (implicit) using SciPy
# -----------------------------------------------------------------------------

def torch_to_scipy_csr(A: torch.Tensor) -> "sp.csr_matrix":
    """
    Accepts torch dense / COO / CSR and returns SciPy CSR on CPU (float64).
    """
    if not _HAS_SCIPY:
        raise ImportError("SciPy is required for this function.")
    if A.layout == torch.sparse_csr:
        data = A.values().detach().cpu().numpy()
        indices = A.col_indices().detach().cpu().numpy()
        indptr = A.crow_indices().detach().cpu().numpy()
        shape = tuple(A.shape)
        return sp.csr_matrix((data, indices, indptr), shape=shape, dtype=np.float64)
    elif A.layout == torch.sparse_coo:
        A = A.coalesce()
        idx = A.indices().detach().cpu().numpy()
        data = A.values().detach().cpu().numpy()
        shape = tuple(A.shape)
        return sp.csr_matrix((data, (idx[0], idx[1])), shape=shape, dtype=np.float64)
    else:
        return sp.csr_matrix(A.detach().cpu().numpy().astype(np.float64))

def scipy_to_torch_dense(M: "np.ndarray | sp.spmatrix",
                         device: torch.device,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if _HAS_SCIPY and sp.issparse(M):
        M = M.toarray()
    return torch.from_numpy(np.asarray(M)).to(device=device, dtype=dtype)

def make_nullspace_projector(H_csr: "sp.csr_matrix"):
    """
    Returns a callable Pn(v) that projects v (np.ndarray, shape [n]) onto N(H):
        Pn v = v - H^T (H H^T)^{-1} (H v)
    Assumes H has full row rank.
    """
    
    # check if H_csr is indeed csr
    if not sp.isspmatrix_csr(H_csr):
        H_csr = torch_to_scipy_csr(H_csr)
    if not _HAS_SCIPY:
        raise ImportError("SciPy is required for this function.")
    G = (H_csr @ H_csr.T).tocsr()
    # regularization for safety if ill-conditioned
    # lam = 1e-12 * spla.norm(G)
    # G = (G + lam*sp.eye(G.shape[0])).tocsr()

    solve_G = spla.factorized(G.tocsc())  # returns function y = G^{-1} b
    H = H_csr
    HT = H_csr.T.tocsr()

    def Pn(v: np.ndarray) -> np.ndarray:
        Hv = H @ v
        y = solve_G(Hv)
        corr = HT @ y
        return v - corr

    return Pn

def make_T_operator(H_csr: "sp.csr_matrix", L_csr: "sp.csr_matrix") -> "spla.LinearOperator":
    """
    Creates a LinearOperator representing T x = Pn( L( Pn(x) ) ).
    """
    if not _HAS_SCIPY:
        raise ImportError("SciPy is required for this function.")
    n = H_csr.shape[1]
    Pn = make_nullspace_projector(H_csr)

    def matvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x_p = Pn(x)
        y = L_csr @ x_p
        z = Pn(y)
        return z

    return spla.LinearOperator((n, n), matvec=matvec, dtype=np.float64)

def null_space_basis(H: torch.Tensor, q: int):
    m, n = H.shape
    # QR completa sobre H^T para tener Q_full (n×n)
    Q_full, _       = torch.linalg.qr(H.T, mode='complete')  # (n, n)
    nullspace_basis = Q_full[:, m:]                         # (n, q)
    return nullspace_basis
def build_nullspace_operator_sparse(H: torch.Tensor,
                                    L: torch.Tensor,
                                    k: Optional[int] = None,
                                    which: str = "SM",
                                    tol: float = 1e-3,
                                    maxiter: Optional[int] = None,
                                    device: Optional[torch.device] = None,
                                    out_dtype: torch.dtype = torch.float32
                                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the leading k eigenpairs of T = Pn L Pn without forming dense NxN.
    - H: [m, n] torch (dense/COO/CSR)   sensing matrix
    - L: [n, n] torch (dense/COO/CSR)   symmetric (or PSD) operator
    - k: number of eigenpairs to compute. If None or <= 0, uses full nullity(H).

    Returns (S_full, U_raw, evals) where
      S_full : [k, n] dense rows spanning the top eigenspace of T (each row is an eigenvector)
      U_raw  : [n, k] dense column eigenvectors in ambient space
      evals   : [k] eigenvalues

    Notes
    -----
    - This implementation uses an implicit SciPy LinearOperator for T and only requests the needed
      eigenpairs from ARPACK (spla.eigsh), avoiding any dense nullspace projector materialization.
    """
    if not _HAS_SCIPY:
        raise ImportError("SciPy is required for build_nullspace_operator_sparse.")

    if device is None:
        device = H.device

    H_csr = torch_to_scipy_csr(H)
    L_csr = torch_to_scipy_csr(L)
    n = H_csr.shape[1]
    m = H_csr.shape[0]
    
    q = n#-np.linalg.matrix_rank(H_csr.todense())
    print(f'Computed nullity q={q}')
    # Quick return for degenerate cases
    if q == 0 or (k is not None and k <= 0):
        U_raw = torch.zeros((n, 0), device=device, dtype=out_dtype)
        S_full = torch.zeros((0, n), device=device, dtype=out_dtype)
        evals = torch.zeros((0,), device=device, dtype=out_dtype)
        return S_full, U_raw, evals

    if n <= 1:
        raise ValueError(
            "build_nullspace_operator_sparse requires ambient dimension > 1 when ker(H) is non-trivial."
        )

    # Determine how many eigenpairs to request without violating scipy's k < n constraint.
    max_k = min(q, n - 1)
    if max_k <= 0:
        U_raw = torch.zeros((n, 0), device=device, dtype=out_dtype)
        S_full = torch.zeros((0, n), device=device, dtype=out_dtype)
        evals = torch.zeros((0,), device=device, dtype=out_dtype)
        return S_full, U_raw, evals

    if k is None or k <= 0:
        k_eff = max_k
    else:
        k_eff = max(1, min(k, max_k))

    Top = make_T_operator(H_csr, L_csr)

    # Use ARPACK via scipy.sparse.linalg.eigsh on the implicit operator. Request k_eff eigenpairs.
    # which parameter is forwarded; defaults kept as provided by caller.
    print(f"Requesting k={k_eff} eigenpairs from ARPACK (max allowed: {max_k})...")
    evals_np, vecs_np = spla.eigsh(Top, k=k_eff, which=which, tol=tol, maxiter=10)

    # vecs_np shape: (n, k_eff) -- columns are eigenvectors in ambient space (already lie in null(H))
    U_raw = torch.from_numpy(vecs_np).to(device=device, dtype=out_dtype)   # [n, k]
    evals = torch.from_numpy(evals_np).to(device=device, dtype=out_dtype)   # [k]

    # S_full: rows spanning the top eigenspace (each row an eigenvector). Columns from eigsh
    # already live in ker(H) thanks to how Top is defined, so we simply transpose.
    S_full = U_raw.transpose(0, 1).contiguous()  # [k, n]

    return S_full, U_raw, evals

# -----------------------------------------------------------------------------
# Plotting helpers (spy-style). Only used on demand.
# -----------------------------------------------------------------------------

def spy_torch(D: torch.Tensor, markersize: float = 1.0, ax=None):
    """
    Show the sparsity pattern of a PyTorch sparse tensor (COO or CSR).
    """
    import matplotlib.pyplot as plt

    if D.layout == torch.sparse_csr:
        D = D.to_sparse_coo()
    elif D.layout != torch.sparse_coo:
        raise ValueError("D must be sparse COO or CSR")

    D = D.coalesce().cpu()
    idx = D.indices()           # [2, nnz]
    r = idx[0].numpy()
    c = idx[1].numpy()

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(c, r, linestyle='none', marker='.', markersize=markersize)
    ax.set_xlim(-0.5, D.size(1)-0.5)
    ax.set_ylim(D.size(0)-0.5, -0.5)   # invert y for matrix-like view
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Sparse pattern: shape={tuple(D.shape)}, nnz={D._nnz()}")
    ax.set_xlabel("columns")
    ax.set_ylabel("rows")
    plt.show()

def spy_values_torch(D: torch.Tensor, markersize: float = 1.0, ax=None):
    """
    Scatter-plot of nonzero entries colored by their values.
    Works for COO or CSR.
    """
    import matplotlib.pyplot as plt

    if D.layout == torch.sparse_csr:
        D = D.to_sparse_coo()
    elif D.layout != torch.sparse_coo:
        raise ValueError("D must be sparse COO or CSR")

    D = D.coalesce().cpu()
    idx = D.indices().numpy()
    val = D.values().numpy()
    r, c = idx[0], idx[1]

    if ax is None:
        fig, ax = plt.subplots()
    sc = ax.scatter(c, r, s=markersize, c=val)
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Values: shape={tuple(D.shape)}, nnz={D._nnz()}")
    ax.set_xlabel("columns"); ax.set_ylabel("rows")
    plt.colorbar(sc, ax=ax, label="value")
    plt.show()

def show_dense_preview(D: torch.Tensor, max_side: int = 256):
    """
    For small matrices only: convert to dense and imshow.
    """
    import matplotlib.pyplot as plt

    m, n = D.shape
    if max(m, n) > max_side:
        raise ValueError("Matrix too large for dense preview; use spy instead.")
    if D.layout == torch.sparse_csr:
        D = D.to_sparse_coo()
    Dd = D.to_dense().cpu().numpy()
    plt.imshow(Dd)
    plt.title(f"Dense preview {Dd.shape}")
    plt.xlabel("columns"); plt.ylabel("rows")
    plt.gca().invert_yaxis()
    plt.show()

# -----------------------------------------------------------------------------
# Convenience: batched nullspace projection without forming Pn
# -----------------------------------------------------------------------------

def project_onto_nullspace_batch(H: torch.Tensor, V: torch.Tensor, reg: float = 0.0) -> torch.Tensor:
    """
    Project a batch V [B, n] onto ker(H) using implicit formula:
        proj(V) = V - H^T (H H^T + reg*I)^{-1} (H V^T)
    Returns [B, n]. Uses dense solves in m-space.
    """
    if _is_sparse(H):
        Ht = _t(H).to_dense()
        HV = _spmm(H, V.T).to_dense() if _is_sparse(H) else H @ V.T  # [m, B]
        G = _spmm(H, Ht).to_dense()                                  # [m, m]
    else:
        Ht = _t(H)
        HV = H @ V.T                                                 # [m, B]
        G = H @ Ht                                                   # [m, m]

    m = G.shape[0]
    if reg > 0:
        G = G + reg * torch.eye(m, dtype=G.dtype, device=G.device)

    # Solve G Y = HV for all RHS (B columns)
    Y = torch.linalg.solve(G, HV)                                    # [m, B]
    corr = Ht @ Y                                                    # [n, B]
    return (V.T - corr).T

# -----------------------------------------------------------------------------
# __all__ for clean imports
# -----------------------------------------------------------------------------

__all__ = [
    "downsampling_matrix_torch",
    "apply_downsampling_sparse",
    "normalized_coverage",
    "generate_orthogonal_rows_qr",
    "null_projector",
    "orthonormal_basis_from_rows",
    "torch_to_scipy_csr",
    "scipy_to_torch_dense",
    "make_nullspace_projector",
    "make_T_operator",
    "build_nullspace_operator_sparse",
    "spy_torch",
    "spy_values_torch",
    "show_dense_preview",
    "project_onto_nullspace_batch",
]
