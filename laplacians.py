# -*- coding: utf-8 -*-
"""
laplacians.py — Graph Laplacians and builders
---------------------------------------------
This module centralizes graph Laplacian constructors and a CT-specific
selector `build_L_variant_CT` that mirrors MRI/SPC variants.
"""

from typing import Optional
import torch
import torch.nn.functional as F
import re
import torch
import torch.nn.functional as F


def _gaussian2d(ksize, sigma, device, dtype):
    # Create a normalized 2D Gaussian kernel
    assert ksize % 2 == 1 and ksize > 1, "ksize must be odd and > 1"
    half = ksize // 2
    ax = torch.arange(-half, half + 1, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(ax, ax)
    g = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    g = g / g.sum()
    return g


def _log2d(ksize, sigma, device, dtype):
    # Discrete Laplacian of Gaussian (LoG) kernel (a.k.a. Mexican hat)
    assert ksize % 2 == 1 and ksize > 1, "ksize must be odd and > 1"
    half = ksize // 2
    ax = torch.arange(-half, half + 1, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(ax, ax)
    r2 = xx**2 + yy**2
    s2 = sigma**2
    # Continuous LoG, then sampled on grid
    logk = ((r2 - 2 * s2) / (s2**2)) * torch.exp(-r2 / (2 * s2))
    # Zero-mean to ensure sum ~ 0 in discrete form
    logk = logk - logk.mean()
    return logk


def conv_laplacians(ltype, device):
    """
    Supported ltype (case-insensitive):
      - '4nn'                  : 4-neighbor (center negative)
      - '8nn'                  : 8-neighbor (center negative)
      - '4nn_pos'              : 4-neighbor (center positive, CV-style)
      - '8nn_pos'              : 8-neighbor (center positive, CV-style)
      - '4nn_norm'             : 4-neighbor normalized (center -1)
      - '8nn_norm'             : 8-neighbor normalized (center -1)
      - 'diag4'                : diagonal-only 4-neighbor
      - 'identity'             : passthrough
      - LoG (Laplacian of Gaussian), patterns:
            'log{K}:{sigma}'      e.g. 'log5:1.0'
            'log{K}_{sigma}'      e.g. 'log7_1.2'
            'log{K}x{K}:{sigma}'  e.g. 'log9x9:1.4'
      - DoG (Difference of Gaussians), patterns:
            'dog{K}:{s1}:{s2}'    e.g. 'dog7:1.0:2.0'
            'dog{K}_{s1}_{s2}'    e.g. 'dog5_0.8_1.6'
    """
    lt = str(ltype).lower().strip()
    dtype = torch.float32  # default; kernel will be cast to image dtype at use

    # Predefined 3x3 variants
    if lt == "4nn":
        k = torch.tensor([[0., 1., 0.],
                          [1., -4., 1.],
                          [0., 1., 0.]], device=device, dtype=dtype)

    elif lt == "8nn":
        k = torch.tensor([[1., 1., 1.],
                          [1., -8., 1.],
                          [1., 1., 1.]], device=device, dtype=dtype)

    elif lt == "4nn_pos":
        k = torch.tensor([[0., -1., 0.],
                          [-1., 4., -1.],
                          [0., -1., 0.]], device=device, dtype=dtype)

    elif lt == "8nn_pos":
        k = torch.tensor([[-1., -1., -1.],
                          [-1.,  8., -1.],
                          [-1., -1., -1.]], device=device, dtype=dtype)

    elif lt == "4nn_norm":
        k = torch.tensor([[0., 0.25, 0.],
                          [0.25, -1., 0.25],
                          [0., 0.25, 0.]], device=device, dtype=dtype)

    elif lt == "8nn_norm":
        k = torch.tensor([[0.125, 0.125, 0.125],
                          [0.125,  -1.,  0.125],
                          [0.125, 0.125, 0.125]], device=device, dtype=dtype)

    elif lt == "diag4":
        k = torch.tensor([[-1., 0., -1.],
                          [ 0., 4.,  0.],
                          [-1., 0., -1.]], device=device, dtype=dtype)

    elif lt == "identity":
        k = torch.tensor([[0., 0., 0.],
                          [0., 1., 0.],
                          [0., 0., 0.]], device=device, dtype=dtype)

    else:
        # LoG patterns
        m_log = re.fullmatch(r"log(\d+)[xX]?\d*[:_](\d*\.?\d+)", lt)
        # DoG patterns
        m_dog = re.fullmatch(r"dog(\d+)[:_](\d*\.?\d+)[:_](\d*\.?\d+)", lt)

        if m_log:
            ksize = int(m_log.group(1))
            sigma = float(m_log.group(2))
            k = _log2d(ksize, sigma, device, dtype)

        elif m_dog:
            ksize = int(m_dog.group(1))
            s1 = float(m_dog.group(2))
            s2 = float(m_dog.group(3))
            g1 = _gaussian2d(ksize, s1, device, dtype)
            g2 = _gaussian2d(ksize, s2, device, dtype)
            k = g1 - g2
            # Zero-mean to behave like a band-pass Laplacian-ish operator
            k = k - k.mean()

        else:
            raise ValueError(
                f"Unknown laplacian type: {ltype}. "
                "Try one of: 4nn, 8nn, 4nn_pos, 8nn_pos, 4nn_norm, 8nn_norm, diag4, identity, "
                "or LoG (e.g. 'log5:1.0') / DoG (e.g. 'dog7:1.0:2.0')."
            )

    return k.view(1, 1, k.shape[-2], k.shape[-1])


def apply_kernel(image, kernel, padding=None):
    """
    image:  (N, C, H, W)
    kernel: (1, 1, kH, kW)  depthwise-shared across channels
    padding: if None, uses kH//2 (assumes square odd-sized kernel)
    """
    if padding is None:
        padding = kernel.shape[-1] // 2  # works for 3x3, 5x5, 7x7, ...

    # Match kernel dtype to image dtype to avoid type promotion surprises
    kernel = kernel.to(dtype=image.dtype, device=image.device)

    result = []
    for i in range(image.shape[1]):  # per-channel depthwise
        channel = image[:, i:i+1]
        channel_result = F.conv2d(channel, kernel, padding=padding)
        result.append(channel_result)
    return torch.cat(result, dim=1)






@torch.no_grad()
def grid_laplacian(n_side: int, device, dtype=torch.float32, kind: str='4nn'):
    n = n_side * n_side
    rows, cols, vals = [], [], []
    def idx(i, j): return i*n_side + j
    neigh4 = [(1,0),(-1,0),(0,1),(0,-1)]
    neigh8 = neigh4 + [(1,1),(1,-1),(-1,1),(-1,-1)]
    neigh = neigh8 if kind=='8nn' else neigh4
    for i in range(n_side):
        for j in range(n_side):
            p = idx(i,j); deg = 0.0
            for di,dj in neigh:
                ii, jj = i+di, j+dj
                if 0<=ii<n_side and 0<=jj<n_side:
                    q = idx(ii,jj)
                    rows.append(p); cols.append(q); vals.append(-1.0); deg += 1.0
            rows.append(p); cols.append(p); vals.append(deg)
    idxs = torch.tensor([rows, cols], dtype=torch.long, device=device)
    vals = torch.tensor(vals, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idxs, vals, (n,n)).coalesce().to_dense()


@torch.no_grad()
def L_base_laplacian(n_side: int,
                     device,
                     dtype: torch.dtype = torch.float32,
                     w_far: float = 0.7,
                     trace_normalize: bool = True):
    n = n_side * n_side
    off_8  = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
    off_far= [(2,0), (-2,0), (0,2), (0,-2)]
    rows, cols, vals = [], [], []
    def idx(i, j): return i*n_side + j
    for i in range(n_side):
        for j in range(n_side):
            p = idx(i,j)
            for di,dj in off_8:
                ii, jj = i+di, j+dj
                if 0<=ii<n_side and 0<=jj<n_side:
                    q = idx(ii,jj); rows.append(p); cols.append(q); vals.append(1.0)
            for di,dj in off_far:
                ii, jj = i+di, j+dj
                if 0<=ii<n_side and 0<=jj<n_side:
                    q = idx(ii,jj); rows.append(p); cols.append(q); vals.append(float(w_far))
    idxs = torch.tensor([rows, cols], dtype=torch.long, device=device)
    vals = torch.tensor(vals, dtype=dtype, device=device)
    W = torch.sparse_coo_tensor(idxs, vals, (n,n)).coalesce().to_dense()
    W = 0.5*(W + W.T)
    W -= torch.diag(torch.diag(W))
    d = W.sum(1); L = torch.diag(d) - W; L = 0.5*(L + L.T)
    if trace_normalize:
        tr = torch.trace(L).clamp(min=1e-12)
        L = L * (n / tr)
    return L


def laplacian_normalize(L: torch.Tensor, mode: str='sym'):
    """Normalize dense Laplacian matrix."""
    D = torch.diag(torch.diag(L)); W = D - L; d = torch.diag(D)
    if mode=='sym':
        dinv2 = torch.where(d>0, d.rsqrt(), torch.zeros_like(d))
        Dm12 = torch.diag(dinv2)
        Lsym = torch.eye(L.shape[0], device=L.device, dtype=L.dtype) - (Dm12 @ W @ Dm12)
        return 0.5*(Lsym + Lsym.T)
    elif mode=='rw':
        dinv = torch.where(d>0, 1.0/d, torch.zeros_like(d))
        Drw = torch.diag(dinv)
        Lrw = torch.eye(L.shape[0], device=L.device, dtype=L.dtype) - (Drw @ W)
        return Lrw
    return L


def normalize_laplacian_matrix(L: torch.Tensor, normalization: str = "none") -> torch.Tensor:
    """
    Normalize dense Laplacian matrix to reduce sensitivity to alpha_L.
    
    Args:
        L: Dense Laplacian matrix (n x n)
        normalization: Type of normalization ('none', 'trace', 'spectral', 'frobenius')
    
    Returns:
        Normalized Laplacian matrix
    """
    if normalization == "none":
        return L
    
    L_normalized = L.clone()
    
    if normalization == "trace":
        # Normalize to unit trace
        trace = torch.trace(L_normalized).clamp(min=1e-12)
        L_normalized = L_normalized / trace
        
    elif normalization == "spectral":
        # Normalize to unit spectral norm (largest eigenvalue)
        try:
            evals = torch.linalg.eigvals(L_normalized).real
            spectral_norm = torch.max(evals).clamp(min=1e-12)
            L_normalized = L_normalized / spectral_norm
        except:
            # Fallback to Frobenius norm if eigenvalue computation fails
            frob_norm = torch.norm(L_normalized, 'fro').clamp(min=1e-12)
            L_normalized = L_normalized / frob_norm
            
    elif normalization == "frobenius":
        # Normalize to unit Frobenius norm
        frob_norm = torch.norm(L_normalized, 'fro').clamp(min=1e-12)
        L_normalized = L_normalized / frob_norm
    
    return L_normalized


def normalize_laplacian_kernel(kernel: torch.Tensor, normalization: str = "none") -> torch.Tensor:
    """
    Normalize convolutional Laplacian kernel.
    
    Args:
        kernel: Convolutional kernel (1, 1, kH, kW)
        normalization: Type of normalization ('none', 'trace', 'spectral', 'frobenius')
    
    Returns:
        Normalized kernel
    """
    if normalization == "none":
        return kernel
    
    kernel_normalized = kernel.clone()
    
    if normalization == "trace":
        # For kernels, use sum as proxy for trace
        kernel_sum = torch.sum(torch.abs(kernel_normalized)).clamp(min=1e-12)
        kernel_normalized = kernel_normalized / kernel_sum
        
    elif normalization in ["spectral", "frobenius"]:
        # Use Frobenius norm for kernels
        kernel_norm = torch.norm(kernel_normalized).clamp(min=1e-12)
        kernel_normalized = kernel_normalized / kernel_norm
    
    return kernel_normalized


def fractional_laplacian(L: torch.Tensor, alpha: float=0.5):
    L = 0.5*(L+L.T)
    evals, U = torch.linalg.eigh(L)
    evals = torch.clamp(evals, min=0.0)
    return U @ torch.diag(evals.pow(alpha)) @ U.T


@torch.no_grad()
def anisotropic_from_x0(n_side: int, x0_img: torch.Tensor, sigma: float=0.1, kind: str='8nn'):
    assert x0_img.shape==(n_side,n_side)
    n = n_side*n_side; device = x0_img.device; dtype=x0_img.dtype
    neigh4 = [(1,0),(-1,0),(0,1),(0,-1)]
    neigh8 = neigh4 + [(1,1),(1,-1),(-1,1),(-1,-1)]
    neigh = neigh8 if kind=='8nn' else neigh4
    rows, cols, vals = [], [], []
    def idx(i,j): return i*n_side + j
    for i in range(n_side):
        for j in range(n_side):
            p = idx(i,j); xi = x0_img[i,j]; deg_w = 0.0
            for di,dj in neigh:
                ii, jj = i+di, j+dj
                if 0<=ii<n_side and 0<=jj<n_side:
                    q = idx(ii,jj); xj = x0_img[ii,jj]
                    w = torch.exp(-((xi-xj)*(xi-xj))/(sigma*sigma + 1e-12))
                    rows.append(p); cols.append(q); vals.append(float(w)); deg_w += float(w)
            rows.append(p); cols.append(p); vals.append(deg_w)
    idxs = torch.tensor([rows, cols], dtype=torch.long, device=device)
    vals = torch.tensor(vals, dtype=dtype, device=device)
    L = torch.sparse_coo_tensor(idxs, vals, (n,n)).coalesce().to_dense()
    D = torch.diag(torch.diag(L)); W = D - L; W = 0.5*(W + W.T)
    return 0.5*((D - W) + (D - W).T)


def topk_sym(A: torch.Tensor, k: int):
    vals, idx = torch.topk(A, k=k, dim=1)
    m = torch.zeros_like(A, dtype=torch.bool); m.scatter_(1, idx, True)
    W = torch.where(m, A, torch.zeros_like(A))
    return 0.5*(W + W.T)


def learn_graph_from_batch(X_batch: torch.Tensor, k: int=8, l1: float=0.0, l2: float=1e-6):
    if X_batch.dim()==3: Tm,h,w = X_batch.shape; X = X_batch.reshape(Tm,h*w)
    elif X_batch.dim()==2: X = X_batch
    else: raise ValueError
    X = X - X.mean(0,keepdim=True); X = X/(X.std(0,keepdim=True)+1e-6)
    Z = X.T; G = Z @ Z.T; sq = (Z**2).sum(1,keepdim=True)
    dist2 = sq + sq.T - 2*G
    med = torch.median(dist2[dist2>0]).item() if (dist2>0).any() else 1.0
    A = torch.exp(-dist2/(med+1e-12)); A -= torch.diag(torch.diag(A))
    W = topk_sym(A, k=k)
    if l1>0: W = torch.clamp(W - l1*W.abs().max(), min=0.0)
    d = W.sum(1); D = torch.diag(d); L = D - W
    if l2>0: L = L + l2*torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
    return 0.5*(L + L.T)


def product_space_channel_laplacian(Ls: torch.Tensor, C: int=1, gamma_c: float=0.0):
    if C<=1: return Ls
    n = Ls.shape[0]
    Lc = torch.zeros((C,C), device=Ls.device)
    for i in range(C):
        if i>0: Lc[i,i-1] = Lc[i-1,i] = -1.0
        if i<C-1: Lc[i,i+1] = Lc[i+1,i] = -1.0
        Lc[i,i] = (1 if i>0 else 0) + (1 if i<C-1 else 0)
    I_n = torch.eye(n, device=Ls.device); I_c = torch.eye(C, device=Ls.device)
    return torch.kron(Ls, I_c) + gamma_c*torch.kron(I_n, Lc)


def hypergraph_laplacian_from_patches(x0_2d: torch.Tensor, n_clusters: int=16, patch: int=5, iters: int=10):
    device = x0_2d.device; n_side = x0_2d.shape[0]
    pad = patch//2
    Xp = F.unfold(x0_2d.unsqueeze(0).unsqueeze(0), kernel_size=patch, padding=pad).squeeze(0).T
    n, d = Xp.shape
    g = torch.Generator(device=device).manual_seed(0)
    idx_init = torch.randperm(n, generator=g, device=device)[:n_clusters]
    Ctr = Xp[idx_init].clone()
    for _ in range(iters):
        dist = torch.cdist(Xp, Ctr, p=2.0)
        lbl = torch.argmin(dist, dim=1)
        for k in range(n_clusters):
            sel = (lbl==k)
            if sel.any(): Ctr[k] = Xp[sel].mean(0)
    E = n_clusters
    Hm = torch.zeros((n,E), device=device)
    for i in range(n): Hm[i, lbl[i]] = 1.0
    De = torch.diag(Hm.sum(0).clamp(min=1e-12))
    W_e = torch.eye(E, device=device)
    Dv = torch.diag(Hm.sum(1).clamp(min=1e-12))
    Dv_mh = torch.diag(1.0/torch.sqrt(torch.diag(Dv)))
    L = torch.eye(n, device=device) - (Dv_mh @ Hm @ W_e @ torch.linalg.pinv(De) @ Hm.T @ Dv_mh)
    return 0.5*(L + L.T)


@torch.no_grad()
def make_batch_from_x0(x0_img: torch.Tensor, T: int=16, noise_std: float=0.02):
    Xb = []
    for _ in range(T):
        jitter = noise_std * torch.randn_like(x0_img)
        Xb.append((x0_img + jitter).clamp(x0_img.min(), x0_img.max()))
    return torch.stack(Xb,0)


@torch.no_grad()
def build_L_variant_CT(args, ct_op) -> torch.Tensor:
    """
    Build Laplacian variant for CT with support for dense/convolutional modes and normalization.
    Mode is automatically determined from laplacian_type.
    
    Returns:
        For conv mode: Convolutional kernel (1, 1, kH, kW) 
        For dense mode: Dense Laplacian matrix (n x n)
    """
    n_side = args.im_size; device = args.device
    name = (args.laplacian_type or 'grid_4nn').strip()
    normalization = getattr(args, 'laplacian_normalization', 'none')
    
    # Determine mode automatically from laplacian type
    conv_types = ['4nn_pos', '8nn_pos', '4nn_norm', '8nn_norm', 'diag4']
    mode = "conv" if name in conv_types else "dense"
    
    # Build reference image for some variants
    with torch.no_grad():
        one = torch.ones(1, 1, n_side, n_side, device=device)
        x0 = ct_op.pseudoinverse(ct_op.forward(one))
        x0_img = x0[0,0]
        x0_img = (x0_img - x0_img.min())/(x0_img.max()-x0_img.min() + 1e-12)
    
    # For convolutional mode, return normalized kernels directly
    if mode == "conv":
        if name in ['grid', 'grid4', 'grid_4nn']:
            kernel = conv_laplacians('4nn', device)
        elif name == 'grid_8nn':
            kernel = conv_laplacians('8nn', device)
        elif name == 'identity':
            kernel = conv_laplacians('identity', device)
        elif name in ['4nn_pos', '8nn_pos', '4nn_norm', '8nn_norm', 'diag4']:
            kernel = conv_laplacians(name, device)
        elif name in ['sym', 'rw']:
            # Use normalized versions of 4nn
            if name == 'sym':
                kernel = conv_laplacians('4nn_norm', device)
            else:  # rw
                kernel = conv_laplacians('4nn_norm', device)
        else:
            # For variants that don't have direct conv equivalents, fall back to 4nn
            print(f"[WARNING] Convolutional mode not supported for {name}, using 4nn")
            kernel = conv_laplacians('4nn', device)
        
        return normalize_laplacian_kernel(kernel, normalization)
    
    # Dense mode (original behavior)
    if name in ['grid','grid4','grid_4nn']:
        L = grid_laplacian(n_side, device=device, dtype=torch.float32, kind='4nn')
    elif name=='grid_8nn':
        L = grid_laplacian(n_side, device=device, dtype=torch.float32, kind='8nn')
    elif name=='identity':
        L = torch.eye(n_side*n_side, device=device, dtype=torch.float32)
    elif name=='shuffled':
        Lg = grid_laplacian(n_side, device=device, dtype=torch.float32, kind='4nn')
        g = torch.Generator(device=device).manual_seed(args.seed); n = Lg.shape[0]
        perm = torch.randperm(n, generator=g, device=device); L = Lg[perm][:,perm]
    elif name=='anisotropic':
        L = anisotropic_from_x0(n_side, x0_img, sigma=args.ani_sigma, kind='8nn')
    elif name=='misaligned':
        L = anisotropic_from_x0(n_side, torch.rot90(x0_img,1,(0,1)), sigma=args.ani_sigma, kind='8nn')
    elif name in ['learned','knn_patches']:
        L = learn_graph_from_batch(make_batch_from_x0(x0_img, T=args.learn_T, noise_std=args.learn_jitter),
                                    k=args.knn_k, l1=args.learn_l1, l2=args.learn_l2)
    elif name=='multiscale':
        L_local = grid_laplacian(n_side, device=device, dtype=torch.float32, kind='4nn')
        L_non   = learn_graph_from_batch(make_batch_from_x0(x0_img, T=args.learn_T, noise_std=args.learn_jitter),
                                          k=args.knn_k, l1=args.learn_l1, l2=args.learn_l2)
        L = L_local + L_non
    elif name=='sym':
        L = laplacian_normalize(grid_laplacian(n_side, device=device, dtype=torch.float32, kind='4nn'), mode='sym')
    elif name=='rw':
        L = laplacian_normalize(grid_laplacian(n_side, device=device, dtype=torch.float32, kind='4nn'), mode='rw')
    elif name=='fractional':
        L = fractional_laplacian(grid_laplacian(n_side, device=device, dtype=torch.float32, kind='4nn'), alpha=args.frac_alpha)
    elif name=='product':
        L = product_space_channel_laplacian(grid_laplacian(n_side, device=device, dtype=torch.float32, kind='4nn'), C=args.channels, gamma_c=args.prod_gamma_c)
    elif name=='hyper':
        L = hypergraph_laplacian_from_patches(x0_img, n_clusters=args.hyper_k, patch=args.knn_patch, iters=args.learn_iters)
    elif name in ['L_base','base']:
        L = L_base_laplacian(n_side, device=device, dtype=torch.float32, w_far=getattr(args, 'base_wfar', 0.7), trace_normalize=True)
    else:
        L = grid_laplacian(n_side, device=device, dtype=torch.float32, kind='4nn')
    
    # Make symmetric and apply normalization
    L = 0.5*(L + L.T)
    L = normalize_laplacian_matrix(L, normalization)
    
    return L

