import numpy as np
from pathlib import Path


import torch


import torchvision
import torchvision.transforms as T


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)

def laplacian_8nn(w: int, h: int, device=None, dtype=None):
    size = w * h
    dtype = dtype or torch.float32
    device = device or torch.device("cpu")
    adj = torch.zeros((size, size), dtype=dtype, device=device)
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def idx(x, y):
        return y * w + x

    for y in range(h):
        for x in range(w):
            i = idx(x, y)
            for dx, dy in neighbors:
                xx, yy = x + dx, y + dy
                if 0 <= xx < w and 0 <= yy < h:
                    j = idx(xx, yy)
                    adj[i, j] = 1.0

    degree = torch.diag(adj.sum(dim=1))
    lap = degree - adj
    trace = torch.trace(lap)
    if trace > 0:
        lap = (size / trace) * lap
    return lap.float()


def hadamard(n):
    if n == 1:
        return np.array([[1]])
    else:
        h = hadamard(n // 2)
        return np.block([[h, h], [h, -h]])


def null_space_basis(H: torch.Tensor, q: int):
    m, n = H.shape
    # QR completa sobre H^T para tener Q_full (n×n)
    Q_full, _       = torch.linalg.qr(H.T, mode='complete')  # (n, n)
    nullspace_basis = Q_full[:, m:]                         # (n, q)
    return nullspace_basis

def generate_orthogonal_rows_qr(H: torch.Tensor, p: int):

    nullspace_basis = null_space_basis(H,p)

    # combinaciones ortonormales aleatorias dentro del nullspace
    P  = torch.randn(nullspace_basis.shape[1], p, device=H.device, dtype=H.dtype)
    U, _ = torch.linalg.qr(P)     # Q reducido: (n-m, p)
    # cada fila nueva
    S = U.T.matmul(nullspace_basis.T)  # (p, n)
    return S


def spc_matrix(rows: int, cols: int, device=None):
    H = hadamard(cols)
    H = torch.from_numpy(H).to(dtype=torch.float32, device=device)
    return H[:rows, :], H[rows:, :]


def build_nullspace_operator(H, L, q):
    n,m = H.shape
    N = null_space_basis(H,q)
    L = (L*n)/torch.trace(L)
    T = N.T @ (L @ N)
    eigvals, eigvecs = torch.linalg.eigh(T.double())
    eigvecs = eigvecs.float()    
    U_raw = N @ eigvecs[:, torch.argsort(eigvals)].float()
    S_full = U_raw.T
    return S_full




def null_projector(H):
    I = torch.eye(H.shape[1], dtype=H.dtype, device=H.device)
    return I - torch.linalg.pinv(H) @ H

def orthonormal_basis_from_rows(S):
    if S.numel() == 0:
        return torch.zeros((S.shape[1], 0), dtype=S.dtype, device=S.device)
    Q, _ = torch.linalg.qr(S.T, mode="reduced")
    return Q

def normalized_coverage(S_rows, Pn, X, mean):
    Xc = X - mean
    Xn = Xc @ Pn.T
    Cb = (Xn.T @ Xn) / max(1, X.shape[0])
    eps = torch.tensor(1e-12, dtype=Cb.dtype, device=Cb.device)
    trCb = torch.trace(Cb) + eps
    Qp = orthonormal_basis_from_rows(S_rows)
    if S_rows.shape[0] == 0:
        cov = torch.tensor(0.0, dtype=Cb.dtype, device=Cb.device)
    else:
        cov = torch.trace(Qp.T @ Cb @ Qp) / trCb
        # cov = torch.trace(S_rows @ Cb @ S_rows.T) / trCb
    return cov


def load_cifar10_split(width, height, n_train, n_val, n_test, device=None, grayscale=True):
    transforms = [T.Resize((height, width)), T.ToTensor()]
    if grayscale:
        transforms.insert(0, T.Grayscale())
    tfm = T.Compose(transforms)
    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    numel = width * height * (1 if grayscale else 3)

    def stack_subset(dataset, start, count):
        if count <= 0:
            return torch.zeros((0, numel), dtype=torch.float32)
        end = start + count
        if end > len(dataset):
            raise ValueError("Requested more CIFAR-10 images than available.")
        samples = []
        for idx in range(start, end):
            img, _ = dataset[idx]
            samples.append(img.view(-1).float())
        return torch.stack(samples, dim=0)

    train = stack_subset(train_ds, 0, n_train)
    val = stack_subset(train_ds, n_train, n_val)
    test = stack_subset(test_ds, 0, n_test)

    if device is not None:
        train = train.to(device)
        val = val.to(device)
        test = test.to(device)
        
    return train, val, test



def load_external_S(path: Path, expected_cols: int, device: torch.device) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix in {".pt", ".pth"}:
        array = torch.load(path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported file extension '{suffix}' for S operator.")
    matrix = array.float() if isinstance(array, torch.Tensor) else torch.as_tensor(array, dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[1] != expected_cols:
        raise ValueError(f"Loaded S has shape {tuple(matrix.shape)}, expected (?, {expected_cols})")
    try:
        matrix = matrix.to(device)
    except Exception as e:
        raise ValueError(f"Could not move S to device {device}: {e}")
    return matrix




def generate_measurements(operator, X, noise_std):
    y = X @ operator.T
    if noise_std > 0:
        noise = torch.randn(y.shape, device=y.device, dtype=y.dtype)
        y = y + noise_std * noise
    return y
