import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

import deepinv as dinv
from deepinv.optim.optim_iterators.optim_iterator import OptimIterator, fStep
from deepinv.optim.optim_iterators.pgd import gStepPGD
from deepinv.unfolded import BaseUnfold

from utils_sr import psnr_fun



class NPN_PGD(dinv.models.Reconstructor):
    def __init__(self, data_fidelity, prior, stepsize, lambd, max_iter, gamma, gamma_decay=0.95, gamma_min=1e-4, gamma_graph=0.0):
        super().__init__()
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.stepsize = stepsize
        self.lambd = lambd
        self.max_iter = max_iter
        self.gamma = gamma
        self.gamma_decay = gamma_decay
        self.gamma_min = gamma_min
        self.gamma_graph = gamma_graph

    def forward(self, x0, y, ys, physics_h, physics_s, xgt=None,B=None, channel_wise=False, **kwargs):
        """Algorithm forward pass.

        :param torch.Tensor y: measurements.
        :param dinv.physics.Physics physics: measurement operator.
        :return: torch.Tensor: reconstructed image.
        """
        x_k = x0

        # Disable autodifferentiation, remove this if you want to unfold
        xks = []
        gamma = self.gamma
        with torch.no_grad():
            loop = tqdm(range(self.max_iter), desc="NPN-PGD iterations")
            for _ in loop:
                # if graph_regularizer:
                if B is not None:
                    if channel_wise:
                        xn = x_k.reshape(x_k.shape[0],  -1) @ B
                        xn = xn.reshape(x_k.shape)
                    else:
                        xn = x_k.reshape(x_k.shape[0], x_k.shape[1], -1) @ B
                        xn = xn.reshape(x_k.shape)
                    u = x_k - self.stepsize * (
                        self.data_fidelity.grad(x_k, y, physics_h)
                        + gamma * self.data_fidelity.grad(x_k, ys, physics_s)
                       ) -  self.gamma_graph/(torch.norm(xn.reshape(x_k.shape[0],-1),dim=1,keepdim=True).unsqueeze(1).unsqueeze(1))* xn  # Gradient step
                else: 
                    u = x_k - self.stepsize * (
                        self.data_fidelity.grad(x_k, y, physics_h)
                        + gamma * self.data_fidelity.grad(x_k, ys, physics_s)
                    )  # Gradient step
                
                x_k = self.prior.prox(
                    u, sigma_denoiser=self.lambd * self.stepsize
                )  # Proximal step
                xks.append(x_k)
                # Decrease gamma
                gamma = max(gamma * self.gamma_decay, self.gamma_min)
                if xgt is not None:
                    psnr = psnr_fun(x_k, xgt)
                loop.set_postfix({"gamma": gamma, "psnr": psnr.item() if xgt is not None else None}, refresh=False)
        return x_k, xks


def _expand_parameter(param, max_iter):
    """Helper to align scalar/list parameters with the number of unfolding steps."""
    if isinstance(param, (list, tuple)):
        if len(param) == max_iter:
            return list(param)
        if len(param) == 1:
            return list(param) * max_iter
        raise ValueError(
            f"Parameter list has {len(param)} entries but max_iter is {max_iter}."
        )
    return [param for _ in range(max_iter)]


def _build_gamma_schedule(gamma0, decay, gamma_min, max_iter):
    gamma = gamma0
    schedule = []
    for _ in range(max_iter):
        schedule.append(float(gamma))
        gamma = max(gamma * decay, gamma_min)
    return schedule


class NPNDataFidelity(dinv.optim.DataFidelity):
    """Couples two physics & measurement pairs into a single differentiable data-fidelity."""

    def __init__(self, primary, secondary=None):
        super().__init__()
        self.primary = primary
        self.secondary = secondary

    def fn(self, x, y, physics, **kwargs):
        y_main, y_side = self._split_target(y, "measurements")
        physics_main, physics_side = self._split_target(physics, "physics")
        if y_main is None or physics_main is None:
            raise ValueError("Primary measurements and physics must be provided.")
        loss = self.primary.fn(x, y_main, physics_main)
        if self.secondary and y_side is not None and physics_side is not None:
            loss = loss + self.secondary.fn(x, y_side, physics_side)
        return loss

    def grad(self, x, y, physics, npn_gamma=1.0, **kwargs):
        y_main, y_side = self._split_target(y, "measurements")
        physics_main, physics_side = self._split_target(physics, "physics")
        if y_main is None or physics_main is None:
            raise ValueError("Primary measurements and physics must be provided.")
        grad = self.primary.grad(x, y_main, physics_main)
        if (
            self.secondary
            and y_side is not None
            and physics_side is not None
            and npn_gamma is not None
        ):
            grad = grad + npn_gamma * self.secondary.grad(x, y_side, physics_side)
        return grad

    @staticmethod
    def _split_target(value, name):
        if isinstance(value, Mapping):
            return value.get("primary"), value.get("secondary")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) == 2:
                return value[0], value[1]
            # If a sequence of different length is provided, fall back to primary-only usage.
            return value, None
        try:
            if hasattr(value, "__getitem__") and len(value) == 2:
                return value[0], value[1]
        except TypeError:
            pass
        return value, None


class fStepNPN(fStep):
    """f-step with NPN coupling and optional graph regularization."""

    def forward(self, x, cur_data_fidelity, cur_params, y, physics, **kwargs):
        stepsize = cur_params["stepsize"]
        gamma = cur_params.get("npn_gamma", 0.0)
        grad = cur_data_fidelity.grad(x, y, physics, npn_gamma=gamma)
        x_next = x - stepsize * grad

        gamma_graph = cur_params.get("gamma_graph", 0.0)
        graph_operator = kwargs.get("graph_operator", None)
        if graph_operator is not None and gamma_graph is not None:
            x_next = x_next - self._graph_step(x, graph_operator, gamma_graph)
        return x_next

    @staticmethod
    def _graph_step(x, graph_operator, gamma_graph, eps=1e-8):
        if graph_operator is None:
            return torch.zeros_like(x)
        if not torch.is_tensor(graph_operator):
            graph_operator = torch.as_tensor(
                graph_operator, dtype=x.dtype, device=x.device
            )
        else:
            graph_operator = graph_operator.to(device=x.device, dtype=x.dtype)

        xn = x.reshape(x.shape[0], x.shape[1], -1)
        graph_term = torch.matmul(xn, graph_operator).reshape_as(x)
        norm = (
            graph_term.reshape(graph_term.shape[0], -1)
            .norm(dim=1, keepdim=True)
            .clamp_min(eps)
        )
        expand_shape = [graph_term.shape[0]] + [1] * (graph_term.ndim - 1)
        norm = norm.view(*expand_shape)
        normalized = graph_term / norm

        if not torch.is_tensor(gamma_graph):
            gamma_graph = torch.tensor(
                gamma_graph, dtype=x.dtype, device=x.device
            )
        else:
            gamma_graph = gamma_graph.to(dtype=x.dtype, device=x.device)

        return gamma_graph.reshape((1,) + (1,) * (normalized.ndim - 1)) * normalized


class NPNPGDIteration(OptimIterator):
    """PGD iterator equipped with neural proximal network regularization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.f_step = fStepNPN(**kwargs)
        self.g_step = gStepPGD(**kwargs)
        # if self.g_first:
        #     self.requires_grad_g = False
        # else:
        #     self.requires_prox_g = True


# class NPNPGDIterator(NPNPGDIteration):
#     """Alias kept for users expecting an *Iterator* suffix."""

#     pass

def build_pgd_unfold(
    data_fidelity,
    prior,
    *,
    max_iter=10,
    stepsize=0.5,
    lambd=0.01,
    trainable_params=("stepsize", "lambda"),
    device=torch.device("cpu"),
    custom_init= None,
):
    """Factory returning an unfolded reconstructor with the custom PGD iterator."""
    iterator = dinv.optim.optim_iterators.pgd.PGDIteration()
    params_algo = {
        "stepsize": _expand_parameter(stepsize, max_iter),
        "lambda": _expand_parameter(lambd, max_iter),
    }

    return BaseUnfold(
        iterator=iterator,
        params_algo=params_algo,
        data_fidelity=[data_fidelity],
        prior=prior,
        max_iter=max_iter,
        trainable_params=list(trainable_params),
        device=device,
        custom_init=custom_init
        
    )

def build_npn_pgd_unfold(
    primary_fid,
    secondary_fid,
    prior,
    *,
    max_iter=10,
    stepsize=0.5,
    lambd=0.01,
    g_param=0.01,
    gamma0=1.0,
    gamma_decay=0.95,
    gamma_min=1e-4,
    gamma_graph=0.0,
    trainable_params=("stepsize", "lambda", "npn_gamma"),
    device=torch.device("cpu"),
    custom_init= None,
):
    """Factory returning an unfolded reconstructor with the custom NPN PGD iterator."""
    iterator = NPNPGDIteration()
    params_algo = {
        "stepsize": _expand_parameter(stepsize, max_iter),
        "lambda": _expand_parameter(lambd, max_iter),
        "npn_gamma": _build_gamma_schedule(gamma0, gamma_decay, gamma_min, max_iter),
        "gamma_graph": _expand_parameter(gamma_graph, max_iter),
    }
    if g_param is not None:
        params_algo["g_param"] = _expand_parameter(g_param, max_iter)

    data_fidelity = [NPNDataFidelity(primary_fid, secondary_fid)]
    

    return BaseUnfold(
        iterator=iterator,
        params_algo=params_algo,
        data_fidelity=data_fidelity,
        prior=prior,
        max_iter=max_iter,
        trainable_params=list(trainable_params),
        device=device,
        custom_init=custom_init
        
    )


class MyPGD(dinv.models.Reconstructor):
    def __init__(self, data_fidelity, prior, stepsize, lambd, max_iter):
        super().__init__()
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.stepsize = stepsize
        self.lambd = lambd
        self.max_iter = max_iter

    def forward(self, x0, y, physics,xgt=None, **kwargs):
        """Algorithm forward pass.

        :param torch.Tensor y: measurements.
        :param dinv.physics.Physics physics: measurement operator.
        :return: torch.Tensor: reconstructed image.
        """
        x_k = x0

        # Disable autodifferentiation, remove this if you want to unfold
        xks = []
        with torch.no_grad():
            loop = tqdm(range(self.max_iter), desc="PGD iterations")
            for _ in loop:
                u = x_k - self.stepsize * self.data_fidelity.grad(
                    x_k, y, physics
                )  # Gradient step
                x_k = self.prior.prox(
                    u, sigma_denoiser=self.lambd * self.stepsize
                )  # Proximal step
                xks.append(x_k)
                # Add fista acceleration here if desired
                
                    
                
                    
                if xgt is not None:
                    psnr = psnr_fun(x_k, xgt)
                loop.set_postfix({"psnr": psnr.item() if xgt is not None else None}, refresh=False)
        return x_k,xks
    




# class MyADMM()
