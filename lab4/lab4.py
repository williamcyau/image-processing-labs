import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'lab3'))

import torch
from lab3 import forward, adjoint, load_pgm, load_kernel

def rho(delta, sigma_x, p, q, T):
    """Return the QGGMRF potential of equation (3), elementwise.

    delta    float64 tensor of any shape
    returns  float64 tensor of the same shape
    """
    abs_delta = torch.abs(delta)
    r = torch.abs(delta) / (T * sigma_x)
    r_pow = torch.pow(r, q - p)
    return (torch.pow(abs_delta, p) / (p * torch.pow(sigma_x, p))) * (r_pow / (1 + r_pow))


def rho_prime(delta, sigma_x, p, q, T):
    """Return the QGGMRF influence function of equation (4), elementwise.

    delta    float64 tensor of any shape
    returns  float64 tensor of the same shape
    """
    abs_delta = torch.abs(delta)
    r = abs_delta / (T * sigma_x)
    r_pow = torch.pow(r, q - p)
    numerator = r_pow * (q / p + r_pow)
    denominator = torch.pow(1 + r_pow, 2)
    return (torch.pow(abs_delta, p - 1) / torch.pow(sigma_x, p)) * (numerator / denominator) * torch.sign(delta)