import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from lab3 import load_pgm, load_kernel, forward, adjoint, adjoint_autograd


def main():
    # Load image
    x = load_pgm("kodim23.pgm")

    # Load kernel
    a = load_kernel("levin09_kernels/levin09_kernel1.txt")


    torch.manual_seed(0)
    u = torch.randn(512, 768, dtype=torch.float64)
    v = torch.randn(512, 768, dtype=torch.float64)

    check1 = (adjoint(v, a) - adjoint_autograd(v, a)).abs().max()

    lhs = (forward(u, a) * v).sum()
    rhs = (u * adjoint(v, a)).sum()
    check2 = (lhs - rhs).abs() / lhs.abs()

    check3 = (forward(u, a) - adjoint(u, a)).abs().max()

    print("check 1", check1)
    print("check 2", check2)
    print("check 3", check3)

if __name__ == "__main__":
    main()
