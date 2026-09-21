import torch
from lab3 import load_pgm, load_kernel, forward, prior_grad, cost, cost_grad

if __name__ == "__main__":
    # Load image and kernel
    x = load_pgm("kodim23.pgm")
    a = load_kernel("levin09_kernels/levin09_kernel1.txt")

    # Parameters
    sigma_w = 1.0
    sigma_x = 1.0

    # Seed for reproducibility
    torch.manual_seed(0)
    u = torch.randn(512, 768, dtype=torch.float64)
    v = torch.randn(512, 768, dtype=torch.float64)

    # Create y_small with compatible size (32x32)
    x_small = torch.randn(32, 32, dtype=torch.float64)
    y_small = forward(x_small, a)

    # Check 4: Verify prior_grad is self-adjoint
    check4 = ((u * prior_grad(v, sigma_x)).sum()
              - (v * prior_grad(u, sigma_x)).sum()).abs()

    # Check 5: Verify cost_grad matches autograd
    z = torch.randn(32, 32, dtype=torch.float64, requires_grad=True)
    auto = torch.autograd.grad(cost(z, y_small, a, sigma_w, sigma_x), z)[0]
    check5 = (auto - cost_grad(z.detach(), y_small, a, sigma_w, sigma_x)).abs().max()

    # Print results
    print("check 4", check4)
    print("check 5", check5)
