import torch
from lab3 import load_pgm, load_kernel, forward, cost, cost_grad, gradient_descent

if __name__ == "__main__":
    # Load image and kernel
    x = load_pgm("kodim23.pgm")
    a = load_kernel("levin09_kernels/levin09_kernel1.txt")

    # Create blurred image
    y = forward(x, a)

    # Parameters
    sigma_w = 1.0
    sigma_x = 1.0

    # Test gradient descent on a small region
    x_small = x[:64, :64]
    y_small = forward(x_small, a)

    # Run gradient descent
    x_hat, cost_history = gradient_descent(y_small, a, sigma_w, sigma_x, num_iters=10)

    print("Gradient Descent Results:")
    print(f"Initial cost: {cost_history[0]:.6f}")
    print(f"Final cost: {cost_history[-1]:.6f}")
    print(f"Cost decrease: {cost_history[0] - cost_history[-1]:.6f}")
    print(f"Number of iterations: {len(cost_history) - 1}")
