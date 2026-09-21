import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from lab3 import set_seed, load_pgm, load_kernel, forward, cost, cost_grad, normalized_rmse, prior_grad


def main():
    # Load original image
    x = load_pgm("kodim23.pgm")

    # Load blur kernel
    a = load_kernel("levin09_kernels/levin09_kernel1.txt")

    # Parameters
    sigma_w = 0.02
    sigma_x = 0.05
    nrmse_threshold = 0.002
    max_iters = 1000

    # Create blurred and noisy image: y = A*x + w
    y_blurred = forward(x, a)
    set_seed()
    w = torch.randn_like(x) * sigma_w
    y = y_blurred + w
    y = torch.clamp(y, 0, 1)

    print("Deconvolution with 4× Larger Step Size")
    print("=" * 60)
    print(f"Kernel shape: {a.shape}")
    print(f"Noise std (sigma_w): {sigma_w}")
    print(f"Prior std (sigma_x): {sigma_x}")
    print("=" * 60)

    # Initialize x_hat = y
    x_hat = y.clone().detach()
    x_prev = y.clone().detach()

    cost_history = []
    nrmse_history = []
    iteration = 0

    # Step size: 4x larger than standard
    alpha_standard = 1.0 / (sigma_w ** (-2) + 2 * sigma_x ** (-2))
    alpha = 4.0 * alpha_standard

    print(f"\nStandard step size (alpha): {alpha_standard:.6e}")
    print(f"Enlarged step size (4x):   {alpha:.6e}")
    print("\nGradient Descent with Early Stopping")
    print("-" * 60)

    # Gradient descent with early stopping
    diverged = False
    while iteration < max_iters:
        # Compute cost and gradient using imported functions
        c = cost(x_hat, y, a, sigma_w, sigma_x)
        cost_history.append(c.item())

        grad = cost_grad(x_hat, y, a, sigma_w, sigma_x)

        # Update
        x_hat = x_hat - alpha * grad
        x_hat = torch.clamp(x_hat, 0, 1)

        # Check for NaN (numerical instability)
        if torch.isnan(x_hat).any():
            print(f"\n⚠ Numerical instability at iteration {iteration + 1}")
            print(f"  Step size α = {alpha:.6e} is too large")
            cost_history.pop()  # remove the cost that led to NaN
            diverged = True
            break

        # Compute NRMSE
        nrmse = normalized_rmse(x_hat, x_prev)
        nrmse_history.append(nrmse.item())

        # Print progress every 5 iterations or at convergence
        if iteration % 5 == 0 or nrmse < nrmse_threshold or iteration < 3:
            print(f"  Iteration {iteration + 1:3d}: Cost = {c.item():.2f}, NRMSE = {nrmse.item():.6e}")

        # Check for divergence (cost increasing rapidly)
        if iteration > 10 and cost_history[-1] > 10 * cost_history[0]:
            print(f"\n⚠ Divergence detected at iteration {iteration + 1}")
            print(f"  Cost exploding: {cost_history[0]:.2f} → {cost_history[-1]:.2f}")
            print(f"  Step size α = {alpha:.6e} exceeds stability threshold")
            diverged = True
            break

        # Check stopping criterion
        if nrmse < nrmse_threshold and iteration > 0:
            print(f"\n✓ Converged at iteration {iteration + 1}")
            print(f"  Final NRMSE: {nrmse.item():.6e}")
            break

        x_prev = x_hat.clone()
        iteration += 1

    total_iterations = iteration + 1
    print(f"  Total iterations: {total_iterations}")

    # =====================================================
    # Save Cost Plot
    # =====================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(cost_history, "b-", linewidth=2.5, label="Cost function")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cost", fontsize=12)
    ax.set_title(f"Cost vs Iteration (4× Step Size, $\\sigma_x = {sigma_x}$)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add annotation for convergence point
    if total_iterations < len(cost_history):
        conv_idx = total_iterations - 1
        ax.axvline(x=conv_idx, color="r", linestyle="--", linewidth=1.5, alpha=0.7, label="Convergence")
        ax.plot(conv_idx, cost_history[conv_idx], "ro", markersize=8)

    plt.tight_layout()
    plt.savefig("d10_cost_large_stepsize.png", dpi=150, bbox_inches="tight")
    print("\n✓ Figure saved: d10_cost_large_stepsize.png")

    # =====================================================
    # Print Timing Analysis
    # =====================================================
    print("\n" + "=" * 60)
    print("Performance Analysis")
    print("=" * 60)

    print(f"Iterations to convergence: {total_iterations}")
    print(f"Cost reduction: {cost_history[0]:.2f} → {cost_history[-1]:.2f}")
    print(f"Cost decrease: {cost_history[0] - cost_history[-1]:.2f}")

    # Wall-clock time for 50 iterations with large step size
    x_hat_test = y.clone().detach()
    start_time = time.time()
    for i in range(50):
        c = cost(x_hat_test, y, a, sigma_w, sigma_x)
        grad = cost_grad(x_hat_test, y, a, sigma_w, sigma_x)
        x_hat_test = x_hat_test - alpha * grad
        x_hat_test = torch.clamp(x_hat_test, 0, 1)
    elapsed = time.time() - start_time

    print(f"\nWall-clock time (50 iterations): {elapsed:.4f} seconds")

    print("=" * 60)


if __name__ == "__main__":
    main()
