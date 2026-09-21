import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from lab3 import set_seed, load_pgm, a_identity, cost, cost_grad, normalized_rmse, prior_grad


def main():
    # Load original image
    x = load_pgm("kodim23.pgm")

    # Parameters
    sigma_w = 0.05
    sigma_x_values = [0.01, 0.05, 0.25]
    nrmse_threshold = 0.002
    max_iters = 1000

    # Create noisy image: y = x + w
    set_seed()
    w = torch.randn_like(x) * sigma_w
    y = x + w
    y = torch.clamp(y, 0, 1)

    # Store results
    restored_images = []
    convergence_data = {}

    print("Denoising Experiments with Identity Operator (A = I)")
    print("=" * 60)

    # Run gradient descent for each sigma_x
    for sigma_x in sigma_x_values:
        print(f"\nProcessing sigma_x = {sigma_x}")

        # Initialize x_hat = y
        x_hat = y.clone().detach()
        x_prev = y.clone().detach()

        cost_history = []
        nrmse_history = []
        iteration = 0

        # Step size
        alpha = 1.0 / (sigma_w ** (-2) + 2 * sigma_x ** (-2))

        # Gradient descent with early stopping
        while iteration < max_iters:
            # Compute cost and gradient using imported functions with identity kernel
            c = cost(x_hat, y, a_identity, sigma_w, sigma_x)
            cost_history.append(c.item())

            grad = cost_grad(x_hat, y, a_identity, sigma_w, sigma_x)

            # Update
            x_hat = x_hat - alpha * grad
            x_hat = torch.clamp(x_hat, 0, 1)

            # Compute NRMSE
            nrmse = normalized_rmse(x_hat, x_prev)
            nrmse_history.append(nrmse.item())

            # Check stopping criterion
            if nrmse < nrmse_threshold and iteration > 0:
                print(f"  Converged at iteration {iteration + 1}")
                print(f"  Final NRMSE: {nrmse.item():.6f}")
                break

            x_prev = x_hat.clone()
            iteration += 1

        restored_images.append(x_hat)
        convergence_data[sigma_x] = {
            "cost": cost_history,
            "nrmse": nrmse_history,
            "iterations": iteration + 1,
        }

        print(f"  Total iterations: {iteration + 1}")

    # =====================================================
    # Deliverable 1: 4-panel figure (noisy + 3 restored)
    # =====================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Noisy image
    im = axes[0, 0].imshow(y.numpy(), cmap="gray_r", vmin=0, vmax=1)
    axes[0, 0].set_title("Noisy Image (y)", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # Restored images
    for idx, (sigma_x, x_restored) in enumerate(zip(sigma_x_values, restored_images)):
        ax = axes.flat[idx + 1]
        im = ax.imshow(x_restored.numpy(), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"Restored ($\\sigma_x = {sigma_x}$)", fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("d7_denoising_results.png", dpi=150, bbox_inches="tight")
    print("\n✓ Figure saved: d7_denoising_results.png")

    # =====================================================
    # Deliverable 2: Convergence analysis for sigma_x = 0.05
    # =====================================================
    sigma_x_mid = 0.05
    data = convergence_data[sigma_x_mid]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Panel 2a: Cost vs iteration
    axes[0].plot(data["cost"], "b-", linewidth=2)
    axes[0].set_xlabel("Iteration", fontsize=11)
    axes[0].set_ylabel("Cost", fontsize=11)
    axes[0].set_title(f"Cost vs Iteration ($\\sigma_x = {sigma_x_mid}$)", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Panel 2b: NRMSE vs iteration (log scale)
    axes[1].semilogy(data["nrmse"], "r-", linewidth=2, label="NRMSE")
    axes[1].axhline(
        y=0.002, color="k", linestyle="--", linewidth=1.5, label="Threshold (0.002)"
    )
    axes[1].set_xlabel("Iteration", fontsize=11)
    axes[1].set_ylabel("NRMSE", fontsize=11)
    axes[1].set_title(f"NRMSE vs Iteration ($\\sigma_x = {sigma_x_mid}$)", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("d7_convergence_analysis.png", dpi=150, bbox_inches="tight")
    print("✓ Figure saved: d7_convergence_analysis.png")

    # =====================================================
    # Deliverable 3: Wall-clock time for 50-iteration runs
    # =====================================================
    print("\n" + "=" * 60)
    print("Wall-clock time for one 50-iteration run:")
    print("=" * 60)

    for sigma_x in sigma_x_values:
        x_hat = y.clone().detach()
        alpha = 1.0 / (sigma_w ** (-2) + 2 * sigma_x ** (-2))

        start_time = time.time()
        for i in range(50):
            Ax = x_hat
            residual = Ax - y
            data_grad = residual / (sigma_w ** 2)
            grad = data_grad + prior_grad(x_hat, sigma_x)
            x_hat = x_hat - alpha * grad
            x_hat = torch.clamp(x_hat, 0, 1)
        elapsed = time.time() - start_time

        print(f"  sigma_x = {sigma_x:>5}: {elapsed:.4f} seconds")

    print("=" * 60)


if __name__ == "__main__":
    main()
