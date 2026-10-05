import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'lab3'))

from lab4 import rho, rho_prime
from lab3 import cost, cost_grad, forward, adjoint, load_pgm, load_kernel


def rho_quadratic(delta, sigma_x):
    """Quadratic potential from equation (2)."""
    sigma_x = torch.tensor(sigma_x, dtype=torch.float64) if not isinstance(sigma_x, torch.Tensor) else sigma_x
    return (delta ** 2) / (2 * sigma_x ** 2)


def rho_prime_quadratic(delta, sigma_x):
    """Quadratic influence function from equation (2)."""
    sigma_x = torch.tensor(sigma_x, dtype=torch.float64) if not isinstance(sigma_x, torch.Tensor) else sigma_x
    return delta / (sigma_x ** 2)


def plot_task2a():
    """Plot potential and influence function for Task 2a.

    Compares quadratic potential (equation 2) with QGGMRF potential (equation 3)
    and their influence functions.
    """
    # Parameters from Task 2a (as torch tensors)
    p = torch.tensor(2.0, dtype=torch.float64)
    q = torch.tensor(1.2, dtype=torch.float64)
    T = torch.tensor(1.0, dtype=torch.float64)
    sigma_x = torch.tensor(0.2, dtype=torch.float64)

    # Create delta range
    delta = torch.linspace(-0.3, 0.3, 500, dtype=torch.float64)

    # Compute potentials
    rho_quad = rho_quadratic(delta, sigma_x)
    rho_qggmrf = rho(delta, sigma_x, p, q, T)

    # Compute influence functions
    rho_prime_quad = rho_prime_quadratic(delta, sigma_x)
    rho_prime_qggmrf = rho_prime(delta, sigma_x, p, q, T)

    # Convert to numpy for plotting
    delta_np = delta.numpy()
    rho_quad_np = rho_quad.numpy()
    rho_qggmrf_np = rho_qggmrf.numpy()
    rho_prime_quad_np = rho_prime_quad.numpy()
    rho_prime_qggmrf_np = rho_prime_qggmrf.numpy()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Potential functions
    ax1.plot(delta_np, rho_quad_np, 'b-', label='Quadratic (Eq. 2)', linewidth=2)
    ax1.plot(delta_np, rho_qggmrf_np, 'r-', label='QGGMRF (Eq. 3)', linewidth=2)
    ax1.set_xlabel(r'$\Delta$', fontsize=12)
    ax1.set_ylabel(r'$\rho(\Delta)$', fontsize=12)
    ax1.set_title(r'Potential Functions (p=2.0, q=1.2, T=1.0, $\sigma_x$=0.2)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Influence functions
    ax2.plot(delta_np, rho_prime_quad_np, 'b-', label="Quadratic (Eq. 2)", linewidth=2)
    ax2.plot(delta_np, rho_prime_qggmrf_np, 'r-', label="QGGMRF (Eq. 4)", linewidth=2)
    ax2.set_xlabel(r'$\Delta$', fontsize=12)
    ax2.set_ylabel(r"$\rho'(\Delta)$", fontsize=12)
    ax2.set_title(r"Influence Functions (p=2.0, q=1.2, T=1.0, $\sigma_x$=0.2)", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('task2a_potentials_and_influences.png', dpi=150, bbox_inches='tight')
    print("Plot saved as task2a_potentials_and_influences.png")
    plt.show()


def plot_task4b():
    """Plot QGGMRF potential with touching surrogates for Task 4b.

    Plots the potential along with two surrogate functions that touch
    at Delta' = 0.02 and Delta' = 0.15.
    """
    # Parameters from Task 4b (as torch tensors)
    p = torch.tensor(2.0, dtype=torch.float64)
    q = torch.tensor(1.2, dtype=torch.float64)
    T = torch.tensor(1.0, dtype=torch.float64)
    sigma_x = torch.tensor(0.2, dtype=torch.float64)

    # Create delta range
    delta = torch.linspace(-0.3, 0.3, 500, dtype=torch.float64)

    # Compute potential
    rho_val = rho(delta, sigma_x, p, q, T)

    # Compute surrogates for two delta prime values
    delta_prime_1 = torch.tensor(0.02, dtype=torch.float64)
    delta_prime_2 = torch.tensor(0.15, dtype=torch.float64)

    # Surrogate function: (rho'(delta') / (2*delta')) * (delta^2 - delta'^2) + rho(delta')
    def compute_surrogate(delta, delta_prime):
        rho_prime_val = rho_prime(delta_prime, sigma_x, p, q, T)
        rho_val_at_prime = rho(delta_prime, sigma_x, p, q, T)
        return (rho_prime_val / (2 * delta_prime)) * (delta ** 2 - delta_prime ** 2) + rho_val_at_prime

    surrogate_1 = compute_surrogate(delta, delta_prime_1)
    surrogate_2 = compute_surrogate(delta, delta_prime_2)

    # Convert to numpy for plotting
    delta_np = delta.numpy()
    rho_np = rho_val.numpy()
    surrogate_1_np = surrogate_1.numpy()
    surrogate_2_np = surrogate_2.numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot potential
    ax.plot(delta_np, rho_np, 'b-', label=r'$\rho(\Delta)$ (Eq. 3)', linewidth=2.5)

    # Plot surrogates
    ax.plot(delta_np, surrogate_1_np, 'r--', label=r"Surrogate at $\Delta' = 0.02$", linewidth=2)
    ax.plot(delta_np, surrogate_2_np, 'g--', label=r"Surrogate at $\Delta' = 0.15$", linewidth=2)

    # Mark the touching points for surrogate 1
    ax.plot([-0.02, 0.02], [rho(torch.tensor(0.02, dtype=torch.float64), sigma_x, p, q, T).item(),
                            rho(torch.tensor(0.02, dtype=torch.float64), sigma_x, p, q, T).item()],
            'ro', markersize=8, markeredgewidth=1.5, markeredgecolor='darkred', markerfacecolor='red')

    # Mark the touching points for surrogate 2
    ax.plot([-0.15, 0.15], [rho(torch.tensor(0.15, dtype=torch.float64), sigma_x, p, q, T).item(),
                            rho(torch.tensor(0.15, dtype=torch.float64), sigma_x, p, q, T).item()],
            'go', markersize=8, markeredgewidth=1.5, markeredgecolor='darkgreen', markerfacecolor='green')

    ax.set_xlabel(r'$\Delta$', fontsize=13)
    ax.set_ylabel(r'$\rho(\Delta)$', fontsize=13)
    ax.set_title(r'QGGMRF Potential with Touching Surrogates (p=2.0, q=1.2, T=1.0, $\sigma_x$=0.2)', fontsize=13)
    ax.legend(fontsize=11, loc='upper center')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('task4b_surrogate.png', dpi=150, bbox_inches='tight')
    print("Plot saved as task4b_surrogate.png")
    plt.show()


if __name__ == "__main__":
    plot_task2a()
    plot_task4b()
