import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from lab4 import rho, rho_prime


def rho_quadratic(delta, sigma_x):
    """Quadratic potential from equation (2)."""
    return (delta ** 2) / (2 * sigma_x ** 2)


def rho_prime_quadratic(delta, sigma_x):
    """Quadratic influence function from equation (2)."""
    return delta / (sigma_x ** 2)


def plot_task2a():
    """Plot potential and influence function for Task 2a.

    Compares quadratic potential (equation 2) with QGGMRF potential (equation 3)
    and their influence functions.
    """
    # Parameters from Task 2a
    p, q, T, sigma_x = 2.0, 1.2, 1.0, 0.2

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


if __name__ == "__main__":
    plot_task2a()
