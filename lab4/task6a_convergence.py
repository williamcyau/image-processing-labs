import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'lab3'))

from lab3 import load_pgm, load_kernel, set_seed
from lab4 import reconstruct


def run_task6a():
    """Run baseline QGGMRF reconstruction and verify cost decreases.

    Task 6a: Check that the cost decreases at every iteration by running
    the baseline QGGMRF and plotting the true cost against iteration number.
    """
    # Set random seed for reproducibility
    set_seed(42)

    # Load data
    y = load_pgm('./kodim23.pgm')
    a = load_kernel('./levin09_kernels/levin09_kernel1.txt')

    # QGGMRF parameters
    p = torch.tensor(2.0, dtype=torch.float64)
    q = torch.tensor(1.2, dtype=torch.float64)
    T = torch.tensor(1.0, dtype=torch.float64)
    sigma_x = torch.tensor(0.2, dtype=torch.float64)
    sigma_w = torch.tensor(1.0, dtype=torch.float64)

    # Reconstruction parameters
    num_iters = 50
    omega = torch.tensor(1.0, dtype=torch.float64)  # Standard steepest descent

    print("Running baseline QGGMRF reconstruction...")
    x_recon, cost_history = reconstruct(y, a, sigma_w, sigma_x, p, q, T, num_iters, omega)

    # Check if cost decreases at every iteration
    cost_decreases = all(cost_history[i+1] <= cost_history[i] for i in range(len(cost_history)-1))

    # Print statistics
    print(f"Initial cost: {cost_history[0]:.6f}")
    print(f"Final cost: {cost_history[-1]:.6f}")
    print(f"Cost decrease: {cost_history[0] - cost_history[-1]:.6f}")
    print(f"Cost decreases at every iteration: {cost_decreases}")

    # Check for any increases
    for i in range(len(cost_history)-1):
        if cost_history[i+1] > cost_history[i]:
            print(f"WARNING: Cost increased at iteration {i}: {cost_history[i]} -> {cost_history[i+1]}")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    iterations = list(range(len(cost_history)))
    ax.plot(iterations, cost_history, 'b-', linewidth=2, marker='o', markersize=4)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('True Cost f(x)', fontsize=12)
    ax.set_title('Convergence: True Cost vs Iteration (Baseline QGGMRF)', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('task6a_convergence.png', dpi=150, bbox_inches='tight')
    print("Plot saved as task6a_convergence.png")
    plt.show()

    return cost_history, cost_decreases


if __name__ == "__main__":
    cost_history, cost_decreases = run_task6a()
