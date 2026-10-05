import sys
from pathlib import Path
import torch
import time
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'lab3'))

from lab3 import load_pgm, load_kernel, forward, set_seed
from lab4 import reconstruct, true_cost


def compute_rmse(x_true, x_recon):
    """Compute RMSE between true and reconstructed images."""
    diff = x_true - x_recon
    mse = (diff ** 2).mean()
    rmse = torch.sqrt(mse).item()
    return rmse


def run_d10():
    """Generate D10 deliverable: table of results for all 6 runs (R1-R6)."""

    set_seed(42)

    # Load data
    x_true = load_pgm('./kodim23.pgm')
    a = load_kernel('./levin09_kernels/levin09_kernel1.txt')

    # Create blurred and noisy observation
    y_blurred = forward(x_true, a)
    w = torch.randn_like(x_true) * 0.02
    y = y_blurred + w
    y = torch.clamp(y, 0, 1)

    sigma_w = torch.tensor(0.02, dtype=torch.float64)

    # Define the 6 runs
    runs = {
        'R1': {'prior': 'Gaussian', 'p': 2.0, 'q': 2.0, 'T': 100.0, 'sigma_x': 0.036},
        'R2': {'prior': 'QGGMRF', 'p': 2.0, 'q': 1.2, 'T': 1.0, 'sigma_x': 0.020},
        'R3': {'prior': 'QGGMRF', 'p': 2.0, 'q': 1.2, 'T': 1.0, 'sigma_x': 0.010},
        'R4': {'prior': 'QGGMRF', 'p': 2.0, 'q': 1.2, 'T': 1.0, 'sigma_x': 0.040},
        'R5': {'prior': 'Gaussian', 'p': 2.0, 'q': 2.0, 'T': 100.0, 'sigma_x': 0.018},
        'R6': {'prior': 'Gaussian', 'p': 2.0, 'q': 2.0, 'T': 100.0, 'sigma_x': 0.072},
    }

    # Collect results
    results_list = []

    for run_name in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']:
        params = runs[run_name]
        print(f"\nRunning {run_name}: {params['prior']} (σ_x = {params['sigma_x']})...")

        p = torch.tensor(params['p'], dtype=torch.float64)
        q = torch.tensor(params['q'], dtype=torch.float64)
        T = torch.tensor(params['T'], dtype=torch.float64)
        sigma_x = torch.tensor(params['sigma_x'], dtype=torch.float64)

        # Time the reconstruction
        start_time = time.time()
        x_recon, _ = reconstruct(y, a, sigma_w, sigma_x, p, q, T, num_iters=50, omega=torch.tensor(1.0, dtype=torch.float64))
        elapsed_time = time.time() - start_time

        # Compute final cost
        final_cost = true_cost(x_recon, y, a, sigma_w, sigma_x, p, q, T).item()

        # Compute RMSE
        rmse = compute_rmse(x_true, x_recon)

        print(f"  Final Cost: {final_cost:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  Wall Clock Time: {elapsed_time:.2f}s")

        results_list.append({
            'Run': run_name,
            'Prior': params['prior'],
            'σ_x': params['sigma_x'],
            'Final Cost': final_cost,
            'RMSE': rmse,
            'Time (s)': elapsed_time
        })

    # Create DataFrame
    df = pd.DataFrame(results_list)

    # Display table
    print("\n" + "="*90)
    print("D10 RESULTS TABLE")
    print("="*90)
    print(df.to_string(index=False))
    print("="*90)

    # Save table to CSV
    df.to_csv('task_d10_results.csv', index=False)
    print("\nTable saved to task_d10_results.csv")

    # Save table as formatted text
    with open('task_d10_results.txt', 'w') as f:
        f.write("D10: Reconstruction Results Summary\n")
        f.write("="*90 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "="*90 + "\n")
        f.write("\nMetrics:\n")
        f.write("  Final Cost: MAP cost value at final iteration\n")
        f.write("  RMSE: Root Mean Squared Error against x_true in [0,1] units\n")
        f.write("  Time (s): Wall clock time for reconstruction (50 iterations)\n")

    print("Table saved to task_d10_results.txt")

    return df


if __name__ == "__main__":
    run_d10()
