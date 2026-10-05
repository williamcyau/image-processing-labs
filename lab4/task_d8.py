import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'lab3'))

from lab3 import load_pgm, load_kernel, forward, set_seed
from lab4 import reconstruct


def compute_rmse(x_true, x_recon):
    """Compute RMSE between true and reconstructed images."""
    diff = x_true - x_recon
    mse = (diff ** 2).mean()
    rmse = torch.sqrt(mse).item()
    return rmse


def run_d8():
    """Generate D8 deliverable: sigma_x parameter study with 6 reconstructions."""

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

    # Run all reconstructions
    results = {}
    for run_name, params in runs.items():
        print(f"Running {run_name}: {params['prior']} (σ_x = {params['sigma_x']})...")

        p = torch.tensor(params['p'], dtype=torch.float64)
        q = torch.tensor(params['q'], dtype=torch.float64)
        T = torch.tensor(params['T'], dtype=torch.float64)
        sigma_x = torch.tensor(params['sigma_x'], dtype=torch.float64)

        x_recon, _ = reconstruct(y, a, sigma_w, sigma_x, p, q, T, num_iters=50, omega=torch.tensor(1.0, dtype=torch.float64))

        rmse = compute_rmse(x_true, x_recon)
        print(f"  RMSE: {rmse:.6f}")

        results[run_name] = {
            'image': np.clip(x_recon.detach().numpy(), 0, 1),
            'rmse': rmse,
            'params': params
        }

    # Convert x_true for visualization
    x_true_np = np.clip(x_true.numpy(), 0, 1)

    # Determine common grayscale limits
    vmin = x_true_np.min()
    vmax = x_true_np.max()
    for run_data in results.values():
        vmin = min(vmin, run_data['image'].min())
        vmax = max(vmax, run_data['image'].max())

    # Create Figure 1: All 6 full reconstructions
    fig1, axes1 = plt.subplots(2, 3, figsize=(16, 11))
    fig1.suptitle('σ_x Parameter Study: Full Reconstructions', fontsize=15, fontweight='bold', y=0.995)

    run_order = ['R3', 'R2', 'R4', 'R5', 'R1', 'R6']  # Order: QGGMRF (center/2, center, center×2), Gaussian (center/2, center, center×2)

    for idx, (ax, run_name) in enumerate(zip(axes1.flat, run_order)):
        result = results[run_name]
        prior = result['params']['prior']
        sigma_x = result['params']['sigma_x']
        rmse = result['rmse']

        im = ax.imshow(result['image'], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'{run_name}: {prior}\nσ_x = {sigma_x:.3f}, RMSE = {rmse:.4f}',
                    fontsize=11, fontweight='bold')
        ax.axis('off')

    # Add shared colorbar
    cbar_ax = fig1.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Intensity', fontsize=11, fontweight='bold')

    plt.subplots_adjust(right=0.91, wspace=0.25, hspace=0.35)
    plt.savefig('task_d8_full_reconstructions.png', dpi=150, bbox_inches='tight')
    print("Figure 1 saved: task_d8_full_reconstructions.png")
    plt.close()

    # Extract crop region (same as D6: rows 192-320, columns 128-256)
    crop_r_start, crop_r_end = 192, 320
    crop_c_start, crop_c_end = 128, 256

    x_true_crop = x_true_np[crop_r_start:crop_r_end, crop_c_start:crop_c_end]

    # Create Figure 2: All 6 crops
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 11))
    fig2.suptitle('σ_x Parameter Study: Crop Comparison (rows 192–320, cols 128–256)',
                 fontsize=15, fontweight='bold', y=0.995)

    for idx, (ax, run_name) in enumerate(zip(axes2.flat, run_order)):
        result = results[run_name]
        crop = result['image'][crop_r_start:crop_r_end, crop_c_start:crop_c_end]
        prior = result['params']['prior']
        sigma_x = result['params']['sigma_x']
        rmse = result['rmse']

        im = ax.imshow(crop, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'{run_name}: {prior}\nσ_x = {sigma_x:.3f}, RMSE = {rmse:.4f}',
                    fontsize=11, fontweight='bold')
        ax.axis('off')

    # Add shared colorbar
    cbar_ax2 = fig2.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar2 = plt.colorbar(im, cax=cbar_ax2)
    cbar2.set_label('Intensity', fontsize=11, fontweight='bold')

    plt.subplots_adjust(right=0.91, wspace=0.25, hspace=0.35)
    plt.savefig('task_d8_crops.png', dpi=150, bbox_inches='tight')
    print("Figure 2 saved: task_d8_crops.png")
    plt.close()

    print("\nSummary:")
    for run_name in run_order:
        result = results[run_name]
        print(f"  {run_name}: {result['params']['prior']:10s} σ_x = {result['params']['sigma_x']:.3f}  RMSE = {result['rmse']:.6f}")


if __name__ == "__main__":
    run_d8()
