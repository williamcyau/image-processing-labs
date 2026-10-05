import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'lab3'))

from lab3 import load_pgm, load_kernel, forward, set_seed
from lab4 import reconstruct


def run_d6():
    """Generate D6 deliverable: compare Gaussian and QGGMRF reconstructions.

    Creates 4 images: x_true, y, R1 (Gaussian), R2 (QGGMRF)
    Plus their 128x128 crops from a sharp edge region.
    """
    # Set seed for reproducibility
    set_seed(42)

    # Load data
    x_true = load_pgm('./kodim23.pgm')
    a = load_kernel('./levin09_kernels/levin09_kernel1.txt')

    # Create blurred and noisy observation y = A * x_true + w
    y_blurred = forward(x_true, a)
    w = torch.randn_like(x_true) * 0.02  # Gaussian noise with sigma_w = 0.02
    y = y_blurred + w
    y = torch.clamp(y, 0, 1)

    sigma_w = torch.tensor(0.02, dtype=torch.float64)

    print("Running R1 (Gaussian MRF) reconstruction...")
    # R1: Gaussian MRF with p=2, q=2, T=100
    # Use center sigma_x = 0.036
    p1 = torch.tensor(2.0, dtype=torch.float64)
    q1 = torch.tensor(2.0, dtype=torch.float64)
    T1 = torch.tensor(100.0, dtype=torch.float64)
    sigma_x1 = torch.tensor(0.036, dtype=torch.float64)

    x_r1, _ = reconstruct(y, a, sigma_w, sigma_x1, p1, q1, T1, num_iters=50, omega=torch.tensor(1.0, dtype=torch.float64))

    print("Running R2 (QGGMRF) reconstruction...")
    # R2: QGGMRF with T=1, center sigma_x = 0.020
    p2 = torch.tensor(2.0, dtype=torch.float64)
    q2 = torch.tensor(1.2, dtype=torch.float64)
    T2 = torch.tensor(1.0, dtype=torch.float64)
    sigma_x2 = torch.tensor(0.020, dtype=torch.float64)

    x_r2, _ = reconstruct(y, a, sigma_w, sigma_x2, p2, q2, T2, num_iters=50, omega=torch.tensor(1.0, dtype=torch.float64))

    # Convert to numpy (unconstrained from optimization)
    x_true_np = x_true.numpy()
    y_np = y.detach().numpy()
    x_r1_np = x_r1.detach().numpy()
    x_r2_np = x_r2.detach().numpy()

    # Clamp for visualization only
    x_true_vis = np.clip(x_true_np, 0, 1)
    y_vis = np.clip(y_np, 0, 1)
    x_r1_vis = np.clip(x_r1_np, 0, 1)
    x_r2_vis = np.clip(x_r2_np, 0, 1)

    # Find a good crop region with high contrast edge
    crop_size = 128
    H, W = x_true_np.shape

    # Find region with high gradient magnitude
    best_score = -np.inf
    best_r, best_c = 0, 0

    for r in range(0, H - crop_size, 32):
        for c in range(0, W - crop_size, 32):
            crop = x_true_vis[r:r+crop_size, c:c+crop_size]
            grad_r = np.diff(crop, axis=0)
            grad_c = np.diff(crop, axis=1)
            grad_mag = np.sqrt(grad_r[:, :-1]**2 + grad_c[:-1, :]**2).mean()

            if grad_mag > best_score:
                best_score = grad_mag
                best_r, best_c = r, c

    crop_coords = (best_r, best_c)
    print(f"Using crop region at ({best_r}, {best_c})")

    # Extract crops
    x_true_crop = x_true_vis[best_r:best_r+crop_size, best_c:best_c+crop_size]
    y_crop = y_vis[best_r:best_r+crop_size, best_c:best_c+crop_size]
    x_r1_crop = x_r1_vis[best_r:best_r+crop_size, best_c:best_c+crop_size]
    x_r2_crop = x_r2_vis[best_r:best_r+crop_size, best_c:best_c+crop_size]

    # Determine common grayscale limits for visualization
    vmin = min(x_true_vis.min(), y_vis.min(), x_r1_vis.min(), x_r2_vis.min())
    vmax = max(x_true_vis.max(), y_vis.max(), x_r1_vis.max(), x_r2_vis.max())

    # Create figure with 2 rows, 4 columns + shared colorbar
    fig = plt.figure(figsize=(17, 8))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.05], hspace=0.25, wspace=0.25)

    axes_list = []
    for i in range(2):
        for j in range(4):
            axes_list.append(fig.add_subplot(gs[i, j]))

    # Row 1: Full images
    im0 = axes_list[0].imshow(x_true_vis, cmap='gray', vmin=vmin, vmax=vmax)
    axes_list[0].set_title('$x_{true}$ (Original)', fontsize=11, fontweight='bold')
    axes_list[0].axis('off')

    axes_list[1].imshow(y_vis, cmap='gray', vmin=vmin, vmax=vmax)
    axes_list[1].set_title('$y$ (Blurred + Noise)', fontsize=11, fontweight='bold')
    axes_list[1].axis('off')

    axes_list[2].imshow(x_r1_vis, cmap='gray', vmin=vmin, vmax=vmax)
    axes_list[2].set_title('R1: Gaussian MRF', fontsize=11, fontweight='bold')
    axes_list[2].axis('off')

    axes_list[3].imshow(x_r2_vis, cmap='gray', vmin=vmin, vmax=vmax)
    axes_list[3].set_title('R2: QGGMRF', fontsize=11, fontweight='bold')
    axes_list[3].axis('off')

    # Row 2: 128x128 crops
    axes_list[4].imshow(x_true_crop, cmap='gray', vmin=vmin, vmax=vmax)
    axes_list[4].set_title(f'Crop [{best_r}:{best_r+crop_size}, {best_c}:{best_c+crop_size}]', fontsize=10)
    axes_list[4].axis('off')

    axes_list[5].imshow(y_crop, cmap='gray', vmin=vmin, vmax=vmax)
    axes_list[5].set_title('Crop', fontsize=10)
    axes_list[5].axis('off')

    axes_list[6].imshow(x_r1_crop, cmap='gray', vmin=vmin, vmax=vmax)
    axes_list[6].set_title('Crop', fontsize=10)
    axes_list[6].axis('off')

    axes_list[7].imshow(x_r2_crop, cmap='gray', vmin=vmin, vmax=vmax)
    axes_list[7].set_title('Crop', fontsize=10)
    axes_list[7].axis('off')

    # Add shared colorbar
    cbar_ax = fig.add_subplot(gs[:, 4])
    cbar = plt.colorbar(im0, cax=cbar_ax)
    cbar.set_label('Intensity', fontsize=11)

    plt.savefig('task_d6_reconstructions.png', dpi=150, bbox_inches='tight')
    print("Figure saved as task_d6_reconstructions.png")
    plt.close()

    return crop_coords


if __name__ == "__main__":
    crop_coords = run_d6()
    print(f"Crop coordinates: row {crop_coords[0]} to {crop_coords[0]+128}, col {crop_coords[1]} to {crop_coords[1]+128}")
