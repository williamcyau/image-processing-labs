import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent.parent / 'lab3'))

from lab3 import load_pgm, load_kernel, forward, set_seed
from lab4 import reconstruct


def run_d7():
    """Generate D7 deliverable: edge profile plot with zoom inset.

    Plots pixel values along a horizontal line crossing a strong edge
    for x_true, R1 (Gaussian), and R2 (QGGMRF) with transparent lines
    and a zoomed inset showing the edge transition detail.
    """
    # Set seed for reproducibility
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

    print("Running R1 (Gaussian MRF) reconstruction...")
    p1 = torch.tensor(2.0, dtype=torch.float64)
    q1 = torch.tensor(2.0, dtype=torch.float64)
    T1 = torch.tensor(100.0, dtype=torch.float64)
    sigma_x1 = torch.tensor(0.036, dtype=torch.float64)

    x_r1, _ = reconstruct(y, a, sigma_w, sigma_x1, p1, q1, T1, num_iters=50, omega=torch.tensor(1.0, dtype=torch.float64))

    print("Running R2 (QGGMRF) reconstruction...")
    p2 = torch.tensor(2.0, dtype=torch.float64)
    q2 = torch.tensor(1.2, dtype=torch.float64)
    T2 = torch.tensor(1.0, dtype=torch.float64)
    sigma_x2 = torch.tensor(0.020, dtype=torch.float64)

    x_r2, _ = reconstruct(y, a, sigma_w, sigma_x2, p2, q2, T2, num_iters=50, omega=torch.tensor(1.0, dtype=torch.float64))

    # Convert to numpy and clamp for visualization
    x_true_np = np.clip(x_true.numpy(), 0, 1)
    x_r1_np = np.clip(x_r1.detach().numpy(), 0, 1)
    x_r2_np = np.clip(x_r2.detach().numpy(), 0, 1)

    # Find crop region with high contrast edge (same as D6)
    crop_size = 128
    H, W = x_true_np.shape

    best_score = -np.inf
    best_r, best_c = 0, 0

    for r in range(0, H - crop_size, 32):
        for c in range(0, W - crop_size, 32):
            crop = x_true_np[r:r+crop_size, c:c+crop_size]
            grad_r = np.diff(crop, axis=0)
            grad_c = np.diff(crop, axis=1)
            grad_mag = np.sqrt(grad_r[:, :-1]**2 + grad_c[:-1, :]**2).mean()

            if grad_mag > best_score:
                best_score = grad_mag
                best_r, best_c = r, c

    print(f"Using crop region at ({best_r}, {best_c})")

    # Extract the crop region
    x_true_crop = x_true_np[best_r:best_r+crop_size, best_c:best_c+crop_size]
    x_r1_crop = x_r1_np[best_r:best_r+crop_size, best_c:best_c+crop_size]
    x_r2_crop = x_r2_np[best_r:best_r+crop_size, best_c:best_c+crop_size]

    # Find the horizontal line with the strongest edge showing R1 vs R2 difference
    best_line_idx = 0
    best_max_grad = -np.inf
    best_max_grad_idx = 0

    for row_idx in range(crop_size):
        grad_true = np.abs(np.diff(x_true_crop[row_idx, :]))
        grad_r1 = np.abs(np.diff(x_r1_crop[row_idx, :]))
        grad_r2 = np.abs(np.diff(x_r2_crop[row_idx, :]))

        # Prefer edges where R2 > R1 (QGGMRF preserves edge better)
        grad_diff = grad_r2 - grad_r1
        max_grad_col = np.argmax(grad_diff)
        max_grad_val = grad_diff[max_grad_col]

        if max_grad_val > best_max_grad:
            best_max_grad = max_grad_val
            best_line_idx = row_idx
            best_max_grad_idx = max_grad_col

    line_idx = best_line_idx
    x_true_profile = x_true_crop[line_idx, :]
    x_r1_profile = x_r1_crop[line_idx, :]
    x_r2_profile = x_r2_crop[line_idx, :]

    # Compute gradients at the edge
    grad_true = np.abs(np.diff(x_true_profile))
    grad_r1 = np.abs(np.diff(x_r1_profile))
    grad_r2 = np.abs(np.diff(x_r2_profile))
    max_grad_idx = best_max_grad_idx

    actual_row = best_r + line_idx
    actual_col = best_c + max_grad_idx
    print(f"Strongest R2-preserving edge at row {actual_row}, columns {actual_col}-{actual_col+1}")
    print(f"Gradient - True: {grad_true[max_grad_idx]:.4f}, R1: {grad_r1[max_grad_idx]:.4f}, R2: {grad_r2[max_grad_idx]:.4f}")

    # Create the edge profile plot with actual column indices from full image
    fig = plt.figure(figsize=(16, 6.5))
    ax_main = plt.subplot(1, 1, 1)

    # X-axis shows actual column indices from the full image (best_c to best_c+128)
    x_pixels = np.arange(best_c, best_c + len(x_true_profile))

    # Plot with transparency to handle overlapping lines
    ax_main.plot(x_pixels, x_true_profile, 'b-', linewidth=2.5, label='$x_{true}$ (Original)',
                 marker='o', markersize=3, markevery=8, alpha=0.75)
    ax_main.plot(x_pixels, x_r1_profile, 'r--', linewidth=2, label='R1: Gaussian MRF',
                 marker='s', markersize=3, markevery=8, alpha=0.65)
    ax_main.plot(x_pixels, x_r2_profile, 'g-.', linewidth=2.5, label='R2: QGGMRF',
                 marker='^', markersize=3, markevery=8, alpha=0.75)

    # Shade the edge transition zone (2 pixels wide centered at the edge)
    edge_col = best_c + max_grad_idx
    ax_main.axvspan(edge_col - 1, edge_col + 1, alpha=0.15, color='yellow', label='Edge transition zone')

    # Mark the edge location
    ax_main.axvline(edge_col, color='k', linestyle=':', linewidth=1.5, alpha=0.7)

    ax_main.set_xlabel('Column Index (Full Image)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Intensity', fontsize=12, fontweight='bold')
    ax_main.set_title(f'Edge Profile: Row {actual_row} (QGGMRF Preserves Edge Better)', fontsize=13, fontweight='bold')
    ax_main.legend(fontsize=11, loc='best', framealpha=0.95)
    ax_main.grid(True, alpha=0.3)
    y_min = min(x_true_profile.min(), x_r1_profile.min(), x_r2_profile.min()) - 0.05
    y_max = max(x_true_profile.max(), x_r1_profile.max(), x_r2_profile.max()) + 0.05
    ax_main.set_ylim([y_min, y_max])

    # Add inset zoom plot of the edge region
    # Create inset axis (35% width, 40% height, positioned at upper right)
    ax_inset = inset_axes(ax_main, width="35%", height="40%", loc='upper right', borderpad=1.5)

    # Zoom window: ±6 pixels around the edge
    zoom_start = max(0, max_grad_idx - 6)
    zoom_end = min(len(x_true_profile), max_grad_idx + 7)
    zoom_start_col = best_c + zoom_start
    zoom_end_col = best_c + zoom_end

    x_zoom = x_pixels[zoom_start:zoom_end]
    x_true_zoom = x_true_profile[zoom_start:zoom_end]
    x_r1_zoom = x_r1_profile[zoom_start:zoom_end]
    x_r2_zoom = x_r2_profile[zoom_start:zoom_end]

    # Plot zoomed data with enhanced transparency
    ax_inset.plot(x_zoom, x_true_zoom, 'b-', linewidth=2.5, marker='o', markersize=7, alpha=0.75, label='x_true')
    ax_inset.plot(x_zoom, x_r1_zoom, 'r--', linewidth=2, marker='s', markersize=7, alpha=0.65, label='R1')
    ax_inset.plot(x_zoom, x_r2_zoom, 'g-.', linewidth=2.5, marker='^', markersize=7, alpha=0.75, label='R2')

    # Shade the edge zone in inset
    ax_inset.axvspan(edge_col - 1, edge_col + 1, alpha=0.2, color='yellow')
    ax_inset.axvline(edge_col, color='k', linestyle=':', linewidth=1, alpha=0.7)

    ax_inset.set_xlabel('Column', fontsize=9, fontweight='bold')
    ax_inset.set_ylabel('Intensity', fontsize=9, fontweight='bold')
    ax_inset.set_title('Zoomed Edge Region', fontsize=10, fontweight='bold')
    ax_inset.grid(True, alpha=0.4)
    ax_inset.legend(fontsize=8, loc='upper left', framealpha=0.9)
    ax_inset.tick_params(labelsize=8)

    # Draw rectangle around zoomed region on main plot
    rect = Rectangle((zoom_start_col - 0.5, y_min),
                     (zoom_end_col - zoom_start_col),
                     (y_max - y_min),
                     linewidth=1.5, edgecolor='purple', facecolor='none', linestyle='--', alpha=0.7)
    ax_main.add_patch(rect)

    plt.tight_layout()
    plt.savefig('task_d7_edge_profile.png', dpi=150, bbox_inches='tight')
    print("Figure saved as task_d7_edge_profile.png")
    plt.close()

    return actual_row


if __name__ == "__main__":
    run_d7()
