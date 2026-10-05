import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'lab3'))

from lab3 import load_pgm, load_kernel, forward, set_seed
from lab4 import reconstruct, surrogate_weights


def run_d9():
    """Generate D9 deliverable: image of sum_r b~_{s,r} at the final iteration of R2."""
    set_seed(42)

    x_true = load_pgm('./kodim23.pgm')
    a = load_kernel('./levin09_kernels/levin09_kernel1.txt')

    y = torch.clamp(forward(x_true, a) + torch.randn_like(x_true) * 0.02, 0, 1)
    sigma_w = torch.tensor(0.02, dtype=torch.float64)

    p = torch.tensor(2.0, dtype=torch.float64)
    q = torch.tensor(1.2, dtype=torch.float64)
    T = torch.tensor(1.0, dtype=torch.float64)
    sigma_x = torch.tensor(0.020, dtype=torch.float64)

    print("Running R2 (QGGMRF) reconstruction...")
    x_recon, _ = reconstruct(y, a, sigma_w, sigma_x, p, q, T, num_iters=50,
                             omega=torch.tensor(1.0, dtype=torch.float64))

    # Sum the 8 neighbor planes of b~_{s,r} at the final iterate
    weight_sum = surrogate_weights(x_recon, sigma_x, p, q, T).sum(dim=0).numpy()

    # Display range from the 1st-99th percentiles of the interior, so the
    # image border (fewer valid neighbors) does not stretch the scale
    interior = weight_sum[1:-1, 1:-1]
    vmin, vmax = np.percentile(interior, [1, 99])

    print(f"Full range:    [{weight_sum.min():.2f}, {weight_sum.max():.2f}]")
    print(f"Interior range:[{interior.min():.2f}, {interior.max():.2f}], std {interior.std():.2f}")
    print(f"Display range: [{vmin:.2f}, {vmax:.2f}]")

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(weight_sum, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(r'$\sum_{r \in \partial s} \tilde{b}_{s,r}$ at final iteration of R2'
                 f'\nDisplay range: [{vmin:.1f}, {vmax:.1f}] (1st-99th percentile of interior)',
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r'$\sum_r \tilde{b}_{s,r}$')

    plt.savefig('task_d9_weight_sum.png', dpi=150, bbox_inches='tight')
    print("Figure saved as task_d9_weight_sum.png")
    plt.close()


if __name__ == "__main__":
    run_d9()
