import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from lab3 import load_pgm, load_kernel, forward, adjoint


def main():
    # Load image
    x = load_pgm("kodim23.pgm")

    # Load kernel
    a = load_kernel("levin09_kernels/levin09_kernel1.txt")

    # Compute forward and adjoint
    for_result = forward(x, a)
    adj_result = adjoint(x, a)

    # Compute difference
    diff = for_result - adj_result

    # Create 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Enable TeX rendering
    plt.rcParams['text.usetex'] = True

    # Panel a: Original image (inverted grayscale)
    im0 = axes[0, 0].imshow(x.numpy(), cmap='gray_r')
    axes[0, 0].set_title(r'$x$ (original image)', fontsize=12)
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Panel b: Forward result (inverted grayscale)
    im1 = axes[0, 1].imshow(for_result.numpy(), cmap='gray_r')
    axes[0, 1].set_title(r'$Ax$', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Panel c: Adjoint result (inverted grayscale)
    im2 = axes[1, 0].imshow(adj_result.numpy(), cmap='gray_r')
    axes[1, 0].set_title(r'$A^T x$', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Panel d: Difference with custom colormap (0 maps to gray)
    # Use TwoSlopeNorm to center colormap at 0
    diff_np = diff.numpy()
    vmin = diff_np.min()
    vmax = diff_np.max()
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im3 = axes[1, 1].imshow(diff_np, cmap='RdBu_r', norm=norm)
    axes[1, 1].set_title(r'$Ax - A^T x$', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('d1_visualization.png', dpi=150, bbox_inches='tight')
    print("Figure saved as d1_visualization.png")
    plt.show()


if __name__ == "__main__":
    main()
