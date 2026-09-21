import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from lab3 import load_pgm, load_kernel, forward, adjoint, adjoint_autograd


def main():
    def prior_cost(x, sigma_x):
        """Return the prior term (1/2) x^t B x of equation (7).

        x        (H, W) float64 tensor
        sigma_x  float
        returns  float64 scalar tensor
        """
        # Define kernel g
        g = torch.tensor([
            [1.0, 2.0, 1.0],
            [2.0, 0.0, 2.0],
            [1.0, 2.0, 1.0]
        ], dtype=torch.float64) / 12.0

        # Pad x and ones(x) with zeros (pad=1 on all sides for 3x3 kernel)
        x_padded = F.pad(x.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='constant', value=0)
        ones_x_padded = F.pad(torch.ones_like(x).unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='constant', value=0)

        # Apply conv2d with kernel g
        g_kernel = g.unsqueeze(0).unsqueeze(0)
        conv_g_ones = F.conv2d(ones_x_padded, g_kernel, padding=0)
        conv_g_x = F.conv2d(x_padded, g_kernel, padding=0)

        # Compute Bx = (conv2d(g, ones) - conv2d(g, x)) / sigma_x^2
        Bx = (conv_g_ones - conv_g_x) / (sigma_x ** 2)

        # Remove batch and channel dimensions
        Bx = Bx.squeeze(0).squeeze(0)

        # Return prior cost = (1/2) * <x, Bx>
        return 0.5 * (x * Bx).sum()


    def prior_grad(x, sigma_x):
        """Return B x, the gradient of prior_cost, from equation (7).

        x        (H, W) float64 tensor
        sigma_x  float
        returns  (H, W) float64 tensor
        """
        # Define kernel g
        g = torch.tensor([
            [1.0, 2.0, 1.0],
            [2.0, 0.0, 2.0],
            [1.0, 2.0, 1.0]
        ], dtype=torch.float64) / 12.0

        # Pad x and ones(x) with zeros (pad=1 on all sides for 3x3 kernel)
        x_padded = F.pad(x.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='constant', value=0)
        ones_x_padded = F.pad(torch.ones_like(x).unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='constant', value=0)

        # Apply conv2d with kernel g
        g_kernel = g.unsqueeze(0).unsqueeze(0)
        conv_g_ones = F.conv2d(ones_x_padded, g_kernel, padding=0)
        conv_g_x = F.conv2d(x_padded, g_kernel, padding=0)

        # Compute Bx = (conv2d(g, ones) - conv2d(g, x)) / sigma_x^2
        Bx = (conv_g_ones - conv_g_x) / (sigma_x ** 2)

        # Remove batch and channel dimensions
        return Bx.squeeze(0).squeeze(0)

    # Load image
    x = load_pgm("kodim23.pgm")

    # Define kernel g
    g = torch.tensor([
        [1.0, 2.0, 1.0],
        [2.0, 0.0, 2.0],
        [1.0, 2.0, 1.0]
    ], dtype=torch.float64) / 12.0

    # Compute conv_g_x
    x_padded = F.pad(x.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='constant', value=0)
    g_kernel = g.unsqueeze(0).unsqueeze(0)
    conv_g_x = F.conv2d(x_padded, g_kernel, padding=0).squeeze(0).squeeze(0)

    # Compute prediction error e = x - conv_g_x
    e = x - conv_g_x

    # Compute standard deviation over interior (excluding 1-pixel border)
    e_interior = e[1:-1, 1:-1]
    std_interior = e_interior.std()

    # Display image with adjustment: +0.5 and clip to [0, 1]
    e_display = torch.clamp(e + 0.5, 0, 1)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(e_display.numpy(), cmap='gray_r')
    ax.set_title(r'Noncausal Prediction Error $e = x - g \ast x$', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('d4_prediction_error.png', dpi=150, bbox_inches='tight')
    print("Figure saved as d4_prediction_error.png")

    # Print results
    print(f"\nNoncausal Prediction Error Statistics:")
    print(f"Standard deviation (interior): {std_interior.item():.6f}")
    print(f"Mean (interior): {e_interior.mean().item():.6f}")
    print(f"Min/Max (full): [{e.min().item():.6f}, {e.max().item():.6f}]")

    plt.show()


if __name__ == "__main__":
    main()
