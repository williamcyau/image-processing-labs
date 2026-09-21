import numpy as np
import torch
import torch.nn.functional as F


# Identity kernel for identity operator (1x1 convolution acts as identity)
a_identity = torch.ones((1, 1), dtype=torch.float64)


def set_seed(seed=42):
    """Set random seed for reproducibility.

    seed     int, random seed (default 42)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_pgm(path):
    """Read a binary PGM image file.

    path     path to a P5 PGM file
    returns  (H, W) float64 tensor with values in [0, 1]
    """
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P5"
        width, height = map(int, f.readline().split())
        assert int(f.readline()) == 255
        raw = np.frombuffer(f.read(), dtype=np.uint8).reshape(height, width)
    return torch.tensor(raw / 255.0, dtype=torch.float64)


def load_kernel(path):
    """Read a blur kernel from a text file.

    path     path to a kernel file, one row of the kernel per line
    returns  (Ka, Kb) float64 tensor, nonnegative and summing to 1
    """
    return torch.tensor(np.loadtxt(path), dtype=torch.float64)


def normalized_rmse(x_n, x_prev):
    """Compute normalized RMSE between consecutive iterations.

    Measures relative change: NRMSE = ||x(n) - x(n-1)|| / ||x(n-1)||

    x_n      (H, W) float64 tensor, current iterate
    x_prev   (H, W) float64 tensor, previous iterate
    returns  float64 scalar tensor
    """
    diff = x_n - x_prev
    norm_diff = torch.sqrt((diff ** 2).sum())
    norm_prev = torch.sqrt((x_prev ** 2).sum())
    return norm_diff / norm_prev


def forward(x, a):
    """Apply the blur operator A of equation (2).

    x        (H, W) float64 tensor, the image
    a        (Ka, Kb) float64 tensor, the blur kernel, odd sized in both axes
    returns  (H, W) float64 tensor, A x
    """
    H, W = x.shape
    Ka, Kb = a.shape

    # Flatten x and a to vectors
    x_flat = x.flatten()
    a_flat = a.flatten()

    # Flip the vector a (first element becomes last, etc.)
    a_flipped = torch.flip(a_flat, dims=[0])

    # Reshape flipped kernel back to 2D for conv2d
    a_flipped_2d = a_flipped.reshape(Ka, Kb)

    # Add batch and channel dimensions for conv2d
    x_reshaped = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    a_kernel = a_flipped_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, Ka, Kb)

    # Circular padding to maintain output size and implement circular boundaries
    pad_h = Ka // 2
    pad_w = Kb // 2
    x_padded = F.pad(x_reshaped, (pad_w, pad_w, pad_h, pad_h), mode='circular')

    # Apply conv2d with no additional padding (already padded circularly)
    output = F.conv2d(x_padded, a_kernel, padding=0)

    # Remove batch and channel dimensions to restore original shape
    return output.squeeze(0).squeeze(0)

def adjoint(v, a):
    """Apply the transpose operator A^t of equation (4).

    v        (H, W) float64 tensor
    a        (Ka, Kb) float64 tensor, the same kernel passed to forward
    returns  (H, W) float64 tensor, A^t v
    """
    H, W = v.shape
    Ka, Kb = a.shape

    # Flatten v and a to vectors
    v_flat = v.flatten()
    a_flat = a.flatten()

    # Use a without flipping (unlike forward)
    a_no_flip_2d = a_flat.reshape(Ka, Kb)

    # Add batch and channel dimensions for conv2d
    v_reshaped = v.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    a_kernel = a_no_flip_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, Ka, Kb)

    # Circular padding to maintain output size and implement circular boundaries
    pad_h = Ka // 2
    pad_w = Kb // 2
    v_padded = F.pad(v_reshaped, (pad_w, pad_w, pad_h, pad_h), mode='circular')

    # Apply conv2d with no additional padding (already padded circularly)
    output = F.conv2d(v_padded, a_kernel, padding=0)

    # Remove batch and channel dimensions to restore original shape
    return output.squeeze(0).squeeze(0)

def adjoint_autograd(v, a):
    """Apply A^t, obtained from forward() by automatic differentiation.

    v        (H, W) float64 tensor
    a        (Ka, Kb) float64 tensor, the same kernel passed to forward
    returns  (H, W) float64 tensor, A^t v
    """
    x = torch.zeros_like(v, requires_grad=True)
    return torch.autograd.grad((forward(x, a) * v).sum(), x)[0]


def _apply_laplacian(x, sigma_x):
    """Apply the scaled weighted graph Laplacian: B*x = (1/σ_x²)(D-W)*x

    Helper function for computing the Laplacian operator on x.

    x        (H, W) float64 tensor
    sigma_x  float
    returns  (H, W) float64 tensor
    """
    # Define kernel g (weights for pairwise differences)
    g = torch.tensor([
        [1.0, 2.0, 1.0],
        [2.0, 0.0, 2.0],
        [1.0, 2.0, 1.0]
    ], dtype=torch.float64) / 12.0

    # Pad x with zeros (pad=1 on all sides for 3x3 kernel)
    x_padded = F.pad(x.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='constant', value=0)
    ones_padded = F.pad(torch.ones_like(x).unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='constant', value=0)

    # Apply conv2d with kernel g
    g_kernel = g.unsqueeze(0).unsqueeze(0)
    conv_g_x = F.conv2d(x_padded, g_kernel, padding=0).squeeze(0).squeeze(0)
    conv_g_ones = F.conv2d(ones_padded, g_kernel, padding=0).squeeze(0).squeeze(0)

    # Compute scaled Laplacian: B*x = (1/σ_x²) * [(D-W)*x]
    return (x * conv_g_ones - conv_g_x) / (sigma_x ** 2)


def prior_cost(x, sigma_x):
    """Return the prior term (1/2) <x, B*x> of equation (5).

    Pairwise GMRF prior with scaled Laplacian B = (1/σ_x²)(D-W):
    c_prior(x) = (1/2) * <x, B*x>

    x        (H, W) float64 tensor
    sigma_x  float
    returns  float64 scalar tensor
    """
    Bx = _apply_laplacian(x, sigma_x)
    return 0.5 * (x * Bx).sum()


def prior_grad(x, sigma_x):
    """Return B*x, the gradient of prior_cost, from equation (5).

    Gradient of pairwise GMRF prior with scaled Laplacian:
    ∇c_prior(x) = B*x = (1/σ_x²) * (D-W)*x

    x        (H, W) float64 tensor
    sigma_x  float
    returns  (H, W) float64 tensor
    """
    return _apply_laplacian(x, sigma_x)


def cost(x, y, a, sigma_w, sigma_x):
    """Return the MAP cost c(x) of equation (8).

    x, y     (H, W) float64 tensors
    a        (Ka, Kb) float64 tensor, the blur kernel
    sigma_w  float, noise standard deviation
    sigma_x  float, prior standard deviation
    returns  float64 scalar tensor
    """
    # Compute data fidelity term: 0.5 * (1/sigma_w^2) * ||y - A*x||^2
    Ax = forward(x, a)
    residual = y - Ax
    data_cost = 0.5 * (residual ** 2).sum() / (sigma_w ** 2)

    # Return total MAP cost
    return data_cost + prior_cost(x, sigma_x)


def cost_grad(x, y, a, sigma_w, sigma_x):
    """Return the gradient of the MAP cost, equation (9).

    x, y     (H, W) float64 tensors
    a        (Ka, Kb) float64 tensor, the blur kernel
    sigma_w  float, noise standard deviation
    sigma_x  float, prior standard deviation
    returns  (H, W) float64 tensor
    """
    # Compute gradient of data fidelity term: (1/sigma_w^2) * A^T * (A*x - y)
    Ax = forward(x, a)
    residual = Ax - y
    data_grad = adjoint(residual, a) / (sigma_w ** 2)

    # Return total gradient
    return data_grad + prior_grad(x, sigma_x)


def gradient_descent(y, a, sigma_w, sigma_x, num_iters, alpha=None):
    """Minimize the MAP cost by gradient descent, equation (10).

    Starts at x = y.  Uses the step size of equation (11) when alpha
    is None.

    y        (H, W) float64 tensor, observed blurred image
    a        (Ka, Kb) float64 tensor, the blur kernel
    sigma_w  float, noise standard deviation
    sigma_x  float, prior standard deviation
    num_iters int, number of gradient descent iterations
    alpha    float or None, step size (computed if None)
    returns  (x_hat, cost_history), an (H, W) tensor and a list of
             length num_iters + 1 whose first entry is c(y)
    """
    # Compute step size if not provided
    if alpha is None:
        alpha = 1.0 / (sigma_w**(-2) + 2*sigma_x**(-2))

    # Initialize x to y (observed blurred image)
    x = y.clone().detach()

    # Initialize cost history
    cost_history = [cost(x, y, a, sigma_w, sigma_x).item()]

    # Gradient descent iterations
    for i in range(num_iters):
        # Compute gradient
        grad = cost_grad(x, y, a, sigma_w, sigma_x)

        # Update x
        x = x - alpha * grad

        # Record cost
        cost_history.append(cost(x, y, a, sigma_w, sigma_x).item())

    return x, cost_history