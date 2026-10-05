import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'lab3'))

import torch
from lab3 import forward, adjoint, load_pgm, load_kernel

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
             ( 0, -1),          ( 0, 1),
             ( 1, -1), ( 1, 0), ( 1, 1)]

B_WEIGHTS = [1/12, 1/6, 1/12,
             1/6,        1/6,
             1/12, 1/6, 1/12]


def rho(delta, sigma_x, p, q, T):
    """Return the QGGMRF potential of equation (3), elementwise.

    Computes ρ(Δ) for all elements of the input tensor, element-wise.

    delta    float64 tensor of any shape
    sigma_x  float, prior standard deviation
    p, q, T  float, QGGMRF parameters
    returns  float64 tensor of the same shape as delta
    """
    abs_delta = torch.abs(delta)
    r = torch.abs(delta) / (T * sigma_x)
    r_pow = torch.pow(r, q - p)
    return (torch.pow(abs_delta, p) / (p * torch.pow(sigma_x, p))) * (r_pow / (1 + r_pow))


def rho_prime(delta, sigma_x, p, q, T):
    """Return the QGGMRF influence function of equation (4), elementwise.

    Computes ρ'(Δ) for all elements of the input tensor, element-wise.

    delta    float64 tensor of any shape
    sigma_x  float, prior standard deviation
    p, q, T  float, QGGMRF parameters
    returns  float64 tensor of the same shape as delta
    """
    abs_delta = torch.abs(delta)
    r = abs_delta / (T * sigma_x)
    r_pow = torch.pow(r, q - p)
    numerator = r_pow * (q / p + r_pow)
    denominator = torch.pow(1 + r_pow, 2)
    return (torch.pow(abs_delta, p - 1) / torch.pow(sigma_x, p)) * (numerator / denominator) * torch.sign(delta)

def neighbor_diffs(x):
    """Return the differences x[s] - x[r] for the 8 neighbor offsets.

    Plane i holds x minus x shifted by NEIGHBORS[i].  Entries whose
    neighbor falls outside the image are set to zero, and the matching
    entries of the mask are zero, so that a pair counts only when both
    pixels are inside.

    x        (H, W) float64 tensor
    returns  (diffs, valid), each (8, H, W), float64 and bool
    """
    H, W = x.shape
    diffs = torch.zeros((8, H, W), dtype=torch.float64)
    valid = torch.ones((8, H, W), dtype=torch.bool)

    for i, (dr, dc) in enumerate(NEIGHBORS):
        # Shift x by the neighbor offset
        x_shifted = torch.roll(x, shifts=(dr, dc), dims=(0, 1))

        # Compute the difference
        diffs[i] = x - x_shifted

        # Mask out the boundary regions where the neighbor is outside
        # If dr > 0, the top dr rows are invalid (wrapped from bottom)
        if dr > 0:
            valid[i, :dr, :] = False
        # If dr < 0, the bottom -dr rows are invalid (wrapped from top)
        elif dr < 0:
            valid[i, dr:, :] = False

        # If dc > 0, the left dc columns are invalid (wrapped from right)
        if dc > 0:
            valid[i, :, :dc] = False
        # If dc < 0, the right -dc columns are invalid (wrapped from left)
        elif dc < 0:
            valid[i, :, dc:] = False

        # Zero out invalid entries in diffs
        diffs[i][~valid[i]] = 0

    return diffs, valid


def surrogate_weights(x, sigma_x, p, q, T):
    """Return the b̃_{s,r} coefficients of equations (14) and (15).

    Computes the reweighted neighbor coefficients used to form the surrogate
    Hessian from equation (14), with tiny differences floored so the Delta' -> 0
    limit of equation (15) is reached. Applies the neighbor validity mask so
    boundary pairs contribute zero.

    x        (H, W) float64 tensor, current image
    sigma_x  float, prior standard deviation
    p, q, T  float, QGGMRF parameters
    returns  (8, H, W) float64 tensor, zero where the pair is invalid
    """
    # Get neighbor differences and validity mask
    diffs, valid = neighbor_diffs(x)

    # Convert B_WEIGHTS to tensor and reshape for broadcasting
    b_weights = torch.tensor(B_WEIGHTS, dtype=torch.float64).reshape(8, 1, 1)

    # Floor small differences to avoid numerical issues
    floor_val = torch.tensor(1e-8, dtype=torch.float64)
    diffs_floored = torch.where(torch.abs(diffs) < floor_val, floor_val, diffs)

    # Equation (14) for every pair. The floor at 1e-8 stands in for the
    # Delta' -> 0 limit of equation (15), since the QGGMRF factor tends to 1 there.
    abs_diffs = torch.abs(diffs_floored)
    r = abs_diffs / (T * sigma_x)
    r_pow = torch.pow(r, q - p)

    numerator = r_pow * (q / p + r_pow)
    denominator = torch.pow(1 + r_pow, 2)

    b_tilde = b_weights * (torch.pow(abs_diffs, p - 2) / (2 * torch.pow(sigma_x, p))) * (numerator / denominator)

    # Zero out invalid entries
    b_tilde = torch.where(valid, b_tilde, torch.tensor(0.0, dtype=torch.float64))

    return b_tilde


def true_cost(x, y, a, sigma_w, sigma_x, p, q, T):
    """Return the true cost f(x) of equation (1).

    Equation (1): f(x) = (1/2σ_w²)||y - Ax||² + (1/2)∑_s∑_{r∈∂s} b_{s,r}ρ(x_s - x_r)

    Differentiable in x, so torch.autograd can take its gradient. Floor
    |x_s - x_r| at a small value (1e-8) before the QGGMRF power to avoid
    NaN/overflow from |Δ/(Tσ_x)|^(q-p). Apply the neighbor validity mask
    after rho so floored border pairs contribute exactly zero.

    x        (H, W) float64 tensor
    y        (H, W) float64 tensor, observed blurred image
    a        (Ka, Kb) float64 tensor, blur kernel
    sigma_w  float, noise standard deviation
    sigma_x  float, prior standard deviation
    p, q, T  float, QGGMRF parameters
    returns  float64 scalar tensor
    """
    # Data fidelity term: (1/2σ_w²)||y - Ax||²
    Ax = forward(x, a)
    residual = y - Ax
    data_cost = 0.5 * (residual ** 2).sum() / (sigma_w ** 2)

    # Prior term: (1/2)∑∑ b_{s,r} ρ(x_s - x_r)
    diffs, valid = neighbor_diffs(x)

    # Floor small differences to avoid numerical issues with the diverging exponent
    floor_val = torch.tensor(1e-8, dtype=torch.float64)
    diffs_floored = torch.where(torch.abs(diffs) < floor_val, floor_val, diffs)

    # Compute ρ(Δ) for each pair
    rho_vals = rho(diffs_floored, sigma_x, p, q, T)

    # Apply validity mask and neighbor weights
    b_weights = torch.tensor(B_WEIGHTS, dtype=torch.float64).reshape(8, 1, 1)
    rho_masked = torch.where(valid, rho_vals, torch.tensor(0.0, dtype=torch.float64))

    # Compute prior cost
    prior_cost = 0.5 * (b_weights * rho_masked).sum()

    return data_cost + prior_cost


def true_gradient(x, y, a, sigma_w, sigma_x, p, q, T):
    """Return the gradient of the true cost, equation (16).

    Equation (16): [∇f(x)]_s = -(1/σ_w²)[A^t(y - Ax)]_s + ∑_{r∈∂s} b_{s,r} ρ'(x_s - x_r)

    Computed via automatic differentiation using torch.autograd.grad(true_cost(...), x).
    x must have requires_grad enabled. The forward and adjoint of a must be torch
    operations so the gradient graph reaches x. Equation (16) is the analytic form
    to verify this implementation against.

    x        (H, W) float64 tensor with requires_grad=True
    y        (H, W) float64 tensor, observed blurred image
    a        (Ka, Kb) float64 tensor, blur kernel
    sigma_w  float, noise standard deviation
    sigma_x  float, prior standard deviation
    p, q, T  float, QGGMRF parameters
    returns  (H, W) float64 tensor
    """
    # Ensure x has requires_grad enabled
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)

    # Compute the true cost
    cost = true_cost(x, y, a, sigma_w, sigma_x, p, q, T)

    # Compute gradient via autograd
    grad = torch.autograd.grad(cost, x, create_graph=False)[0]

    return grad


def optimal_step_size(x, g, a, sigma_w, sigma_x, p, q, T):
    """Return the exact surrogate step size α* of equations (17)-(18).

    Equations (17)-(18): α* = (g^t g) / (g^t H g), where g = ∇f(x)

    The denominator g^t H g is the quadratic form of the surrogate Hessian:
    g^t H g = (1/σ_w²)||Ag||² + ∑_s∑_{r∈∂s} b̃_{s,r} (g_s - g_r)²

    Never explicitly forms H. Instead, computes the quadratic form using:
    - One forward projection A g
    - Surrogate weights b̃_{s,r} computed at current x
    - Neighbor differences of the gradient g

    x        (H, W) float64 tensor, current image
    g        (H, W) float64 tensor, gradient from true_gradient
    a        (Ka, Kb) float64 tensor, blur kernel
    sigma_w  float, noise standard deviation
    sigma_x  float, prior standard deviation
    p, q, T  float, QGGMRF parameters
    returns  float64 scalar tensor
    """
    # Numerator: g^t g
    g_norm_sq = (g ** 2).sum()

    # Denominator: g^t H g
    # Data term: (1/σ_w²)||Ag||²
    Ag = forward(g, a)
    data_quadform = (Ag ** 2).sum() / (sigma_w ** 2)

    # Prior term: ∑∑ b̃_{s,r} (g_s - g_r)²
    b_tilde = surrogate_weights(x, sigma_x, p, q, T)
    g_diffs, valid = neighbor_diffs(g)

    # Compute (g_s - g_r)² for each neighbor pair
    g_diffs_sq = g_diffs ** 2

    # Sum weighted squared differences
    prior_quadform = (b_tilde * g_diffs_sq).sum()

    # Total quadratic form
    g_Hg = data_quadform + prior_quadform

    # Compute step size
    alpha_star = g_norm_sq / g_Hg

    return alpha_star


def reconstruct(y, a, sigma_w, sigma_x, p, q, T, num_iters, omega):
    """Steepest descent with the surrogate step size, starting at y.

    Given: y, A, sigma_w, sigma_x, p, q, T, N iterations, omega in (0, 2)
    Initialize x = y
    Record the TRUE cost f(x) from (1) and (3)
    For k = 0 to N-1:
        g          = true_gradient(x)          # torch autograd of f
        alpha_star = optimal_step_size(x, g)   # exact step for the surrogate at x
        x          = x - omega * alpha_star * g
        Record the TRUE cost f(x) from (1) and (3)

    returns  (x, cost_history), the final image and a list of the true
             cost after each iteration, of length num_iters + 1 whose
             first entry is f(y)
    """
    # Initialize x to the observed (blurred) image
    x = y.clone().detach().requires_grad_(True)

    # Initialize cost history with the initial cost
    cost_history = [true_cost(x.detach(), y, a, sigma_w, sigma_x, p, q, T).item()]

    # Steepest descent iterations
    for k in range(num_iters):
        # Compute gradient of the true cost
        g = true_gradient(x, y, a, sigma_w, sigma_x, p, q, T)

        # Compute the exact surrogate step size
        alpha = optimal_step_size(x.detach(), g, a, sigma_w, sigma_x, p, q, T)

        # Update x with relaxation parameter omega
        with torch.no_grad():
            x.data = x.data - omega * alpha * g

        # Record the true cost at this iteration
        cost_history.append(true_cost(x.detach(), y, a, sigma_w, sigma_x, p, q, T).item())

    return x.detach(), cost_history



    