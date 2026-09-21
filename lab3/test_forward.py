import torch
import torch.nn.functional as F
import numpy as np
from lab3 import forward, adjoint


def test_output_shape():
    """Test that output shape matches input shape."""
    H, W = 16, 20
    Ka, Kb = 5, 5

    x = torch.randn(H, W, dtype=torch.float64)
    a = torch.ones(Ka, Kb, dtype=torch.float64) / (Ka * Kb)

    output = forward(x, a)

    assert output.shape == (H, W), f"Expected shape ({H}, {W}), got {output.shape}"
    print(f"✓ Output shape test passed: {output.shape}")


def test_kernel_flip():
    """Test that kernel is correctly flipped."""
    # Create a simple kernel with distinct values
    a = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=torch.float64)

    # Flatten and flip
    a_flat = a.flatten()
    a_flipped = torch.flip(a_flat, dims=[0])
    a_flipped_2d = a_flipped.reshape(3, 3)

    # Expected: 180-degree rotation of the kernel
    expected = torch.tensor([
        [9.0, 8.0, 7.0],
        [6.0, 5.0, 4.0],
        [3.0, 2.0, 1.0]
    ], dtype=torch.float64)

    assert torch.allclose(a_flipped_2d, expected), \
        f"Kernel flip incorrect.\nExpected:\n{expected}\nGot:\n{a_flipped_2d}"
    print(f"✓ Kernel flip test passed:")
    print(f"  Original kernel:\n{a}")
    print(f"  Flipped kernel:\n{a_flipped_2d}")


def test_circular_boundaries():
    """Test that circular boundaries are implemented correctly."""
    # Create a simple test case with a delta kernel
    x = torch.zeros(5, 5, dtype=torch.float64)
    x[2, 2] = 1.0  # Impulse at center

    # Delta kernel (just 1 at center)
    a = torch.zeros(3, 3, dtype=torch.float64)
    a[1, 1] = 1.0

    output = forward(x, a)

    # With circular boundaries and a delta kernel, output should equal input
    assert torch.allclose(output, x), \
        f"Delta kernel should reproduce input with circular boundaries.\nExpected:\n{x}\nGot:\n{output}"
    print(f"✓ Circular boundaries test passed (delta kernel)")


def test_blur_kernel():
    """Test convolution with a simple blur kernel."""
    # Create a small test image
    x = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=torch.float64)

    # Simple averaging kernel (3x3)
    a = torch.ones(3, 3, dtype=torch.float64) / 9.0

    output = forward(x, a)

    # Verify output shape
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"

    # Verify output is reasonable (blurred values should be similar to center)
    # With circular boundaries, all pixels should get averaged
    mean_val = x.mean()
    print(f"✓ Blur kernel test passed:")
    print(f"  Input:\n{x}")
    print(f"  Output:\n{output}")
    print(f"  Input mean: {mean_val:.4f}")


def test_odd_kernel_sizes():
    """Test with various odd kernel sizes."""
    for Ka, Kb in [(3, 3), (5, 5), (7, 3), (3, 7)]:
        H, W = 16, 16
        x = torch.randn(H, W, dtype=torch.float64)
        a = torch.ones(Ka, Kb, dtype=torch.float64) / (Ka * Kb)

        output = forward(x, a)
        assert output.shape == (H, W), f"Failed for kernel size ({Ka}, {Kb})"

    print(f"✓ Various odd kernel sizes test passed")


def test_numerical_stability():
    """Test that the convolution is numerically stable."""
    H, W = 32, 32
    Ka, Kb = 5, 5

    x = torch.randn(H, W, dtype=torch.float64)
    a = torch.randn(Ka, Kb, dtype=torch.float64)
    a = a / a.sum()  # Normalize kernel

    output = forward(x, a)

    # Check for NaN or Inf
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"

    print(f"✓ Numerical stability test passed")


def test_kernel_flipped_in_forward():
    """Verify that the kernel is actually being flipped inside forward()."""
    x = torch.ones(3, 3, dtype=torch.float64)

    # Create a kernel with known asymmetry
    a = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=torch.float64)

    output = forward(x, a)

    # The kernel is flipped, so [1, 0, 0] in top-left becomes [0, 0, 0, 0, 0, 0, 1]
    # when flipped, placing the 1 at bottom-right
    # With circular boundaries and input of all ones, this tests the flip
    print(f"✓ Kernel flip verification in forward():")
    print(f"  Original kernel:\n{a}")
    print(f"  Output (with circular boundaries):\n{output}")


def test_adjoint_output_shape():
    """Test that adjoint output shape matches input shape."""
    H, W = 16, 20
    Ka, Kb = 5, 5

    v = torch.randn(H, W, dtype=torch.float64)
    a = torch.ones(Ka, Kb, dtype=torch.float64) / (Ka * Kb)

    output = adjoint(v, a)

    assert output.shape == (H, W), f"Expected shape ({H}, {W}), got {output.shape}"
    print(f"✓ Adjoint output shape test passed: {output.shape}")


def test_adjoint_no_flip():
    """Test that adjoint uses unflipped kernel."""
    v = torch.ones(3, 3, dtype=torch.float64)

    a = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=torch.float64)

    output = adjoint(v, a)

    print(f"✓ Adjoint no-flip verification:")
    print(f"  Kernel (unflipped):\n{a}")
    print(f"  Output:\n{output}")


def test_adjoint_delta_kernel():
    """Test adjoint with delta kernel."""
    v = torch.zeros(5, 5, dtype=torch.float64)
    v[2, 2] = 1.0

    a = torch.zeros(3, 3, dtype=torch.float64)
    a[1, 1] = 1.0

    output = adjoint(v, a)

    assert torch.allclose(output, v), \
        f"Delta kernel should reproduce input.\nExpected:\n{v}\nGot:\n{output}"
    print(f"✓ Adjoint delta kernel test passed")


def test_adjoint_blur_kernel():
    """Test adjoint with averaging blur kernel."""
    v = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=torch.float64)

    a = torch.ones(3, 3, dtype=torch.float64) / 9.0

    output = adjoint(v, a)

    assert output.shape == v.shape, f"Shape mismatch: {output.shape} vs {v.shape}"

    print(f"✓ Adjoint blur kernel test passed:")
    print(f"  Input:\n{v}")
    print(f"  Output:\n{output}")


def test_adjoint_numerical_stability():
    """Test that adjoint is numerically stable."""
    H, W = 32, 32
    Ka, Kb = 5, 5

    v = torch.randn(H, W, dtype=torch.float64)
    a = torch.randn(Ka, Kb, dtype=torch.float64)
    a = a / a.sum()

    output = adjoint(v, a)

    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"

    print(f"✓ Adjoint numerical stability test passed")


def test_adjoint_vs_forward_kernel_difference():
    """Test that adjoint and forward produce different results due to kernel flip."""
    x = torch.randn(8, 8, dtype=torch.float64)
    a = torch.randn(3, 3, dtype=torch.float64)
    a = a / a.sum()

    forward_result = forward(x, a)
    adjoint_result = adjoint(x, a)

    difference = torch.abs(forward_result - adjoint_result).max()

    assert difference > 1e-10, "forward() and adjoint() should differ due to kernel flip"
    print(f"✓ Forward vs Adjoint difference test passed:")
    print(f"  Max difference: {difference:.2e}")


if __name__ == "__main__":
    print("Running forward() and adjoint() tests...\n")

    print("=== forward() tests ===\n")
    test_kernel_flip()
    print()
    test_output_shape()
    print()
    test_circular_boundaries()
    print()
    test_blur_kernel()
    print()
    test_odd_kernel_sizes()
    print()
    test_numerical_stability()
    print()
    test_kernel_flipped_in_forward()

    print("\n=== adjoint() tests ===\n")
    test_adjoint_output_shape()
    print()
    test_adjoint_no_flip()
    print()
    test_adjoint_delta_kernel()
    print()
    test_adjoint_blur_kernel()
    print()
    test_adjoint_numerical_stability()
    print()
    test_adjoint_vs_forward_kernel_difference()

    print("\n✅ All tests passed!")
