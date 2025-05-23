#!/usr/bin/env python3
"""Test script to verify the SoftNCutsLoss fix for inplace operations."""

import torch
import torch.nn as nn
from src.cell_interactome.loss.soft_ncuts import SoftNCutsLoss


def test_soft_ncuts_gradient():
    """Test that SoftNCutsLoss doesn't have inplace operation issues."""
    print("Testing SoftNCutsLoss gradient computation...")

    # Enable anomaly detection to catch inplace operations
    torch.autograd.set_detect_anomaly(True)

    # Create test inputs matching the shape from the error
    batch_size = 5
    num_classes = 2
    crop_size = [256, 256, 256]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create random inputs
    labels = torch.randn(
        batch_size, num_classes, *crop_size, device=device, requires_grad=True
    )
    inputs = torch.randn(batch_size, 1, *crop_size, device=device, requires_grad=True)

    # Apply softmax to labels to make them proper probabilities
    labels = torch.softmax(labels, dim=1)

    # Create loss function
    loss_fn = SoftNCutsLoss(
        data_shape=crop_size,
        intensity_sigma=1.0,
        spatial_sigma=4.0,
        radius=2,
    ).to(device)

    try:
        # Forward pass
        loss = loss_fn(labels=labels, inputs=inputs)
        print(f"Forward pass successful. Loss: {loss.item():.6f}")

        # Backward pass
        loss.backward()
        print("Backward pass successful - no inplace operation errors!")

        # Check gradients exist
        assert labels.grad is not None, "Labels gradient is None"
        assert inputs.grad is not None, "Inputs gradient is None"
        print(f"Gradients computed successfully:")
        print(f"  Labels grad norm: {labels.grad.norm().item():.6f}")
        print(f"  Inputs grad norm: {inputs.grad.norm().item():.6f}")

        return True

    except RuntimeError as e:
        if "inplace operation" in str(e):
            print(f"ERROR: Inplace operation detected: {e}")
            return False
        else:
            print(f"ERROR: Other runtime error: {e}")
            return False
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return False


def test_multiple_iterations():
    """Test multiple forward/backward passes to ensure no accumulated issues."""
    print("\nTesting multiple iterations...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2  # Smaller batch for faster testing
    num_classes = 2
    crop_size = [64, 64, 64]  # Smaller size for faster testing

    loss_fn = SoftNCutsLoss(
        data_shape=crop_size,
        intensity_sigma=1.0,
        spatial_sigma=4.0,
        radius=2,
    ).to(device)

    for i in range(5):
        # Create fresh inputs for each iteration
        labels = torch.randn(
            batch_size, num_classes, *crop_size, device=device, requires_grad=True
        )
        inputs = torch.randn(
            batch_size, 1, *crop_size, device=device, requires_grad=True
        )
        labels = torch.softmax(labels, dim=1)

        # Forward and backward
        loss = loss_fn(labels=labels, inputs=inputs)
        loss.backward()

        print(f"  Iteration {i + 1}: Loss = {loss.item():.6f}")

    print("Multiple iterations completed successfully!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing SoftNCutsLoss fix for inplace operations")
    print("=" * 60)

    success1 = test_soft_ncuts_gradient()
    success2 = test_multiple_iterations()

    if success1 and success2:
        print("\n✅ ALL TESTS PASSED - SoftNCutsLoss fix is working!")
    else:
        print("\n❌ TESTS FAILED - Issues still exist")

    print("=" * 60)
