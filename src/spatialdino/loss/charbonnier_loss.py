import torch


def generalized_charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.45,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Implements the generalized Charbonnier loss function from https://cs.brown.edu/~dqsun/pubs/cvpr_2010_flow.pdf

    Parameters
    ----------
    pred : torch.Tensor
        predicted tensor
    target : torch.Tensor
        target tensor
    alpha : float, optional
        Shape parameter that controls the robustness, by default 0.45
    eps : float, optional
        Small constant for numerical stability, by default 1e-3

    Returns
    -------
    torch.Tensor
        Computed loss value
    """
    return (((pred - target) ** 2 + eps**2) ** alpha).mean()
