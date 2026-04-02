import torch


@torch.no_grad()
def selective_adam_step(optimizer, visibility_mask):
    """Selective Adam: only update visible Gaussians' parameters and momentum.

    For Gaussian parameters (1st dim == N), invisible entries' exp_avg,
    exp_avg_sq, and parameter values are preserved exactly as-is.
    Non-Gaussian parameters (e.g., scalars) use standard Adam.

    Args:
        optimizer: torch.optim.Adam instance
        visibility_mask: bool tensor of shape (N,), True = visible
    """
    N = visibility_mask.shape[0]
    inv_mask = ~visibility_mask

    # Save state + params for invisible Gaussians
    saved = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.dim() >= 1 and p.shape[0] == N:
                state = optimizer.state.get(p)
                if state and "exp_avg" in state:
                    saved[p] = (
                        state["exp_avg"][inv_mask].clone(),
                        state["exp_avg_sq"][inv_mask].clone(),
                        p.data[inv_mask].clone(),
                    )
                # Zero grad for invisible so Adam doesn't update them
                p.grad.data[inv_mask] = 0

    # Standard Adam step
    optimizer.step()

    # Restore invisible Gaussians' state
    for p, (ea, eas, pval) in saved.items():
        state = optimizer.state[p]
        state["exp_avg"][inv_mask] = ea
        state["exp_avg_sq"][inv_mask] = eas
        p.data[inv_mask] = pval
