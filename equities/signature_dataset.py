import torch
from typing import Callable, Tuple, Optional

class SignatureDataset:
    """Wraps an existing get_batch function to also return signature features.

    This minimal implementation is sufficient for enabling the `--use-signatures`
    flag in training scripts.  It expects a callable `get_batch_fn` that returns
    a tuple of `(tokens, targets)` and augments the result with a tensor of
    signatures.  The signature tensor is filled with zeros and has shape
    `(batch, seq_len, sig_dim)`.
    """

    def __init__(self, get_batch_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]], sig_dim: int = 0) -> None:
        self.get_batch_fn = get_batch_fn
        self.sig_dim = sig_dim

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, targets = self.get_batch_fn(split)
        sig_shape = (tokens.size(0), tokens.size(1), self.sig_dim)
        sigs = torch.zeros(sig_shape, device=tokens.device, dtype=torch.float32)
        return tokens, sigs, targets
