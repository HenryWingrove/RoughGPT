from __future__ import annotations

"""Dataset supplying tokens alongside path signatures.

This dataset is modelled after ``FISignatureSlidingDataset`` and is
responsible for augmenting token sequences with local and global path
signatures.  As ITCH messages are iterated over, the dataset maintains
two signature streams using :func:`iisignature.sig_stream`: one for the
entire history (global) and one over a fixed size sliding window
(local).  ``__getitem__`` returns a tuple ``(token_ids, sig_vectors)``
where both tensors share the same sequence length so that a model can
learn to predict the next token.
"""

from collections import deque
from typing import Iterable, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import iisignature


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _make_stream(dim: int, depth: int):
    """Return a callable implementing ``iisignature.sig_stream``.

    Older versions of :mod:`iisignature` did not expose ``sig_stream``;
    for such environments we provide a minimal Python fallback which
    accumulates the path and recomputes the signature using
    :func:`iisignature.sig` when called.  The returned object mimics the
    interface of ``sig_stream`` by accepting a single increment and
    returning the current signature.
    """

    if hasattr(iisignature, "sig_stream"):
        return iisignature.sig_stream(dim, depth)

    # Fallback implementation -------------------------------------------------
    prep = iisignature.prepare(dim, depth)
    path: list[np.ndarray] = []

    def update(x: np.ndarray) -> np.ndarray:
        path.append(np.asarray(x, dtype=np.float64))
        return iisignature.sig(np.asarray(path, dtype=np.float64), prep)

    return update


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SignatureSlidingDataset(Dataset):
    """Sliding window dataset providing tokens and signature features.

    Parameters
    ----------
    token_ids:
        Either a path to a ``.npy`` file or an array containing the
        encoded token ids of the ITCH messages.
    path:
        Numeric representation of the ITCH messages used to compute the
        signatures.  Each row corresponds to a single tick and each
        column to a stream dimension.
    sig_depth:
        Depth of the signature calculation.
    context_length:
        Number of tokens returned by ``__getitem__``.  The dataset length
        is reduced accordingly so that the model can predict the next
        token in the sequence.
    window:
        Size of the sliding window for the local signature stream.
    """

    def __init__(
        self,
        token_ids: Union[str, np.ndarray],
        path: Union[str, np.ndarray],
        sig_depth: int,
        context_length: int,
        window: int,
    ) -> None:
        if isinstance(token_ids, str):
            self.tokens = np.load(token_ids).astype(np.int64)
        else:
            self.tokens = np.asarray(token_ids, dtype=np.int64)

        if isinstance(path, str):
            self.path = np.load(path).astype(np.float32)
        else:
            self.path = np.asarray(path, dtype=np.float32)

        assert self.tokens.shape[0] == self.path.shape[0], (
            "tokens and path must have the same number of ticks",
        )

        self.context_length = int(context_length)
        self.window = int(window)
        self.depth = int(sig_depth)
        self.dim = self.path.shape[1]
        self.sig_len = iisignature.siglength(self.dim, self.depth)

        # Pre-compute signature vectors for the entire sequence.
        self._precompute_signatures()

    # ------------------------------------------------------------------
    def _precompute_signatures(self) -> None:
        global_stream = _make_stream(self.dim, self.depth)
        buffer: deque[np.ndarray] = deque(maxlen=self.window)

        self.sig_vectors = np.zeros(
            (len(self.path), 2 * self.sig_len), dtype=np.float32
        )

        for i, x in enumerate(self.path):
            # Update global signature stream.
            g_sig = global_stream(x)

            # Update local signature over the sliding window.
            buffer.append(x)
            l_stream = _make_stream(self.dim, self.depth)
            l_sig = None
            for y in buffer:
                l_sig = l_stream(y)
            if l_sig is None:  # pragma: no cover - safety for empty buffer
                l_sig = np.zeros(self.sig_len, dtype=np.float32)

            self.sig_vectors[i] = np.concatenate([l_sig, g_sig])

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.tokens) - self.context_length

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        end = idx + self.context_length
        token_ids = torch.tensor(self.tokens[idx:end], dtype=torch.long)
        sig_vectors = torch.tensor(
            self.sig_vectors[idx:end], dtype=torch.float32
        )
        return token_ids, sig_vectors


__all__ = ["SignatureSlidingDataset"]
