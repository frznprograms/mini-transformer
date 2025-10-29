import torch
import numpy as np
from loguru import logger
from torch.utils.data import Dataset
from dataclasses import dataclass


@dataclass
class SegmentedCharDataset(Dataset):
    encoded: torch.Tensor
    segments: list[tuple[int, int]]
    context_size: int

    def __post_init__(self):
        """
        Precomputes how many (X, y) pairs each segment contributes.
        For a given segment defined by start and end indices (s, e), the number of valid
        sliding windows within that segment is:
            max((e - s) - context_size, 0)
        Each sliding window corresponds to a pair:
            X = encoded[i : i + context_size]
            y = encoded[i + 1 : i + 1 + context_size]
        where i ranges from s to e - context_size - 1.
        Args:
            encoded (torch.Tensor): 1D tensor of integer token IDs representing the text.
                                    This tensor is shared across all dataset splits
                                    (train, val, test) to avoid re-encoding.
            segments (list[tuple[int, int]]): List of (start, end) index pairs, each defining
                                    a contiguous, non-overlapping slice of the encoded text.
                                    These represent the disjoint text regions assigned to
                                    this dataset split (train, val, or test).
            context_size (int): Length of each input sequence (sliding window).
                                Must be <= model's maximum sequence length,
                                but does NOT need to match d_model.
        Returns:
            None
        """
        self._counts = []
        for start, end in self.segments:
            c = max((end - start) - self.context_size, 0)
            self._counts.append(c)

        self._cumsum = np.cumsum(self._counts)  # for O(logn) index mapping
        self._total = int(self._cumsum[-1]) if len(self._cumsum) else 0
        logger.success("Prepared needed utilities for dataset successfully.")

    def __len__(self):
        return self._total

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Map global idx to (segment_id, offset) using binary search on cumulative counts
        if idx < 0 or idx >= self._total:
            raise IndexError
        seg_id = int(np.searchsorted(self._cumsum, idx, side="right"))
        prev_cum = 0 if seg_id == 0 else self._cumsum[seg_id - 1]
        offset_in_seg = idx - prev_cum

        s, e = self.segments[seg_id]
        start = s + offset_in_seg
        X = self.encoded[start : start + self.context_size]
        y = self.encoded[start + 1 : start + 1 + self.context_size]

        return torch.tensor(X), torch.tensor(y)
