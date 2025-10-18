from torch.utils.data import Dataset
import torch


class CharDataset(Dataset):
    def __init__(self, text: str, context_size: int = 5):
        vocab = list("abcdefghijklmnopqrstuvwxyz ")
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.context_size = context_size

        self.encoded = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def __len__(self) -> int:
        # number of valid (context, next_token) pairs
        return len(self.encoded) - self.context_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # create y as X that has been offset by 1, each window non-overlapping
        start = idx * self.context_size
        X = self.encoded[start : start + self.context_size]
        y = self.encoded[start + 1 : start + 1 + self.context_size]

        return X, y
