from torch.utils.data import Dataset
import torch


class Text8Dataset(Dataset):
    def __init__(self, text: str, context_size: int = 5):
        self.tokens = [token for token in text.split()]  # word-level tokenization
        # NOTE: vocab doubles as word ids
        self.vocab = {word: idx for idx, word in enumerate(set(self.tokens))}
        self.context_size = context_size

    def __len__(self) -> dict[str, int]:
        len_properties = {
            "vocab_size": len(self.vocab),
            "context_size": self.context_size,
            "num_windows": len(self.vocab) // self.context_size,
        }

        return len_properties

    def __getitem__(self, idx: int) -> tuple(torch.Tensor, torch.Tensor):
        X = self.tokens[idx : idx + self.context_size]
        y = self.tokens[idx + self.context_size]

        return torch.tensor(X), torch.tensor(y)
