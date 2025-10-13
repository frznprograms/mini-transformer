from torch.utils.data import Dataset
import torch


class Text8Dataset(Dataset):
    def __init__(self, text: str, context_size: int = 5):
        self.tokens = text.split()  # word-level tokenization
        # NOTE: vocab doubles as word ids
        self.vocab = {word: idx for idx, word in enumerate(sorted(set(self.tokens)))}
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.context_size = context_size

        self.encoded = [self.vocab[word] for word in self.tokens]

    @property
    def info(self) -> dict[str, int]:
        len_properties = {
            "vocab_size": len(self.vocab),
            "context_size": self.context_size,
            "num_windows": len(self.vocab) // self.context_size,
        }

        return len_properties

    def __len__(self) -> int:
        # number of valid (context, next_token) pairs
        return len(self.encoded) - self.context_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        X = self.encoded[idx : idx + self.context_size]
        y = self.encoded[idx + self.context_size]

        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)
