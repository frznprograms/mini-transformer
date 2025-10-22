import torch
import numpy as np
from loguru import logger
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class CharDataset(Dataset):
    def __init__(self, text: str, context_size: int = 5):
        vocab = list("abcdefghijklmnopqrstuvwxyz ")
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        logger.success("Prepared character ids and reverse mappings.")
        self.context_size = context_size

        arr = np.fromiter(
            (self.stoi.get(ch, self.stoi[" "]) for ch in tqdm(text, leave=False)),
            dtype=np.int64,
        )
        self.encoded = torch.from_numpy(arr)
        logger.success("Prepared token encodings.")

    def __len__(self) -> int:
        # number of valid (context, next_token) pairs
        return len(self.encoded) - self.context_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # create y as X that has been offset by 1
        X = self.encoded[idx : idx + self.context_size]
        y = self.encoded[idx + 1 : idx + 1 + self.context_size]

        return X, y

    def decode(self, ids: torch.Tensor) -> str:
        return "".join(self.itos[i.item()] for i in ids)  # type: ignore


if __name__ == "__main__":
    with open("data/text8", "r") as f:
        full_text = f.read()

    max_size = 7000
    data = full_text[:max_size]

    cd = CharDataset(text=data, context_size=10)

    actual_text = data[:11]
    print(f"Actual X: {actual_text[:10]}")
    print(f"Actual y: {actual_text[1:]}")

    X, y = cd[0]
    print(f"X: {X}")
    print(f"y: {y}")

    # check that reverse mapping works
    X_itos = cd.decode(X)
    y_itos = cd.decode(y)
    print(f"Decoded X: {X_itos}")
    print(f"Decoded y: {y_itos}")
