import random
from typing import Optional
import torch
from loguru import logger
import numpy as np
from tqdm.auto import tqdm

from src.datasets.segment import SegmentedCharDataset


VOCAB = list("abcdefghijklmnopqrstuvwxyz ")
STOI = {ch: i for i, ch in enumerate(VOCAB)}
ITOS = {i: ch for ch, i in STOI.items()}


def set_device(device_type: str = "auto") -> str:
    """
    Searches for available hardware and sets device for model training
    based on user preference.
    Args:
        device_type (str): device type specified by user, one of ["auto", "cpu", "cuda", "mps"].
                           Device will default to auto selection unless one is specified.
    Returns:
        str: device type that has been set OR raises ValueError if device_type arg is wrong.
    """
    if device_type == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    elif device_type == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        else:
            logger.warning("CUDA not available, defaulting to CPU.")
            return "cpu"

    elif device_type == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        else:
            logger.warning("MPS not available, defaulting to CPU.")
            return "cpu"

    elif device_type == "cpu":
        return "cpu"

    else:
        raise ValueError(f"Unrecognized device type: {device_type}")


def set_seeds(seed_num: Optional[int], deterministic: bool = True) -> int:
    if seed_num is None:
        logger.warning("A seed was not detected. Setting seed to 42.")
        seed_num = 42
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed_num


@logger.catch(message="Unable to create causal masking.", reraise=True)
def causal_mask(dim: int):
    return torch.tril(torch.ones(dim, dim))


@logger.catch(message="Unable to segment indices", reraise=True)
def segment_indices(
    text: str, segment_len: int, context_size: int
) -> list[tuple[int, int]]:
    text_len = len(text)
    segments = []
    for start in range(0, text_len, segment_len):
        end = min(start + segment_len, text_len)
        if (end - start) >= context_size + 1:  # must support at least one window
            segments.append((start, end))

    logger.success("Segmented dataset into index segments successfully.")

    return segments


def assign_segments_randomly(
    segments: list[tuple[int, int]], ratios: tuple[float, float, float], seed: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    random.Random(seed).shuffle(segments)
    n = len(segments)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    # n_test = n - n_train - n_val

    train = segments[:n_train]
    val = segments[n_train : n_train + n_val]
    test = segments[n_train + n_val :]
    logger.success("Prepared random segment splits successfully.")

    return train, val, test


def encode(text: str) -> torch.Tensor:
    arr = np.fromiter(
        (STOI.get(ch, STOI[" "]) for ch in tqdm(text, leave=False)),
        dtype=np.int64,
    )
    encoded = torch.from_numpy(arr)
    logger.success("Prepared character encodings.")
    return encoded


def decode(ids: torch.Tensor) -> str:
    return "".join(ITOS[i.item()] for i in ids)  # type: ignore


def save_text8_splits(
    text: str,
    path: str,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    segment_len: int = 4096,
    context_size: int = 128,
    seed: int = 42,
) -> None:
    encoded = encode(text=text)

    segments = segment_indices(
        text=text, segment_len=segment_len, context_size=context_size
    )
    if not segments:
        logger.error(
            "No segments created. Reduce segment_len or ensure text is long enough."
        )

    train_seg, val_seg, test_seg = assign_segments_randomly(
        segments=segments, ratios=ratios, seed=seed
    )
    torch.save(
        {
            "encoded": encoded,
            "train_segments": train_seg,
            "val_segments": val_seg,
            "test_segments": test_seg,
            "context_size": context_size,
        },
        path,
    )
    logger.success(f"Saved dataset splits and encoding to {path}")


def load_data_splits(
    path: str,
) -> tuple[
    SegmentedCharDataset, SegmentedCharDataset, SegmentedCharDataset, torch.Tensor
]:
    data = torch.load(path)
    encoded = data["encoded"]
    context_size = data["context_size"]
    train = SegmentedCharDataset(
        encoded=encoded, segments=data["train_segments"], context_size=context_size
    )
    val = SegmentedCharDataset(
        encoded=encoded, segments=data["val_segments"], context_size=context_size
    )
    test = SegmentedCharDataset(
        encoded=encoded, segments=data["test_segments"], context_size=context_size
    )
    logger.success("Loaded text8 dataset successfully. Happy training!")

    return train, val, test, encoded
