import random
from typing import Optional
import torch
from loguru import logger


def set_device(device_type: str = "auto") -> str:
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


@logger.catch(message="Unable to set seed for this run/experiment.", reraise=True)
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
