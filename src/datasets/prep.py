from loguru import logger

from src.utils.helpers import save_text8_splits

if __name__ == "__main__":
    logger.warning(
        "The following operation will override existing data of the same name. Please ensure data will be saved correctly."
    )
    logger.info("Now preparing small experimental dataset splits.")
    with open("data/raw/text8", "r") as f:
        full_text = f.read()

    max_size = 500000
    data = full_text[:max_size]

    save_text8_splits(
        text=data,
        path="data/medium/small_medium_data.pt",
        ratios=(0.8, 0.1, 0.1),
        segment_len=2048,
        context_size=128,
        seed=1,
    )
