from src.utils.helpers import load_data_splits

# if __name__ == "__main__":
#     with open("data/raw/text8", "r") as f:
#         full_text = f.read()
#
#     print(f"Len of full_text = {len(full_text)}.")
#     print(f"First 25: {full_text[:25]}")


if __name__ == "__main__":
    train, val, test, encoded = load_data_splits(path="data/small/small_data.pt")
    print(len(train), len(val), len(test))
