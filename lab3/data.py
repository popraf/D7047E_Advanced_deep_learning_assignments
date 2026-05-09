"""
data.py
=======
Module 1 — Data pipeline for the Flickr8k image-captioning project.

This module is responsible for:
    1. Downloading the Flickr8k dataset from HuggingFace (parquet format).
    2. Extracting images and captions from the parquet files.
    3. Cleaning and tokenizing the captions.
    4. Building a Vocabulary (word <-> index mapping).
    5. Performing an IMAGE-LEVEL train/val/test split, so that no image
       appears in more than one split (each image has ~5 captions; if we
       split by row we leak captions of the same image into train/test
       and BLEU scores become meaningless).
    6. Providing PyTorch Dataset and DataLoader objects.

Public API (used by Person 2 and Person 3):
    get_loaders(batch_size=32, freq_threshold=5, ...)
        -> train_loader, val_loader, test_loader, vocab

Author: Chris
"""

import io
import re
from collections import Counter
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# 1. Vocabulary
# ---------------------------------------------------------------------------
class Vocabulary:
    """Maps words to integers and back.

    Special tokens:
        <PAD> = 0   (used to pad shorter captions in a batch)
        <SOS> = 1   (start-of-sentence; first input to the LSTM)
        <EOS> = 2   (end-of-sentence; tells the model to stop generating)
        <UNK> = 3   (unknown / rare word)
    """

    def __init__(self, freq_threshold: int = 5):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self) -> int:
        return len(self.itos)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    def build(self, sentences: List[str]) -> None:
        """Build the vocabulary from a list of caption strings.

        Words that appear fewer than `freq_threshold` times are dropped
        (they will be mapped to <UNK> at lookup time).
        """
        counter = Counter()
        for sentence in sentences:
            counter.update(self.tokenize(sentence))

        idx = 4  # 0..3 are reserved
        for word, count in counter.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text: str) -> List[int]:
        """Convert a sentence into a list of token IDs."""
        tokens = self.tokenize(text)
        return [self.stoi.get(tok, self.stoi["<UNK>"]) for tok in tokens]


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------
class Flickr8kDataset(Dataset):
    """Flickr8k caption dataset reading from an in-memory dataframe.

    Each row holds:
        - image_id: a unique identifier for the image (used for split logic)
        - image:    raw PIL.Image object
        - caption:  one caption string

    An image may appear multiple times in the dataframe (once per caption),
    but the IMAGE-LEVEL SPLIT in get_loaders() guarantees that all
    captions of a given image stay in the same split.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img = row["image"]
        caption = row["caption"]

        if not isinstance(img, Image.Image):
            # PIL image stored as bytes after parquet loading
            img = Image.open(io.BytesIO(img)).convert("RGB")
        else:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # --- numericalize caption with <SOS> ... <EOS> ---
        tokens = (
            [self.vocab.stoi["<SOS>"]]
            + self.vocab.numericalize(caption)
            + [self.vocab.stoi["<EOS>"]]
        )
        return img, torch.tensor(tokens, dtype=torch.long)


# ---------------------------------------------------------------------------
# 3. Collate function for variable-length captions
# ---------------------------------------------------------------------------
class CapsCollate:
    """Pads all captions in a batch to the length of the longest one."""

    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = torch.stack([item[0] for item in batch], dim=0)
        captions = [item[1] for item in batch]
        lengths = torch.tensor([len(c) for c in captions], dtype=torch.long)
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        return imgs, captions, lengths


# ---------------------------------------------------------------------------
# 4. Image-level split
# ---------------------------------------------------------------------------
def image_level_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the captions dataframe by UNIQUE IMAGE IDs.

    This is the requirement: 'no image is in more than one split'.
    We FIRST split image IDs, THEN pull all rows belonging to those
    images into the corresponding split.
    """
    unique_images = (
        df["image_id"]
        .drop_duplicates()
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    n = len(unique_images)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_imgs = set(unique_images[:n_train])
    val_imgs = set(unique_images[n_train : n_train + n_val])
    test_imgs = set(unique_images[n_train + n_val :])

    train_df = df[df["image_id"].isin(train_imgs)].reset_index(drop=True)
    val_df = df[df["image_id"].isin(val_imgs)].reset_index(drop=True)
    test_df = df[df["image_id"].isin(test_imgs)].reset_index(drop=True)

    # --- sanity check: no image leaks across splits ---
    assert train_imgs.isdisjoint(val_imgs)
    assert train_imgs.isdisjoint(test_imgs)
    assert val_imgs.isdisjoint(test_imgs)

    print(f"Unique images   : {n}")
    print(f"  Train images  : {len(train_imgs)}  ({len(train_df)} captions)")
    print(f"  Val   images  : {len(val_imgs)}  ({len(val_df)} captions)")
    print(f"  Test  images  : {len(test_imgs)}  ({len(test_df)} captions)")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 5. Standard image transforms (matches ImageNet-pretrained CNNs)
# ---------------------------------------------------------------------------
def get_transforms(image_size: int = 224):
    """Returns (train_transform, eval_transform).

    Train uses random crop + flip for augmentation; eval uses deterministic
    resize. Normalization values are the standard ImageNet ones —
    required because the encoder (ResNet/VGG) was pretrained on ImageNet.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    return train_transform, eval_transform


# ---------------------------------------------------------------------------
# 6. Load Flickr8k from HuggingFace (parquet format)
# ---------------------------------------------------------------------------
def load_flickr8k(target_dir: str = "flickr8k") -> pd.DataFrame:
    """Loads the Flickr8k dataset from HuggingFace into a single DataFrame.

    The HuggingFace dataset `jxie/flickr8k` ships as parquet shards under
    a `data/` directory. Each row contains an embedded PIL image plus
    captions.

    Returns a long-format DataFrame with one (image_id, image, caption)
    triple per row — i.e. each image is repeated for each of its captions.
    """
    from datasets import load_dataset

    print("Loading Flickr8k from HuggingFace (jxie/flickr8k)...")
    # Load all splits in the HF dataset and concatenate them. We re-split
    # later, at the image level, to satisfy the assignment requirement.
    ds = load_dataset("jxie/flickr8k", cache_dir=target_dir)

    rows = []
    image_id_counter = 0
    for split_name in ds.keys():
        for example in ds[split_name]:
            # Detect captions field — HF datasets expose either a single
            # "caption" field or a list under "captions"/"sentences".
            captions = _extract_captions(example)
            img = example["image"]
            for cap in captions:
                rows.append(
                    {
                        "image_id": image_id_counter,
                        "image": img,
                        "caption": cap,
                    }
                )
            image_id_counter += 1

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} caption rows over {df['image_id'].nunique()} images")
    return df


def _extract_captions(example: dict) -> List[str]:
    """Pulls all caption strings out of a HuggingFace row.

    Supports several schemas seen in the wild:
      - {"caption_0": "...", "caption_1": "...", ...}   (jxie/flickr8k)
      - {"captions": ["...", "...", ...]}
      - {"caption": "..."}
    """
    # Schema 1: caption_0, caption_1, ...
    numbered = [k for k in example.keys() if k.startswith("caption_")]
    if numbered:
        numbered.sort()  # caption_0, caption_1, caption_2, ...
        return [str(example[k]) for k in numbered if example[k] is not None]

    # Schema 2/3: list or single string
    for key in ("captions", "sentences", "caption"):
        if key in example:
            value = example[key]
            if isinstance(value, list):
                out = []
                for v in value:
                    if isinstance(v, dict):
                        out.append(v.get("raw") or v.get("caption") or str(v))
                    else:
                        out.append(str(v))
                return out
            elif isinstance(value, str):
                return [value]

    raise KeyError(f"No caption field found in example. Keys: {list(example.keys())}")


# ---------------------------------------------------------------------------
# 7. Main entry-point used by Person 2 and Person 3
# ---------------------------------------------------------------------------
def get_loaders(
    batch_size: int = 32,
    freq_threshold: int = 5,
    image_size: int = 224,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    num_workers: int = 2,
    target_dir: str = "flickr8k",
    seed: int = 42,
):
    """End-to-end helper. Call this from train.py / evaluate.py.

    Returns:
        train_loader, val_loader, test_loader, vocab
    """
    # 1. Load dataset (downloads on first run)
    df = load_flickr8k(target_dir)

    # 2. Image-level split (BEFORE building vocab — vocab uses train only,
    #    to avoid information leakage from val/test captions)
    train_df, val_df, test_df = image_level_split(
        df, train_frac=train_frac, val_frac=val_frac, seed=seed
    )

    # 3. Build vocabulary from TRAIN captions only
    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build(train_df["caption"].tolist())
    print(f"Vocabulary size : {len(vocab)}")

    # 4. Datasets
    train_tf, eval_tf = get_transforms(image_size=image_size)
    train_ds = Flickr8kDataset(train_df, vocab, transform=train_tf)
    val_ds = Flickr8kDataset(val_df, vocab, transform=eval_tf)
    test_ds = Flickr8kDataset(test_df, vocab, transform=eval_tf)

    # 5. Loaders
    pad_idx = vocab.stoi["<PAD>"]
    collate = CapsCollate(pad_idx=pad_idx)

    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        collate_fn=collate,
    )
    return train_loader, val_loader, test_loader, vocab


if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab = get_loaders(
        batch_size=8, num_workers=0
    )
    imgs, caps, lens = next(iter(train_loader))
    print("\n--- One training batch ---")
    print(f"images   : {imgs.shape}        (B, 3, H, W)")
    print(f"captions : {caps.shape}        (B, L_padded)")
    print(f"lengths  : {lens.tolist()}")
    print(f"sample 0 : {[vocab.itos[i] for i in caps[0].tolist()]}")
