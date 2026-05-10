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
   appears in more than one split.
6. Providing PyTorch Dataset and DataLoader objects.

Public API:
get_loaders(batch_size=32, freq_threshold=5, ...)
-> train_loader, val_loader, test_loader, vocab
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
    <PAD> = 0
    <SOS> = 1
    <EOS> = 2
    <UNK> = 3
    """

    def __init__(self, freq_threshold: int = 5):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self) -> int:
        return len(self.itos)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    def build(self, sentences: List[str]) -> None:
        counter = Counter()
        for sentence in sentences:
            counter.update(self.tokenize(sentence))

        idx = 4
        for word, count in counter.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.stoi.get(tok, self.stoi["<UNK>"]) for tok in tokens]


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------
class Flickr8kDataset(Dataset):
    """Flickr8k caption dataset reading from an in-memory dataframe.

    If cache_transformed=True, transforms are applied once during __init__
    and the transformed tensors are stored in RAM.

    If cache_transformed=False, raw decoded PIL images are stored and
    transforms are applied on the fly in __getitem__.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        transform=None,
        cache_transformed: bool = True,
    ):
        df = df.reset_index(drop=True)
        self.df = df
        self.vocab = vocab
        self.transform = transform
        self.cache_transformed = cache_transformed
        self.captions = df["caption"].tolist()

        raw_images = df["image"].tolist()

        self.images = []
        self.cached_tensors = []

        for img in raw_images:
            if not isinstance(img, Image.Image):
                img = Image.open(io.BytesIO(img)).convert("RGB")
            else:
                img = img.convert("RGB")

            if self.cache_transformed and self.transform is not None:
                self.cached_tensors.append(self.transform(img))
            else:
                self.images.append(img)

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_transformed:
            img = self.cached_tensors[idx]
        else:
            img = self.images[idx]
            if self.transform is not None:
                img = self.transform(img)

        caption = self.captions[idx]
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

    assert train_imgs.isdisjoint(val_imgs)
    assert train_imgs.isdisjoint(test_imgs)
    assert val_imgs.isdisjoint(test_imgs)

    print(f"Unique images : {n}")
    print(f" Train images : {len(train_imgs)} ({len(train_df)} captions)")
    print(f" Val images   : {len(val_imgs)} ({len(val_df)} captions)")
    print(f" Test images  : {len(test_imgs)} ({len(test_df)} captions)")

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 5. Standard image transforms
# ---------------------------------------------------------------------------
def get_transforms(image_size: int = 224):
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
# 6. Load Flickr8k from HuggingFace
# ---------------------------------------------------------------------------
def _extract_captions(example: dict) -> List[str]:
    numbered = [k for k in example.keys() if k.startswith("caption_")]
    if numbered:
        numbered.sort()
        return [str(example[k]) for k in numbered if example[k] is not None]

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


def load_flickr8k(target_dir: str = "flickr8k") -> pd.DataFrame:
    from datasets import load_dataset

    print("Loading Flickr8k from HuggingFace (jxie/flickr8k)...")
    ds = load_dataset("jxie/flickr8k", cache_dir=target_dir)

    rows = []
    image_id_counter = 0

    for split_name in ds.keys():
        for example in ds[split_name]:
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


# ---------------------------------------------------------------------------
# 7. Main entry-point
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
    cache_train: bool = True,
    cache_val: bool = True,
    cache_test: bool = True,
):
    """End-to-end helper.

    cache_train=False is usually better if your train transform contains
    random augmentation, because caching fixes one random view forever.
    cache_val/test=True is usually safe because eval transforms are deterministic.
    """

    df = load_flickr8k(target_dir)

    train_df, val_df, test_df = image_level_split(
        df,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )

    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build(train_df["caption"].tolist())
    print(f"Vocabulary size : {len(vocab)}")

    train_tf, eval_tf = get_transforms(image_size=image_size)

    train_ds = Flickr8kDataset(
        train_df,
        vocab,
        transform=train_tf,
        cache_transformed=cache_train,
    )
    val_ds = Flickr8kDataset(
        val_df,
        vocab,
        transform=eval_tf,
        cache_transformed=cache_val,
    )
    test_ds = Flickr8kDataset(
        test_df,
        vocab,
        transform=eval_tf,
        cache_transformed=cache_test,
    )

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
        batch_size=8,
        num_workers=0,
        cache_train=False,
        cache_val=True,
        cache_test=True,
    )

    imgs, caps, lens = next(iter(train_loader))
    print("\n--- One training batch ---")
    print(f"images   : {imgs.shape} (B, 3, H, W)")
    print(f"captions : {caps.shape} (B, L_padded)")
    print(f"lengths  : {lens.tolist()}")
    print(f"sample 0 : {[vocab.itos[i] for i in caps[0].tolist()]}")
