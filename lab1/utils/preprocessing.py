import logging
import os
import pickle
import re
from collections import Counter
from typing import Dict, List, Tuple

import nltk
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Constants — keep IDENTICAL across ANN / LSTM / Transformer
RANDOM_SEED = 42
TEST_SIZE   = 0.10   # 10 % held-out test
VAL_SIZE    = 0.10   # 10 % validation (from non-test data)
MAX_SEQ_LEN = 64     # token length cap for Transformer / LSTM
MIN_FREQ    = 2      # min word freq to enter vocabulary (raised from 1 → reduces vocab bloat)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX   = 0
UNK_IDX   = 1

MODEL_DIR = "model"  # shared output directory for vocab + checkpoints


def _ensure_dir(path: str) -> None:
    """Create *path* and all parents if they do not already exist."""
    os.makedirs(path, exist_ok=True)


# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """Lower-case + strip noise. Optionally remove stop-words."""
    text = text.lower()
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", " ", text)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if remove_stopwords:
        stop   = set(stopwords.words("english"))
        tokens = word_tokenize(text)
        text   = " ".join(w for w in tokens if w not in stop)

    return text


# ── Vocabulary ────────────────────────────────────────────────────────────────
class Vocabulary:
    """Word-to-index mapping built from training data."""

    def __init__(self, min_freq: int = MIN_FREQ):
        self.min_freq  = min_freq
        self.token2idx: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2token: Dict[int, str] = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}

    def build(self, sentences: List[str]) -> None:
        counter: Counter = Counter()
        for sent in sentences:
            counter.update(word_tokenize(sent))
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx]   = token

    def encode(self, sentence: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
        tokens = word_tokenize(sentence)[:max_len]
        ids    = [self.token2idx.get(t, UNK_IDX) for t in tokens]
        ids   += [PAD_IDX] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.token2idx)

    def save(self, path: str) -> None:
        _ensure_dir(os.path.dirname(path) or ".")
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.debug(f"[preprocessing] Vocabulary saved → {path}")

    @staticmethod
    def load(path: str) -> "Vocabulary":
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    """Returns (token_ids_tensor, label_tensor) pairs."""

    def __init__(
        self,
        sentences: List[str],
        labels:    List[int],
        vocab:     Vocabulary,
        max_len:   int = MAX_SEQ_LEN,
    ):
        self.encodings = [
            torch.tensor(vocab.encode(s, max_len), dtype=torch.long)
            for s in sentences
        ]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.encodings[idx], self.labels[idx]


# ── Main entry-point ──────────────────────────────────────────────────────────
def load_data(
    filepath:         str  = "amazon_cells_labelled.txt",
    remove_stopwords: bool = False,
    max_seq_len:      int  = MAX_SEQ_LEN,
    batch_size:       int  = 32,
    save_vocab:       bool = True,
    vocab_path:       str  = os.path.join(MODEL_DIR, "vocab.pkl"),
) -> Tuple[DataLoader, DataLoader, DataLoader, "Vocabulary"]:
    """
    Load, clean, split, build vocab, return DataLoaders + vocab.

    Splits are stratified and seeded for fair comparison across
    ANN / LSTM / Transformer.

    Returns
    -------
    train_loader, val_loader, test_loader, vocab
    """
    df = pd.read_csv(filepath, delimiter="\t", header=None, names=["Sentence", "Class"])
    df.dropna(inplace=True)

    df["Sentence"] = df["Sentence"].apply(
        lambda x: clean_text(x, remove_stopwords=remove_stopwords)
    )
    df = df[df["Sentence"].str.strip() != ""].reset_index(drop=True)

    sentences: List[str] = df["Sentence"].tolist()
    labels:    List[int] = df["Class"].astype(int).tolist()

    # Stratified split: 80 % train | 10 % val | 10 % test
    X_train, X_temp, y_train, y_temp = train_test_split(
        sentences, labels,
        test_size    = TEST_SIZE + VAL_SIZE,  # 0.20
        random_state = RANDOM_SEED,
        stratify     = labels,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size    = 0.5,   # equal halves → 10 % val, 10 % test
        random_state = RANDOM_SEED,
        stratify     = y_temp,
    )

    logger.debug(
        f"[preprocessing] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
    )

    # Vocabulary built from TRAINING SET ONLY (no leakage)
    vocab = Vocabulary(min_freq=MIN_FREQ)
    vocab.build(X_train)
    logger.debug(f"[preprocessing] Vocabulary size: {len(vocab)}")

    if save_vocab:
        vocab.save(vocab_path)

    train_ds = SentimentDataset(X_train, y_train, vocab, max_len=max_seq_len)
    val_ds   = SentimentDataset(X_val,   y_val,   vocab, max_len=max_seq_len)
    test_ds  = SentimentDataset(X_test,  y_test,  vocab, max_len=max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, vocab


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_FILE = "amazon_cells_labelled.txt"
    train_loader, val_loader, test_loader, vocab = load_data(filepath=DATA_FILE)

    batch_x, batch_y = next(iter(train_loader))
    print(f"Batch input shape  : {batch_x.shape}")   # (32, 64)
    print(f"Batch label shape  : {batch_y.shape}")   # (32,)
    print(f"Sample labels      : {batch_y[:8].tolist()}")
    print(f"Sample token ids   : {batch_x[0, :12].tolist()}")
    print(f"Vocab size         : {len(vocab)}")
