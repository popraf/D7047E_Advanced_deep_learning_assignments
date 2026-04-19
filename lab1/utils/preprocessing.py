import logging
import os
import pickle
import re
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

import nltk
import pandas as pd
import torch
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

nltk.download("stopwords", quiet=True)

# Constants — keep IDENTICAL across ANN / LSTM / Transformer
RANDOM_SEED  = 42
TEST_SIZE    = 0.10   # 10 % held-out test
VAL_SIZE     = 0.10   # 10 % validation (from non-test data)
MAX_SEQ_LEN  = 64     # token length cap for Transformer / LSTM
MIN_FREQ     = 2      # min word freq to enter vocabulary

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX   = 0
UNK_IDX   = 1

MODEL_DIR = "model"   # shared output directory for vocab + checkpoints

_STOP_WORDS: set = set()   # populated lazily if remove_stopwords=True


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """Lower-case + strip noise. Uses fast .split() — no NLTK tokeniser."""
    text = text.lower()
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", " ", text)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if remove_stopwords:
        global _STOP_WORDS
        if not _STOP_WORDS:
            _STOP_WORDS = set(stopwords.words("english"))
        text = " ".join(w for w in text.split() if w not in _STOP_WORDS)

    return text


# ── Fast tokeniser (module-level for multiprocessing pickling) ────────────────
_REMOVE_SW = False   # set before pool.map call


def _clean_one(text: str) -> str:
    return clean_text(text, remove_stopwords=_REMOVE_SW)


def _parallel_clean(texts: List[str], remove_stopwords: bool = False) -> List[str]:
    """Clean all texts in parallel using all CPU cores."""
    global _REMOVE_SW
    _REMOVE_SW = remove_stopwords
    workers = min(cpu_count(), 8)
    with Pool(workers) as pool:
        cleaned = pool.map(_clean_one, texts, chunksize=2000)
    return cleaned


# ── Vocabulary ────────────────────────────────────────────────────────────────
class Vocabulary:
    """Word-to-index mapping built from pre-cleaned training sentences."""

    def __init__(self, min_freq: int = MIN_FREQ):
        self.min_freq = min_freq
        self.token2idx: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2token: Dict[int, str] = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}

    def build(self, sentences: List[str]) -> None:
        """Build vocab from already-cleaned sentences using fast .split()."""
        counter: Counter = Counter()
        for sent in sentences:
            counter.update(sent.split())          # ← .split() not word_tokenize
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

    def encode(self, sentence: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
        """Encode a pre-cleaned sentence. Uses fast .split()."""
        tokens = sentence.split()[:max_len]       # ← .split() not word_tokenize
        ids = [self.token2idx.get(t, UNK_IDX) for t in tokens]
        ids += [PAD_IDX] * (max_len - len(ids))
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
    """Accepts pre-encoded token-id lists — no tokenisation at Dataset init."""

    def __init__(
        self,
        encoded: List[List[int]],   # already encoded by vocab.encode()
        labels: List[int],
    ):
        self.encodings = [torch.tensor(ids, dtype=torch.long) for ids in encoded]
        self.labels    = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.encodings[idx], self.labels[idx]


def _encode_split(
    sentences: List[str], labels: List[int], vocab: Vocabulary, max_len: int
) -> SentimentDataset:
    """Encode a split into token-id tensors (fast, no NLTK)."""
    encoded = [vocab.encode(s, max_len) for s in sentences]
    return SentimentDataset(encoded, labels)


# ── Main entry-point ──────────────────────────────────────────────────────────
def load_data(
    filepath: str = None,
    hf_dataset=None,
    remove_stopwords: bool = False,
    max_seq_len: int = MAX_SEQ_LEN,
    batch_size: int = 32,
    save_vocab: bool = True,
    vocab_path: str = os.path.join(MODEL_DIR, "vocab.pkl"),
) -> Tuple[DataLoader, DataLoader, DataLoader, "Vocabulary"]:
    """
    Load, clean, split, build vocab, return DataLoaders + vocab.

    Accepts either:
      - filepath   : path to a .txt / .tsv file (original behaviour)
      - hf_dataset : HuggingFace DatasetDict with "train"/"test" splits,
                     columns "text" (str) and "label" (int 0/1).

    Cleaning is parallelised across all CPU cores.
    Tokenisation uses fast .split() — no NLTK word_tokenize overhead.

    Returns
    -------
    train_loader, val_loader, test_loader, vocab
    """
    if hf_dataset is not None:
        raw_train = hf_dataset["train"]
        raw_test  = hf_dataset["test"]

        print(f"[preprocessing] Cleaning {len(raw_train):,} train + "
              f"{len(raw_test):,} test samples (parallel) …")

        all_sentences  = _parallel_clean(raw_train["text"], remove_stopwords)
        all_labels     = [int(x) for x in raw_train["label"]]
        test_sentences = _parallel_clean(raw_test["text"],  remove_stopwords)
        test_labels    = [int(x) for x in raw_test["label"]]

        X_train, X_val, y_train, y_val = train_test_split(
            all_sentences, all_labels,
            test_size=VAL_SIZE,
            random_state=RANDOM_SEED,
            stratify=all_labels,
        )
        X_test, y_test = test_sentences, test_labels

    else:
        if filepath is None:
            raise ValueError("Either filepath or hf_dataset must be provided.")

        df = pd.read_csv(filepath, delimiter="\t", header=None, names=["Sentence", "Class"])
        df.dropna(inplace=True)

        print(f"[preprocessing] Cleaning {len(df):,} samples (parallel) …")
        cleaned = _parallel_clean(df["Sentence"].tolist(), remove_stopwords)
        df["Sentence"] = cleaned
        df = df[df["Sentence"].str.strip() != ""].reset_index(drop=True)

        sentences: List[str] = df["Sentence"].tolist()
        labels:    List[int] = df["Class"].astype(int).tolist()

        X_train, X_temp, y_train, y_temp = train_test_split(
            sentences, labels,
            test_size=TEST_SIZE + VAL_SIZE,
            random_state=RANDOM_SEED,
            stratify=labels,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            random_state=RANDOM_SEED,
            stratify=y_temp,
        )

    print(f"[preprocessing] Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    print("[preprocessing] Building vocabulary …")
    vocab = Vocabulary(min_freq=MIN_FREQ)
    vocab.build(X_train)
    print(f"[preprocessing] Vocabulary size: {len(vocab):,}")

    if save_vocab:
        vocab.save(vocab_path)
        print(f"[preprocessing] Vocab saved → {vocab_path}")

    print("[preprocessing] Encoding splits …")
    train_ds = _encode_split(X_train, y_train, vocab, max_seq_len)
    val_ds   = _encode_split(X_val,   y_val,   vocab, max_seq_len)
    test_ds  = _encode_split(X_test,  y_test,  vocab, max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, vocab


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_FILE = "amazon_cells_labelled.txt"
    train_loader, val_loader, test_loader, vocab = load_data(filepath=DATA_FILE)

    batch_x, batch_y = next(iter(train_loader))
    print(f"Batch input shape : {batch_x.shape}")
    print(f"Batch label shape : {batch_y.shape}")
    print(f"Sample labels     : {batch_y[:8].tolist()}")
    print(f"Sample token ids  : {batch_x[0, :12].tolist()}")
    print(f"Vocab size        : {len(vocab)}")
