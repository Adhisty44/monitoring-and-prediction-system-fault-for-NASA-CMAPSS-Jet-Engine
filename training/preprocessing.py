"""Preprocessing data C-MAPSS: baca data, hitung RUL, buat label."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATASET_PATH = ("dataset/train_FD001.txt")

COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
FEATURES = ["s2", "s3", "s4", "s7", "s8", "s9", "s11",
            "s12", "s13", "s14", "s15", "s17", "s20", "s21"]

NORMAL_MIN_RUL = 80   
WARNING_MIN_RUL = 30   
TEST_SIZE = 0.20
RANDOM_STATE = 42


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    n_units: int

    def __len__(self):
        return len(self.y)


class TurbofanPreprocessor:
    def __init__(self, dataset_path=DATASET_PATH, features=None,
                 normal_min_rul=NORMAL_MIN_RUL, warning_min_rul=WARNING_MIN_RUL,
                 test_size=TEST_SIZE, random_state=RANDOM_STATE):
        self._dataset_path = Path(dataset_path)
        self._features = list(features) if features else list(FEATURES)
        self._normal_min_rul = normal_min_rul
        self._warning_min_rul = warning_min_rul
        self._test_size = test_size
        self._random_state = random_state

    @property
    def features(self):
        return list(self._features)

    def load(self) -> pd.DataFrame:
        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Dataset tidak ditemukan: {self._dataset_path}")
        df = pd.read_csv(self._dataset_path, sep=r"\s+", header=None, names=COLUMNS)
        df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
        return df

    def create_labels(self, rul) -> np.ndarray:
        labels = np.where(rul > self._normal_min_rul, 0,
                          np.where(rul > self._warning_min_rul, 1, 2))
        return labels.astype(int)

    def prepare(self) -> Dataset:
        df = self.load()
        return Dataset(
            X=df[self._features].to_numpy(dtype=float),
            y=self.create_labels(df["RUL"]),
            groups=df["unit"].to_numpy(),
            n_units=int(df["unit"].nunique()),
        )

    def split(self, dataset: Dataset):
        """Pisah per mesin (belum dipakai di alur training sekarang)."""
        splitter = GroupShuffleSplit(n_splits=1, test_size=self._test_size,
                                     random_state=self._random_state)
        return next(splitter.split(dataset.X, dataset.y, groups=dataset.groups))