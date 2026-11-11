"""Kelas dasar trainer. Alur training ditulis sekali di sini."""

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
from sklearn.preprocessing import StandardScaler

from training.preprocessing import TurbofanPreprocessor

ARTIFACT_DIR = ("artifacts/")
RANDOM_STATE = 42


class BaseTrainer(ABC):
    def __init__(self, preprocessor=None, artifact_dir=ARTIFACT_DIR):
        self._preprocessor = preprocessor or TurbofanPreprocessor()
        self._artifact_dir = Path(artifact_dir)

    @property
    @abstractmethod
    def name(self) -> str:
        """Nama pendek model, dipakai untuk nama file (mis. 'svm')."""

    @abstractmethod
    def build_model(self):
        """Mengembalikan model sklearn baru (belum di-fit)."""

    def build_scaler(self):
        return StandardScaler()

    @property
    def model_path(self) -> Path:
        return self._artifact_dir / f"{self.name}_status.joblib"

    @property
    def scaler_path(self) -> Path:
        return self._artifact_dir / "scaler.joblib"

    def run(self) -> None:
        data = self._preprocessor.prepare()

        scaler = self.build_scaler()
        X_scaled = scaler.fit_transform(data.X)

        model = self.build_model()
        model.fit(X_scaled, data.y)

        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        joblib.dump(scaler, self.scaler_path)

        print(f"Model {self.name} disimpan di {self.model_path}")