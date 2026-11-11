from abc import ABC, abstractmethod
from dataclasses import dataclass
from core.enums import HealthStatus
import joblib
import numpy as np
import pandas as pd

class Experiment(ABC):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def predict(self, features):
        pass

class ClassificationModels(Experiment):
    FEATURES = ["s2", "s3", "s4", "s7", "s8", "s9", "s11",
                "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
    
    def __init__(self, name: str, model_path: str, scaler_path: str):
        super().__init__(name)
        self._model = joblib.load(model_path)
        self._scaler = joblib.load(scaler_path)


    def predict(self, features):
        if isinstance(features, pd.DataFrame):
            features = features.values
        elif isinstance(features, list):
            features = np.array(features)

        features = np.atleast_2d(np.asarray(features, dtype=float))
        features = self._scaler.transform(features)
        hasil = self._model.predict(features)
        return [HealthStatus(int(h)) for h in hasil]
    