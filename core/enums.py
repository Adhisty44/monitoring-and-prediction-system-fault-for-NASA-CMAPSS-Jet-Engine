from enum import Enum


class HealthStatus(Enum):
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2

    @property
    def label(self):
        return ("normal", "waspada", "kritis")[self.value]
    
    @classmethod
    def from_label(cls, label) -> "HealthStatus":
        return cls(int(label))

