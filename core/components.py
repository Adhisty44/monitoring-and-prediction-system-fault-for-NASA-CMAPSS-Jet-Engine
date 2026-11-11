from abc import ABC, abstractmethod
from dataclasses import dataclass


from core.enums import HealthStatus

@dataclass(frozen=True)
class Component_health:
    name: str
    health_score: float
    status: HealthStatus

class Component(ABC):
    WARN_AT = 0.7
    CRIT_AT = 0.4

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name
    
    @abstractmethod
    def _score(self, values: dict) -> float:
        pass   

    def update_health(self, health_score: float):
        if health_score >= self.WARN_AT:
            status = HealthStatus.NORMAL
        elif health_score >= self.CRIT_AT:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.CRITICAL
        return status

    def evaluate(self, values: dict) -> Component_health:
        hasil_health = self._score(values)
        batas_skor = max(0.0, min(1.0, hasil_health))
        return Component_health(self.name, batas_skor, self.update_health(batas_skor))