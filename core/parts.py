import json
from core.components import Component


class SensorBasedComponent(Component):
    SENSORS = []          # diisi subclass
    MAX_Z = 3.0

    def __init__(self, name, baseline_path="artifacts/baseline.json"):
        super().__init__(name)
        with open(baseline_path) as f:
            semua = json.load(f)
        self._baseline = {s: semua[s] for s in self.SENSORS}

    def _score(self, reading):
        dev = []
        for s, b in self._baseline.items():
            z = b["arah"] * (reading[s] - b["mean"]) / b["std"]
            dev.append(max(0.0, z))
        return 1.0 - min(sum(dev) / len(dev) / self.MAX_Z, 1.0)


class HighPressureCompressor(SensorBasedComponent):
    SENSORS = ["s3", "s7", "s11", "s12"]

    def __init__(self, **kw):
        super().__init__("High Pressure Compressor", **kw)


class Fan(SensorBasedComponent):
    SENSORS = ["s8", "s13", "s15"]

    def __init__(self, **kw):
        super().__init__("Fan", **kw)


class Turbine(SensorBasedComponent):
    SENSORS = ["s4", "s20", "s21"]

    def __init__(self, **kw):
        super().__init__("Turbine", **kw)