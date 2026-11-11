import pandas as pd
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


class ReplayStream:
    """Memutar data satu mesin, satu siklus per pemanggilan read()."""

    def __init__(self, path: str, unit: int):
        df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
        self._data = df[df["unit"] == unit].reset_index(drop=True)
        if self._data.empty:
            raise ValueError(f"Mesin {unit} tidak ada di {path}")
        self._unit = unit
        self._pos = 0

    @property
    def unit(self):
        return self._unit

    def read(self):
        """Mengembalikan satu baris (pd.Series), atau None kalau data habis."""
        if self._pos >= len(self._data):
            return None
        row = self._data.iloc[self._pos]
        self._pos += 1
        return row

    def reset(self):
        self._pos = 0