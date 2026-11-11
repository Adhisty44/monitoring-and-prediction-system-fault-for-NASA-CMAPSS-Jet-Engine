import json
import pandas as pd

F = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
cols = ["unit","cycle","op1","op2","op3"] + [f"s{i}" for i in range(1, 22)]

df = pd.read_csv("dataset/train_FD001.txt", sep=r"\s+", header=None, names=cols)
df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
sehat = df[df["RUL"] > 100]

out = {}
for s in F:
    c = df[s].corr(df["RUL"])
    out[s] = {"mean": float(sehat[s].mean()),
              "std": float(sehat[s].std()),
              "arah": -1 if c > 0 else 1}
    print(s, "corr:", round(c, 3), "arah:", out[s]["arah"])

with open("artifacts/baseline.json", "w") as f:
    json.dump(out, f, indent=2)
print("tersimpan: artifacts/baseline.json")