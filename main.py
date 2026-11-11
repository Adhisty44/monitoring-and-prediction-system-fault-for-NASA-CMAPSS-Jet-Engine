import time
from framework.experiments import ClassificationModels
from core.parts import HighPressureCompressor, Fan, Turbine
from plugins.ReplayStream import ReplayStream

exp = ClassificationModels("classification", "artifacts/svm_status.joblib", "artifacts/scaler.joblib")
komponen = [HighPressureCompressor(), Fan(), Turbine()]
stream = ReplayStream("dataset/test_FD001.txt", unit=3)

while True:
    row = stream.read()
    if row is None:
        break
    status = exp.predict(row[ClassificationModels.FEATURES].to_frame().T)[0]
    hasil = [f"{k.name}: {k.evaluate(row).health_score:.2f}" for k in komponen]
    print(f"siklus {int(row['cycle']):3d} | mesin: {status.label:8s} | " + " | ".join(hasil), flush=True)
    time.sleep(0.01)