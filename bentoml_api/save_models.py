import joblib
from pathlib import Path
import bentoml

MODELS_DIR = Path("artifacts/models_default")

# Mapea nombre lógico -> fichero
MODEL_FILES = {
    "logreg": MODELS_DIR / "logreg_best.joblib",
    "svm_rbf": MODELS_DIR / "svm_rbf_best.joblib",
    "knn": MODELS_DIR / "knn_best.joblib",
    "mlp": MODELS_DIR / "mlp_best.joblib",
    "rf": MODELS_DIR / "rf_best.joblib",
    "gb": MODELS_DIR / "gb_best.joblib",
    "hgb": MODELS_DIR / "hgb_best.joblib",
    "xgb": MODELS_DIR / "xgb_best.joblib",
}

for name, path in MODEL_FILES.items():
    if not path.exists():
        print(f"Skipping {name}: {path} no existe")
        continue

    model = joblib.load(path)

    bento_model = bentoml.sklearn.save_model(
        f"fault_{name}",
        model,
        metadata={"model_name": name, "model_type": "pretrained"},
    )
    print("Guardado en BentoML:", bento_model.tag)
