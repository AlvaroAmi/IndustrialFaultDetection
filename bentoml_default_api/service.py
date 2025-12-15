from __future__ import annotations

import numpy as np
import bentoml
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


SERVICE_TYPE = "pretrained"  # Distinguir esta API de la de reentreno/actualización
N_FEATURES = 36

MODEL_TAGS: Dict[str, str] = {
    "logreg": "fault_logreg:latest",
    "svm_rbf": "fault_svm_rbf:latest",
    "knn": "fault_knn:latest",
    "mlp": "fault_mlp:latest",
    "rf": "fault_rf:latest",
    "gb": "fault_gb:latest",
    "hgb": "fault_hgb:latest",
    "xgb": "fault_xgb:latest",
}


class PredictRequest(BaseModel):
    api: str = Field(description="Debe ser 'pretrained' para usar esta API")
    model: Optional[str] = Field(default=None)
    data: List[List[float]] = Field(
        examples=[[[0.0] * N_FEATURES]],
    )


class PredictResponse(BaseModel):
    service_type: str
    model: str
    pred: List[int] 


class InfoResponse(BaseModel):
    service_type: str
    available_models: List[str]
    default_model: str
    n_features: int


@bentoml.service(name="industrial_fault_classifier_pretrained")
class FaultPretrainedService:
    def __init__(self) -> None:
        # Cargar todos los modelos disponibles
        self.models: Dict[str, Any] = {}
        for name, tag in MODEL_TAGS.items():
            try:
                self.models[name] = bentoml.sklearn.load_model(tag)
            except Exception:
                pass

        if not self.models:
            raise RuntimeError(
                "No se encontró ningún modelo en el BentoML Model Store. "
            )

        self.default_model = "rf" if "rf" in self.models else next(iter(self.models.keys()))

    @bentoml.api
    def info(self) -> InfoResponse:
        return InfoResponse(
            service_type=SERVICE_TYPE,
            available_models=sorted(self.models.keys()),
            default_model=self.default_model,
            n_features=N_FEATURES,
        )

    @bentoml.api
    def predict(self, req: PredictRequest) -> PredictResponse:
        if req.api != SERVICE_TYPE:
            raise ValueError(f"API incorrecta. expected='{SERVICE_TYPE}' got='{req.api}'")

        model_name = req.model or self.default_model
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' no disponible. Disponibles: {sorted(self.models.keys())}")

        X = np.asarray(req.data, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != N_FEATURES:
            raise ValueError(f"Forma inválida. Se espera (batch_size,{N_FEATURES}), got={X.shape}")

        model = self.models[model_name]
        preds = model.predict(X)

        return PredictResponse(
            service_type=SERVICE_TYPE,
            model=model_name,
            pred=preds.tolist() if hasattr(preds, "tolist") else list(preds),
        )
