from __future__ import annotations

import numpy as np
import bentoml
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional, Any

N_FEATURES = 36

PRETRAINED_MODEL_TAGS = {
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


@bentoml.service(name="industrial_fault_classifier_pretrained")
class FaultPretrainedService:
    def __init__(self) -> None:
        # Cargar todos los modelos preentrenados disponibles
        self.pretrained_models: Dict[str, Any] = {}
        for name, tag in PRETRAINED_MODEL_TAGS.items():
            try:
                self.pretrained_models[name] = bentoml.sklearn.load_model(tag)
            except Exception:
                pass

        if not self.pretrained_models:
            raise RuntimeError(
                "No se encontró ningún modelo en el BentoML Model Store. "
            )

        self.default_model = "rf" if "rf" in self.pretrained_models else next(iter(self.pretrained_models.keys()))

    def find_custom_model_names(self) -> List[str]:
        return sorted(set(
            m.info.metadata.get("model_name") for m in bentoml.models.list() if m.info.metadata.get("model_type") == "custom"
        ))

    @bentoml.api
    def info(self, model_type: Literal["pretrained", "custom"] = "pretrained") -> InfoResponse:
        match model_type:
            case "pretrained":
                return InfoResponse(
                    service_type="pretrained",
                    available_models=sorted(self.pretrained_models.keys()),
                    default_model=self.default_model,
                )
            
            case "custom":
                return InfoResponse(
                    service_type="custom",
                    available_models=self.find_custom_model_names(),
                    default_model="",
                )
            
            case _:
                raise ValueError(f"Tipo de modelo desconocido: '{model_type}'")

    def resolve_custom_model(self, model_name: str):
        for m in bentoml.models.list():
            if m.info.metadata.get("model_type") == "custom" and model_name == m.info.metadata.get("model_name"):
                return bentoml.sklearn.load_model(m.tag)
        
        raise ValueError(f"Modelo custom '{model_name}' no encontrado.")
    
    @bentoml.api
    def predict(self, req: PredictRequest) -> PredictResponse:
        if req.api not in ("pretrained", "custom"):
            raise ValueError(f"API incorrecta. expected='pretrained' or 'custom' got='{req.api}'")


        X = np.asarray(req.data, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != N_FEATURES:
            raise ValueError(f"Forma inválida. Se espera (batch_size,{N_FEATURES}), got={X.shape}")
        
        match req.api:
            case "custom":
                if not req.model:
                    raise ValueError("Se debe especificar el nombre del modelo custom.")
                model = self.resolve_custom_model(req.model)
            
            case "pretrained":      
                model_name = req.model or self.default_model
                if model_name not in self.pretrained_models:
                    raise ValueError(f"Modelo '{model_name}' no disponible. Disponibles: {sorted(self.pretrained_models.keys())}")
                
                model = self.pretrained_models[model_name]

            case _:
                raise ValueError(f"API incorrecta. expected='pretrained' or 'custom' got='{req.api}'")
        
        print(f"Using model: {model}")
        preds = model.predict(X)

        return PredictResponse(
            service_type=req.api,
            model=req.model,
            pred=preds.tolist() if hasattr(preds, "tolist") else list(preds),
        )
