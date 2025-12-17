import pathlib
import json
from typing import List
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import requests

DATA_PATH = pathlib.Path("data/Industrial_fault_detection.csv")
PROCESS_VARS: List[str] = [
    "Temperature",
    "Vibration",
    "Pressure",
    "Flow_Rate",
    "Current",
    "Voltage",]
BENTO_URL = "http://127.0.0.1:3000"   # API pretrained
FEATURES: List[str] = [
    "Temperature","Vibration","Pressure","Flow_Rate","Current","Voltage",
    "FFT_Temp_0","FFT_Vib_0","FFT_Pres_0",
    "FFT_Temp_1","FFT_Vib_1","FFT_Pres_1",
    "FFT_Temp_2","FFT_Vib_2","FFT_Pres_2",
    "FFT_Temp_3","FFT_Vib_3","FFT_Pres_3",
    "FFT_Temp_4","FFT_Vib_4","FFT_Pres_4",
    "FFT_Temp_5","FFT_Vib_5","FFT_Pres_5",
    "FFT_Temp_6","FFT_Vib_6","FFT_Pres_6",
    "FFT_Temp_7","FFT_Vib_7","FFT_Pres_7",
    "FFT_Temp_8","FFT_Vib_8","FFT_Pres_8",
    "FFT_Temp_9","FFT_Vib_9","FFT_Pres_9",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    cols = PROCESS_VARS + [c for c in df.columns if c.startswith("FFT_")] + ["Fault_Type"]
    return df[cols] if set(cols) == set(df.columns) else df


def render_exploracion(df: pd.DataFrame) -> None:
    st.subheader("Exploración de datos")
    st.caption(f"Filas: {len(df):,} | Columnas: {df.shape[1]}")

    st.sidebar.header("Filtros")
    fault_types = sorted(df["Fault_Type"].unique())
    selected_faults = st.sidebar.multiselect(
        "Fault_Type", fault_types, default=fault_types, help="Selecciona clases a analizar")
    sample_n = st.sidebar.slider(
        "Muestra aleatoria",
        min_value=200,
        max_value=len(df),
        value=min(800, len(df)),
        step=100,)
    
    df_filtered = df[df["Fault_Type"].isin(selected_faults)]
    df_sample = (
        df_filtered.sample(n=min(sample_n, len(df_filtered)), random_state=22)
        if len(df_filtered) > 0
        else df_filtered)

    st.markdown("### 1) Resumen general")
    col1, col2, col3 = st.columns(3)
    col1.metric("Clases seleccionadas", len(selected_faults))
    col2.metric("Muestra usada", len(df_sample))
    col3.metric("Total dataset", len(df))

    st.markdown("**Distribución de clases (muestra filtrada)**")
    class_counts = df_sample["Fault_Type"].value_counts().sort_index()
    fig_cls = px.bar(
        class_counts,
        labels={"index": "Fault_Type", "value": "Conteo"},
        text_auto=True,
        title="Conteo por clase",)
    st.plotly_chart(fig_cls, width="stretch")

    st.markdown("**Estadísticas descriptivas (variables de proceso)**")
    st.dataframe(df_sample[PROCESS_VARS].describe().T, width="stretch")

    st.markdown("### 2) Correlaciones")
    corr_vars = st.multiselect(
        "Variables a correlacionar",
        PROCESS_VARS,
        default=PROCESS_VARS,)
    
    if len(corr_vars) >= 2:
        corr = df_sample[corr_vars].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu",
            origin="lower",
            title="Matriz de correlación",
        )
        st.plotly_chart(fig_corr, width="stretch")
    else:
        st.info("Selecciona al menos dos variables para ver correlaciones.")

    st.markdown("### 3) Distribuciones univariantes")
    dist_var = st.selectbox("Variable", PROCESS_VARS)
    fig_dist = px.histogram(
        df_sample,
        x=dist_var,
        color="Fault_Type",
        marginal="box",
        nbins=25,
        opacity=0.7,
        title=f"Distribución de {dist_var} por Fault_Type",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_dist, width="stretch")

    st.markdown("### 4) Relación bivariada")
    x_var = st.selectbox("Eje X", PROCESS_VARS, index=0, key="x_var")
    y_var = st.selectbox("Eje Y", PROCESS_VARS, index=1, key="y_var")
    df_plot = df_sample.assign(Fault_Type_str=df_sample["Fault_Type"].astype(str))
    fig_scatter = px.scatter(
        df_plot,
        x=x_var,
        y=y_var,
        color="Fault_Type_str",
        opacity=0.7,
        title=f"{x_var} vs {y_var} por Fault_Type",
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={"Fault_Type_str": "Fault_Type"},
    )
    st.plotly_chart(fig_scatter, width="stretch")

    st.markdown("### 5) Visualización de FFT")
    fft_cols = [c for c in df.columns if c.startswith("FFT_")]
    if fft_cols:
        fft_to_plot = st.selectbox("FFT a mostrar", fft_cols)
        fig_fft = px.histogram(
            df_sample,
            x=fft_to_plot,
            color="Fault_Type",
            nbins=30,
            opacity=0.7,
            title=f"Distribución de {fft_to_plot} por Fault_Type",
            color_discrete_sequence=px.colors.qualitative.Dark24,
        )
        st.plotly_chart(fig_fft, width="stretch")
    else:
        st.info("No se encontraron columnas FFT_ en el dataset.")


def render_modelado(df: pd.DataFrame) -> None:
    st.subheader("Entrenamiento y evaluación de modelos")
    st.caption("Clasificación de Fault_Type.")

    X = df.drop(columns=["Fault_Type"])
    y = df["Fault_Type"]

    col_conf = st.columns(5)
    test_size = col_conf[0].slider("Proporción test", 0.1, 0.5, 0.2, 0.05)
    random_state = int(col_conf[1].number_input("Random state", min_value=0, max_value=9999, value=67, step=1))
    stratify = col_conf[2].checkbox("Estratificar", value=True)
    use_pca = col_conf[3].checkbox("Aplicar PCA", value=False)
    if use_pca:
        n_components = col_conf[4].number_input("Nº componentes PCA", min_value=1, max_value=36, value=5, step=1)
    else:
        n_components = None 

    model_name = st.selectbox(
        "Modelo",
        [
            "Regresión logística",
            "SVM (RBF)",
            "k-NN",
            "MLP",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
            "LightGBM",
        ],
    )

    def add_pca_to_pipeline(steps):
        if n_components is not None:
            return [("scale", StandardScaler()), ("pca", PCA(n_components=n_components))] + steps[1:]
        else:
            return steps

    if model_name == "Regresión logística":
        C = st.select_slider("C (regularización)", [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0], value=1.0)
        steps = [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=600, n_jobs=None)),
        ]
        if n_components is not None:
            steps.insert(1, ("pca", PCA(n_components=n_components)))
        model = Pipeline(steps)
    elif model_name == "SVM (RBF)":
        C = st.select_slider("C", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0], value=1.0)
        gamma = st.select_slider("gamma", ["scale", "auto", 0.01, 0.05, 0.1, 0.2], value="scale")
        steps = [
            ("scale", StandardScaler()),
            ("clf", SVC(C=C, gamma=gamma)),
        ]
        if n_components is not None:
            steps.insert(1, ("pca", PCA(n_components=n_components)))
        model = Pipeline(steps)
    elif model_name == "k-NN":
        n_neighbors = st.slider("Vecinos", min_value=3, max_value=25, value=7, step=2)
        weights = st.selectbox("weights", ["uniform", "distance"], index=0)
        steps = [
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)),
        ]
        if n_components is not None:
            steps.insert(1, ("pca", PCA(n_components=n_components)))
        model = Pipeline(steps)
    elif model_name == "MLP":
        hidden = st.selectbox("Capas ocultas", [(50,), (100,), (100, 50), (200, 100)], index=1)
        alpha = st.select_slider("alpha (L2)", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.0005)
        activation = st.selectbox("Activación", ["relu", "tanh"], index=0)
        steps = [
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=hidden, activation=activation, alpha=alpha, max_iter=400, random_state=random_state)),
        ]
        if n_components is not None:
            steps.insert(1, ("pca", PCA(n_components=n_components)))
        model = Pipeline(steps)
    elif model_name == "Random Forest":
        n_estimators = st.slider("Árboles", min_value=100, max_value=600, value=300, step=50)
        max_depth = st.slider("Profundidad máxima (None = sin límite)", min_value=2, max_value=20, value=10, step=1)
        max_features = st.selectbox("max_features", ["sqrt", "log2"])
        if n_components is not None:
            steps = [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=n_components)),
                ("clf", RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=None if max_depth == 20 else max_depth,
                    max_features=max_features,
                    random_state=random_state,
                    n_jobs=-1,
                )),
            ]
            model = Pipeline(steps)
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None if max_depth == 20 else max_depth,
                max_features=max_features,
                random_state=random_state,
                n_jobs=-1,
            )
    else:
        if model_name == "Gradient Boosting":
            n_estimators = st.slider("Iteraciones (estimadores)", min_value=50, max_value=400, value=200, step=50)
            learning_rate = st.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.15, 0.2, 0.3], value=0.1)
            max_depth = st.slider("Profundidad de árboles base", min_value=1, max_value=5, value=3, step=1)
            if n_components is not None:
                steps = [
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_components=n_components)),
                    ("clf", GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=random_state,
                    )),
                ]
                model = Pipeline(steps)
            else:
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=random_state,
                )
        elif model_name == "XGBoost":
            try:
                from xgboost import XGBClassifier
            except ImportError:
                st.error("Instala xgboost: pip install xgboost")
                return
            n_estimators = st.slider("Estimadores", min_value=100, max_value=800, value=300, step=50)
            learning_rate = st.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
            max_depth = st.slider("max_depth", min_value=2, max_value=10, value=5, step=1)
            subsample = st.select_slider("subsample", [0.6, 0.7, 0.8, 0.9, 1.0], value=0.8)
            colsample = st.select_slider("colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1.0], value=0.8)
            if n_components is not None:
                steps = [
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_components=n_components)),
                    ("clf", XGBClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        subsample=subsample,
                        colsample_bytree=colsample,
                        random_state=random_state,
                        objective="multi:softmax",
                    )),
                ]
                model = Pipeline(steps)
            else:
                model = XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    colsample_bytree=colsample,
                    random_state=random_state,
                    objective="multi:softmax",
                )
        elif model_name == "LightGBM":
            try:
                from lightgbm import LGBMClassifier
            except ImportError:
                st.error("Instala lightgbm: pip install lightgbm")
                return
            n_estimators = st.slider("Estimadores", min_value=100, max_value=800, value=300, step=50)
            learning_rate = st.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.15, 0.2], value=0.1)
            num_leaves = st.slider("num_leaves", min_value=15, max_value=255, value=63, step=8)
            max_depth = st.slider("max_depth (-1 sin límite)", min_value=-1, max_value=20, value=-1, step=1)
            if n_components is not None:
                steps = [
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_components=n_components)),
                    ("clf", LGBMClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        num_leaves=num_leaves,
                        max_depth=max_depth,
                        random_state=random_state,
                    )),
                ]
                model = Pipeline(steps)
            else:
                model = LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    max_depth=max_depth,
                    random_state=random_state,
                )
        else:
            st.error("Modelo no soportado")
            return

    button_container = st.container(horizontal=True)
    
    if button_container.button("Entrenar y evaluar", type="primary"):
        X_train, X_test, y_train, y_test = train_test_split(
            X,y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None,)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_w = f1_score(y_test, y_pred, average="weighted")

        # Store results in session state
        st.session_state.model_results = {
            "model": model,
            "model_name": model_name,
            "acc": acc,
            "f1_w": f1_w,
            "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            "y_test": y_test,
            "y_pred": y_pred,
            "X": X,
        }

    if "model_results" in st.session_state:
        if button_container.button("Guardar modelo entrenado", type="secondary"):
            import joblib
            from pathlib import Path
            import bentoml
            import unicodedata

            MODELS_DIR = Path("artifacts/models_custom")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model_name_normalized = unicodedata.normalize('NFKD', st.session_state.model_results["model_name"].lower()).encode('ASCII', 'ignore').decode('ASCII')\
                .replace(' ', '_').replace('(', '').replace(')', '')
            
            print(model_name_normalized)
            model_file = MODELS_DIR / f"{model_name_normalized}_custom.joblib"
            joblib.dump(st.session_state.model_results["model"], model_file)

            bento_model = bentoml.sklearn.save_model(
                f"fault_{model_name_normalized}_custom",
                st.session_state.model_results["model"],
                metadata={"model_name": model_name_normalized, "model_type": "custom"}
            )

            st.success(f"Modelo guardado en BentoML: {st.session_state.model_results['model_name']} como {bento_model.tag}")

    # Display results if they exist in session state
    if "model_results" in st.session_state:
        results = st.session_state.model_results
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Accuracy", f"{results['acc']:.3f}")
        col_m2.metric("F1 ponderado", f"{results['f1_w']:.3f}")

        report_df = pd.DataFrame(results["report"]).T.round(3)
        st.markdown("**Métricas por clase**")
        st.dataframe(report_df, width="stretch")

        labels = sorted(results["y_test"].unique())
        cm = confusion_matrix(results["y_test"], results["y_pred"], labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"Real {c}" for c in labels], columns=[f"Pred {c}" for c in labels])
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            labels={"color": "Conteo"},
            title="Matriz de confusión",
        )
        st.plotly_chart(fig_cm, width="stretch")

        if results["model_name"] in {"Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"}:
            importances = getattr(results["model"], "feature_importances_", None)
            if importances is not None:
                imp_df = (
                    pd.DataFrame({"feature": results["X"].columns, "importance": importances})
                    .sort_values("importance", ascending=False)
                    .head(10)
                )
                fig_imp = px.bar(
                    imp_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top 10 importancias de variables",
                )
                st.plotly_chart(fig_imp, width="stretch")

# Obtiene la lista de modelos disponibles desde la API BentoML
@st.cache_data
def fetch_available_pretrained_models() -> list[str]:
    r = requests.post(f"{BENTO_URL}/info", json={"model_type": "pretrained"}, timeout=10)
    r.raise_for_status()
    info = r.json()
    return info.get("available_models", []), info.get("default_model", "")

def fetch_available_custom_models() -> list[str]:
    r = requests.post(f"{BENTO_URL}/info", json={"model_type": "custom"}, timeout=10)
    r.raise_for_status()
    info = r.json()
    return info.get("available_models", []), info.get("default_model", "")

def render_predict() -> None:
    st.header("Prediccion de fallos industriales")
    st.markdown(""" 
    En esta página puede realizarse la predicción de un nuevo conjuto de datos, bien haciendo 
    uso de modelos de machine learning preentrenados o los últimos modelos particulares entrenados. Es 
    necesario proporcionar las 36 caracteristicas (temperatura, vibracion, etc.).
    """)
    st.info("Es preciso asegurar que el servidor BentoML esta siendo ejecutado en el puerto 3000.")
    
    

    with st.container():
        st.subheader("Configuracion del Modelo")
        st.markdown(""" 
        Pretrained se refiere a los modelos de machine learning preentrenados que se ofrecen por 
        defecto para realizar las predicciones; asimismo, custom se corresponden con los últimos 
        modelos personalmente entrenados por el usuario. 
        """)

        model_type = st.radio("Selecciona modelo y valores para predecir:", 
                            options=["Pretrained", "Custom"], 
                            index=0, 
                            key="predict_mode", 
                            horizontal=True)
        
            
        # Traer modelos disponibles desde la API
        try:
            available_models, default_model = fetch_available_pretrained_models() if model_type == "Pretrained" else fetch_available_custom_models()
        except requests.RequestException as e:
            st.error(f"No se pudo consultar /info en la API: {e}")
            return

        default_index = available_models.index(default_model) if default_model in available_models else 0
        model_name = st.selectbox(f"Modelo {model_type}", available_models, index=default_index, placeholder="No hay modelos disponibles")

        if not model_name:
            return
        

    with st.container():
            st.subheader("Entrada de Datos")
            input_method = st.radio(
                "Metodo de entrada:",
                ("Subir CSV", "Subir JSON", "Manual"),
                index=0,
            )

            data = None
            if input_method == "Manual":
                st.markdown("### Introduccion Manual")
                st.caption("Ingresa valores para cada sensor (usa decimales si es necesario).")
                cols = st.columns(6)
                values = []
                for i, feat in enumerate(FEATURES):
                    with cols[i % 6]:
                        values.append(st.number_input(feat, value=0.0, format="%.6f", step=0.01))
                data = values
            elif input_method == "Subir CSV":
                st.markdown("### Subir Archivo CSV")
                st.caption("El archivo debe tener exactamente 36 columnas (una fila de datos).")
                uploaded_csv = st.file_uploader("Selecciona un archivo CSV", type="csv")
                if uploaded_csv is not None:
                    df = pd.read_csv(uploaded_csv)
                    if len(df.columns) == 36 and len(df) >= 1:
                        data = df.iloc[0].values.tolist()
                        st.success("Datos cargados correctamente del CSV.")
                    else:
                        st.error("El CSV debe tener 36 columnas y al menos 1 fila.")
            elif input_method == "Subir JSON":
                st.markdown("### Subir Archivo JSON")
                st.caption("El JSON debe ser un objeto con 36 claves (ej: {\"Temperature\": 25.5, ...}).")
                uploaded_json = st.file_uploader("Selecciona un archivo JSON", type="json")
                if uploaded_json is not None:
                    json_data = json.load(uploaded_json)
                    if isinstance(json_data, dict) and len(json_data) == 36:
                        data = [json_data.get(feature, 0.0) for feature in FEATURES]
                        st.success("Datos cargados correctamente del JSON.")
                    else:
                        st.error("El JSON debe ser un objeto con exactamente 36 claves.")

    with st.container():
        st.subheader("Realizar Prediccion")
        if st.button("Predecir Fallo", type="primary", key="predict_button", use_container_width=True):
            if data and len(data) == 36:
                with st.spinner("Procesando prediccion..."):
                    payload = {
                        "api": model_type.lower(),
                        "model": model_name,
                        "data": [data],
                    }
                    try:
                        r = requests.post(f"{BENTO_URL}/predict", json={"req": payload}, timeout=20)
                        r.raise_for_status()
                        out = r.json()
                        pred = out.get("pred", [None])[0]
                        st.success(f"Prediccion exitosa: Tipo de fallo {pred}")
                        with st.expander("Ver detalles de la respuesta"):
                            st.json(out)
                    except requests.HTTPError as e:
                        st.error(f"Error HTTP {r.status_code}: {r.text}")
                    except requests.RequestException as e:
                        st.error(f"Error de conexion: {e}")
            else:
                st.warning("Proporciona datos validos para las 36 caracteristicas antes de predecir.")


def main() -> None:
    st.set_page_config(page_title="Industrial Fault Detection", layout="wide")
    st.title("Exploración interactiva")

    df = st.cache_data(load_data)()

    seccion = st.sidebar.radio(
        "Secciones",
        ["Exploración de datos", "Modelado", "Predicción"],
    )

    if seccion == "Exploración de datos":
        render_exploracion(df)
    elif seccion == "Modelado":
        render_modelado(df)
    elif seccion == "Predicción":
        render_predict()


if __name__ == "__main__":
    main()
