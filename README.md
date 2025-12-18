Para crear los modelos preentrenados (ya están creados) se ejecuta el archivo train_models.ipynb, el cual entrena y guarda los modelos en el directorio: `artifacts/models_default`. Una vez guardados los modelos, hay que importarlos y guardarlos en bentoml. Para ello, se utiliza el script `bentoml/save_models.py`.

Para utilizar la interfaz de streamlit (la parte de guardado y ejecución de modelos) hace falta tener el servicio de bentoml ejecutando en el puerto 3000. Se hace ejecutando el comando: 
```
bentoml serve bentoml_api.service:FaultPredictionService --port 3000
```

Para iniciar la interfaz de streamlit:

```
python -m streamlit run streamlit_app.py
```

Utilizando la pestaña `Modelado` se pueden entrenar y evaluar distintos modelos con sus respectivos parámetros. Al entrenar un modelo, aparecerá el botón `Guardar modelo entrenado`, para guardar el modelo en bentoml. Después, en la pestaña `Predicción`, se pueden hacer predicciones tanto con los modelos preentrenados como con los modelos recién entrenados, haciendo click en la opción `Custom` y seleccionando el modelo entrenado.