# Explorar un run en Colab, desde su link de Drive

`notebooks/explorar_modelo_colab.ipynb` abre el explorador del modelo de mundo (`explorar_modelo.py`) en Colab con el run de una carpeta de Drive. No hace falta montar Drive ni agregar un acceso directo, aunque la carpeta sea de otra cuenta: basta con que la tuya tenga acceso.

## Qué hace, en orden

1. **Variables** (celda 1): `RAMA` (la rama del repo con el código), `CARPETA_DRIVE` (el link o el id de la carpeta) y, si la carpeta tiene varios runs, `NOMBRE_RUN`.
2. **Permiso**: `auth.authenticate_user()` (Colab pide aceptar una vez).
3. **Instala**: clona la rama y arma un entorno con Python 3.11 (el que pide el repo) con Gradio.
4. **Baja el run** con `drive_run.py`: busca dentro de la carpeta la que tiene `latest.pt` y baja solo `latest.pt`, `.hydra/`, `status.json` y `metrics.jsonl` (~130 MB). Se salta los eventos de TensorBoard (cientos de MB) y el buffer `replay/` (gigas). Lo que ya está en `/content/run` no se baja de nuevo.
5. **Comprueba** que carga (recorrido corto sin página) y muestra las imágenes.
6. **Métricas** del run (galletas, botonazos, casillas, choques).
7. **Abre el explorador** en un marco.

## Lo que hay que saber

- **La rama tiene que estar subida a GitHub** y contener `explorar_modelo.py`, `drive_run.py` y un código que sepa construir el ambiente del run. Si no, la celda 5 lo dice (por ejemplo, «no pude construir el ambiente de este run (env.obstacles=...)»).
- **Sirve CPU.** No necesita GPU.
- **Fuera de Colab:** `python explorar_modelo.py --drive <link> --destino run` hace lo mismo (necesita credenciales de Google: `gcloud auth application-default login --scopes=https://www.googleapis.com/auth/drive.readonly,https://www.googleapis.com/auth/cloud-platform`). Con el run ya en disco: `python explorar_modelo.py --logdir <carpeta>` (puede ser la carpeta que contiene los runs).
- **El run tiene que tener `latest.pt`**: el explorador no puede cargar un run que solo tenga métricas.

## Compatibilidad entre versiones del código

El modelo (`dreamer.py`) tiene que construir los mismos parámetros con los que se guardó el checkpoint. Los runs de `main` con `model.plan2explore=False` (con el explorador Plan2Explore apagado) tienen **exactamente los mismos 184 tensores** que los de esta rama. Si un run se entrenó con `plan2explore=True`, el checkpoint trae módulos extra que este código no tiene y no carga. El ambiente, en cambio, sí tiene que ser el del entrenamiento: ver [`obstaculos.md`](obstaculos.md) para los modos de `env.obstacles`.
