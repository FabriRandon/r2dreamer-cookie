# Experimentos de la presentación

Notebooks de Colab de los experimentos que se mostraron en la presentación del ambiente de la galleta (mapa de pasillos cortos, `hall5`). Son copias de los que están en `notebooks/experimentos-semana-del-15/` y `notebooks/experimentos-semana-del-22/`.

En cada experimento hay dos casos que solo difieren en quién junta los datos:

- **Normal:** el actor de Dreamer juega y junta los datos.
- **Random:** los datos los junta un agente que elige al azar; el actor aprende solo de esos datos.

| Experimento | Lo que ve el agente | Normal | Random | Pasos usados |
|---|---|---|---|---|
| 1 | Mapa completo | `experimento_1_vision_completa_normal.ipynb` | `experimento_1_vision_completa_random.ipynb` | ~200.000 / 750.000 |
| 2 | Vista de 7×7 | `experimento_2_vista_7x7_normal.ipynb` | `experimento_2_vista_7x7_random.ipynb` | 500.000 / 750.000 |
| 3 | Vista de 5×5 | `experimento_3_vista_5x5_normal.ipynb` | `experimento_3_vista_5x5_random.ipynb` | 500.000 / 1.000.000 |
| 4 | Vista de 3×3 | `experimento_4_vista_3x3_normal.ipynb` | `experimento_4_vista_3x3_random.ipynb` | 500.000 / 1.000.000 |
| Extra | Vista de 5×5, repitiendo las experiencias con galleta y con 10 veces más entropía | `experimento_extra_vista_5x5_normal_repetir_galletas_entropia_10x.ipynb` | El random del experimento 3 | 500.000 |

`videos_presentacion.ipynb` muestra los videos del actor de los ocho runs, normal al lado de random.

## Cómo correr un experimento

1. Abre el notebook en Colab: `https://colab.research.google.com/github/FabriRandon/r2dreamer-cookie/blob/main/EXPERIMENTOS%20PRESENTACI%C3%93N/<archivo>.ipynb`.
2. Elige un entorno con GPU (Entorno de ejecución → Cambiar tipo de entorno). Los runs se hicieron con una L4.
3. **Primera vez:** corre las celdas en orden, de la 1 a la 11. La 8 es una prueba corta que comprueba que todo funcione y estima cuánto va a tardar.
4. **Para seguir otro día:** corre las celdas 1 a 6 y después la 10. La 7 dice en qué paso quedó y la 9 abre TensorBoard.
5. **Para detenerlo:** usa el botón de detener de la celda 10 y espera el mensaje `Listo, quedó guardado` antes de cerrar la pestaña.

La celda 6 define cuántos pasos entrena (`PASOS`). En tres notebooks quedó guardado un valor menor que el que se usó en la presentación: cámbialo antes de entrenar si quieres llegar a lo mismo.

| Notebook | `PASOS` guardado | Usado en la presentación |
|---|---|---|
| `experimento_1_vision_completa_random.ipynb` | 500.000 | 750.000 |
| `experimento_2_vista_7x7_random.ipynb` | 500.000 | 750.000 |
| `experimento_4_vista_3x3_random.ipynb` | 750.000 | 1.000.000 |

## Dónde quedan los resultados

Cada notebook guarda el modelo, el buffer y las métricas en Google Drive, en la carpeta que dice `LOGDIR` (celda 6), dentro de `galleta/semana8`, `galleta/semana15` o `galleta/semana22`. Si esa carpeta ya tiene un run, el notebook lo **retoma** desde donde quedó. Para empezar uno nuevo desde cero, cambia `LOGDIR` a otra carpeta.

Para ver si el actor aprendió, mira `episode/eval_score` en TensorBoard: son las galletas por partida en las evaluaciones. Un agente al azar consigue cerca de 0,05 y el máximo posible en este mapa es 23.

## Videos

`videos_presentacion.ipynb` no necesita GPU. Monta el Drive donde están los runs y muestra el video más reciente de cada uno. Si alguno es anterior al último checkpoint, cambia `CREAR_VIDEOS` a `True` en la sección 4 para crearlo de nuevo; quedan en `galleta/presentacion/videos/`.
