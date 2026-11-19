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

## Vista 3×3 con obstáculos

El agente sigue en el mapa `hall5` y recibe una imagen RGB de su vista parcial de 3×3 casillas. `env.obstacles=none` usa el mapa original. `same` coloca un cuadrado verde sólido en cada habitación lateral y `distinct` reemplaza el de la derecha por un triángulo verde. Ambos bloquean una casilla y ocupan las mismas posiciones en las dos variantes. Desde el inicio, la ruta más corta pasando por el botón cuesta 23 acciones para cualquiera de las cuatro esquinas en las tres condiciones.

Desde la raíz del repositorio, esta comparación usa los parámetros de entrenamiento del experimento 4 normal y guarda un run por condición y semilla.

```bash
SEMILLA=0
for modo in none same distinct; do
  .venv/bin/python train.py env=cookie env.task=cookie_partial_hall5_view3 \
    env.obstacles="$modo" env.steps=500000 env.train_ratio=128 env.eval_episode_num=5 \
    seed="$SEMILLA" model.rep_loss=dreamer trainer.random_policy=False \
    trainer.eval_every=25000 trainer.update_log_every=2500 trainer.video_pred_log=True \
    trainer.save_buffer_every=100000 buffer.max_size=500000 buffer.storage_device=cpu \
    logdir="logdir/cookie-obstacles/${modo}_s${SEMILLA}" || break
done
```

Si usas el notebook 4 normal en Colab, usa una copia del repositorio que incluya estos cambios. Al final de la celda 6 añade `OPCIONES += " env.obstacles=same"` y `LOGDIR += "_obstacles_same"`, ajustando ambas líneas para cada condición.

Compara `episode/eval_score`, `episode/eval_presses`, `episode/eval_cookie_visible` y `episode/eval_cookie_in_front`. Las dos últimas cuentan decisiones tomadas con la galleta dentro de la vista y justo delante, respectivamente. Si pulsa el botón pero la galleta casi nunca entra en su vista, la dificultad aparece antes de la recogida. Si tiene muchas oportunidades con la galleta justo delante pero recoge pocas, revisa en los videos qué acción elige en esos pasos. Para comparar condiciones, repite las tres con las mismas semillas adicionales.

## Videos

`videos_presentacion.ipynb` no necesita GPU. Monta el Drive donde están los runs y muestra el video más reciente de cada uno. Si alguno es anterior al último checkpoint, cambia `CREAR_VIDEOS` a `True` en la sección 4 para crearlo de nuevo; quedan en `galleta/presentacion/videos/`.
