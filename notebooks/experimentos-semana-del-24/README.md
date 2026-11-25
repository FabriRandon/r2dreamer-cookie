# Semana del 24: figuras con ubicación aleatoria

Versiones nuevas de los experimentos con obstáculos de la semana del 19, con vistas 3×3 y 5×5. El actor de Dreamer recolecta los datos. Se mantienen 500.000 pasos, semilla inicial 0, decoder Dreamer, train ratio 128 y evaluación cada 25.000 pasos con cinco partidas.

| Notebook | Vista | Resultados en Drive |
|---|---|---|
| `vision_parcial_3x3_obstaculos_aleatorios.ipynb` | 3×3 | `MyDrive/galleta/semana24/obstaculos_aleatorios3x3/{modo}_s{semilla}` |
| `vision_parcial_5x5_obstaculos_aleatorios.ipynb` | 5×5 | `MyDrive/galleta/semana24/obstaculos_aleatorios5x5/{modo}_s{semilla}` |
| `vision_parcial_3x3_obstaculos_aleatorios_sampler1.ipynb` | 3×3 | `MyDrive/galleta/semana24/obstaculos_aleatorios3x3/{modo}_sampler1_s{semilla}` |
| `vision_parcial_5x5_obstaculos_aleatorios_sampler1.ipynb` | 5×5 | `MyDrive/galleta/semana24/obstaculos_aleatorios5x5/{modo}_sampler1_s{semilla}` |

Los notebooks `_sampler1` son iguales a los anteriores salvo `buffer.reward_slices=1`: de las 16 secuencias de cada batch, una se corta alrededor de un paso en que el actor comió una galleta, si ya hay alguna en el buffer. Su sección 6 compara cada condición con y sin sampler.

Al comenzar cada episodio se sortea una casilla para cada figura, una por habitación lateral. En `distinct` también se sortea qué habitación recibe el triángulo y cuál el cuadrado. En `same` ambas figuras son cuadrados. Se mantienen quietas durante todo el episodio, incluso al pulsar de nuevo el botón o recoger galletas. Un sorteo puede repetir una ubicación anterior.

Cada habitación tiene 18 casillas posibles, en posiciones simétricas (en espejo) entre ambas. Se excluyen la fila central, por donde entra el agente, y las esquinas donde aparecen galletas. Todas las esquinas siguen siendo accesibles. La longitud del recorrido puede depender de la casilla sorteada.

Para la misma semilla y número de episodio, `same` y `distinct` bloquean las mismas casillas. La apariencia cambia sin alterar el sorteo de las galletas. `none` conserva el mapa sin figuras.

La distribución elimina la asociación permanente entre forma y habitación. Esto favorece estudiar referencias que el agente debe recordar dentro de una partida, pero no demuestra por sí solo el uso de memoria. Para medirlo directamente se requiere una comparación con y sin estado recurrente y varias semillas.

## Ejecutar en Colab

Abre uno de los notebooks, selecciona una GPU L4 y ejecuta las secciones en orden. El notebook prepara una copia propia de una revisión fija del repositorio y aplica el cambio incluido. No necesita descargar esta carpeta del repositorio: el cambio y la verificación están incorporados en cada notebook, por lo que también se puede subir el archivo directamente a Colab.

La sección 2 valida el entorno y muestra tres episodios de las tres condiciones. Después elige `none`, `same` o `distinct`; cada condición y semilla tiene su propio checkpoint. Las secciones finales comparan evaluaciones y graban al actor aprendido. La prueba corta de entrenamiento es opcional.

## Código y comprobaciones locales

`cookie.py` contiene la variante completa del entorno. Los notebooks incorporan su diferencia respecto de la revisión fija; solo aplican esa diferencia dentro de sus copias de Colab. El archivo principal `envs/cookie.py` y los notebooks anteriores conservan las posiciones fijas.

Desde la raíz del repositorio, con las dependencias de Cookie instaladas:

```bash
.venv/bin/python notebooks/experimentos-semana-del-24/verificar_entorno.py notebooks/experimentos-semana-del-24/cookie.py
```

La comprobación ejecuta 384 episodios entre condiciones, vistas, semillas y repeticiones. Verifica cambios de ubicación, intercambio de formas entre habitaciones, estabilidad dentro de cada partida, reproducibilidad, igualdad de geometría entre `same` y `distinct`, independencia del sorteo de galletas, acceso al botón y recolección de galletas en las cuatro esquinas.
