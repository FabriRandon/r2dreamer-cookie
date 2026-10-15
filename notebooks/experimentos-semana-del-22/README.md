# Semana del 22: aprender con datos de una política random

Los cuatro notebooks recogen datos con **acciones aleatorias uniformes durante todo el entrenamiento** (`trainer.random_policy=True`). El modelo, el actor y el crítico siguen entrenándose (`model.train_actor_critic=True`). En las evaluaciones juega el actor aprendido.

| Observación | Colab |
|---|---|
| Mapa completo | [Abrir](https://colab.research.google.com/github/FabriRandon/r2dreamer-cookie/blob/main/notebooks/experimentos-semana-del-22/vision_completa_random.ipynb) |
| Vista local 3×3 | [Abrir](https://colab.research.google.com/github/FabriRandon/r2dreamer-cookie/blob/main/notebooks/experimentos-semana-del-22/vision_parcial_3x3_random.ipynb) |
| Vista local 5×5 | [Abrir](https://colab.research.google.com/github/FabriRandon/r2dreamer-cookie/blob/main/notebooks/experimentos-semana-del-22/vision_parcial_5x5_random.ipynb) |
| Vista local 7×7 | [Abrir](https://colab.research.google.com/github/FabriRandon/r2dreamer-cookie/blob/main/notebooks/experimentos-semana-del-22/vision_parcial_7x7_random.ipynb) |

Solo cambia la observación. Se mantienen mapa de pasillos cortos, 500.000 pasos, semilla 0, imagen 64×64, decoder Dreamer, train ratio 128, buffer de 500.000 en RAM, muestreo uniforme del buffer (`reward_slices=0`), entropía original (`act_entropy=0.0003`) y evaluación cada 25.000 pasos con 5 partidas.

Los resultados quedan en `MyDrive/galleta/semana22/{tarea}_s0_random`. Cada run empieza desde cero la primera vez y retoma únicamente su carpeta al repetirlo. Los experimentos anteriores se conservan.

Selecciona L4 y ejecuta las secciones 1 a 10. La sección 6 muestra la política configurada; la prueba corta y la sección 10 verifican la configuración efectiva de Hydra antes de iniciar procesos. Rechazan `random_policy=False` y una carpeta anterior incompatible. El guardado y el apagado automático conservan el flujo de los notebooks anteriores.

**Medida de aprendizaje: `episode/eval_score`.** `episode/score` es la recompensa de la política random recolectora y no mide el aprendizaje del actor. El gráfico de la sección 11 y el video de la sección 12 muestran al actor aprendido.
