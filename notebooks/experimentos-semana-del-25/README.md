# Semana del 25: tres obstáculos de formas y colores distintos

Notebook 5×5 con `buffer.reward_slices=1`, basado en el 5×5 con sampler de la semana del 24. En `distinct`, las habitaciones laterales reciben un cuadrado verde y un triángulo morado, intercambiados aleatoriamente por episodio; la habitación del botón recibe un rombo amarillo. Cada figura cambia de casilla dentro de su habitación al reiniciar y permanece quieta durante la partida. `same` es el control de tres cuadrados verdes; `none` no tiene obstáculos.

Cada habitación tiene 18 casillas posibles. En la del botón se excluyen toda la columna central de acceso y las cuatro casillas contiguas al botón. Ninguna figura ocupa pasillos, el botón ni esquinas de galletas. La comprobación recorre las 5.832 combinaciones posibles y exige que todas las casillas transitables permanezcan conectadas. También prueba 384 episodios con vistas 3×3 y 5×5, dos semillas y los tres modos: reproducibilidad, estabilidad durante la partida, geometría compartida entre same/distinct, acceso al botón desde sus cuatro lados, recolección en las cuatro esquinas e independencia del sorteo de galletas.

El notebook incorpora el entorno como parche contra una revisión fija del repositorio y la comprobación completa. Se puede subir directamente a Colab. La vista previa muestra tres episodios de los tres modos.

Entrenamiento: 500.000 pasos, semilla 0 por defecto, actor Dreamer, decoder, train ratio 128, imágenes RGB de 64×64 y evaluación cada 25.000 pasos con cinco partidas. Resultados propios en `MyDrive/galleta/semana25/tres_obstaculos_colores5x5/{modo}_sampler1_s{semilla}`. No retoma los checkpoints de dos obstáculos. La sección 6 compara con las métricas anteriores de 5×5 con sampler, leídas de su carpeta original. El número de obstáculos y los colores cambian a la vez; esta comparación no aísla sus efectos individuales.

Comprobación local desde la raíz del repositorio:

```bash
.venv/bin/python notebooks/experimentos-semana-del-25/verificar_entorno.py notebooks/experimentos-semana-del-25/cookie.py
```
