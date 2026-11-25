# El mapa original: pasillos de 11 casillas

Casi todos los experimentos usan `cookie_partial_hall5_*`: el mapa del juego con los pasillos acortados de 11 a 5 casillas. Con `hall11` el mapa es **el original** (una cuadrícula de 29×18; un test comprueba que es idéntico al de `cookie-env`). Hay dos experimentos del lanzador, los dos con visión 5×5, **2.000.000 de pasos**, episodios de **3000 pasos** y **solo Dreamer** juntando los datos (sin azar ni experto):

| Experimento | Obstáculos | Carpeta |
|---|---|---|
| `exp3-hall11-obstaculos` | uno por sala, nuevo en cada episodio | `cookie_partial_hall11_view5_s0_base_obstaculos` |
| `exp3-hall11-sin-obstaculos` | no | `cookie_partial_hall11_view5_s0_base_sin_obstaculos` |

```bash
python3 server/launch.py run exp3-hall11-obstaculos exp3-hall11-sin-obstaculos --partition <PARTICION> --time 08:00:00 --chain 2
```

## Qué cambia respecto de `exp3-obstaculos`

(Las dos nuevas se comparan entre sí solo por los obstáculos.)

| Parámetro | `exp3-obstaculos` | los hall11 | Por qué |
|---|---|---|---|
| mapa | `hall5` | `hall11` | lo que se pidió |
| `env.steps` | 1.000.000 | **2.000.000** | con episodios largos se ven pocos episodios por paso (abajo) |
| `env.time_limit` y `env.max_steps` | 816 (lo de MiniGrid, `4·alto·ancho`) | **3000** | a las partidas de 2088 pasos del mapa original les puede faltar tiempo para ir, apretar el botón y llegar a la galleta |
| `model.act_entropy` | 0,0003 | **0,003** | que la política no se vuelva determinista antes de encontrar galletas |
| `trainer.eval_every` | 25.000 | **50.000** | una partida de evaluación ahora dura 3000 pasos (3,7 veces las de `hall5`) |
| `trainer.video_tile_size` | 16 | **8** | con 16 px, codificar el video de un episodio ocupa **12 GB de RAM** y 30 s (medido); con 8, 3,5 GB y 6 s |

Lo que **no** cambia: `env.train_ratio` (128), buffer de 500.000, semilla, 5 partidas por evaluación, `batch_size` 16 y `batch_length` 64.

### El límite de 3000 pasos: son dos ajustes

`env.time_limit` solo (el que ya existía) **no basta**: MiniGrid corta cada partida a sus `4·alto·ancho` pasos (816 con pasillos de 5, 2088 con 11) antes de que llegue el límite del contenedor, así que con `time_limit=3000` las partidas seguirían durando 2088 (un test lo comprueba). Por eso se agregó `env.max_steps` (por defecto `null`: el de MiniGrid, como en todos los runs anteriores) y el lanzador pone los dos en 3000. La partida de `hall5` (816) no cambia.

### ¿Subir `train_ratio` con más pasos? No

- Con 2.000.000 de pasos a 128 se hacen **250.000 actualizaciones**, las mismas que 1.000.000 de pasos a 256 (y el doble que `exp3-obstaculos`, 125.000). Los pasos extra ya traen las actualizaciones extra.
- El tiempo manda. Los logs de `exp3-obstaculos` y `exp3-sin-obstaculos` marcan **4 h 22 – 4 h 30 por 1.000.000 de pasos** (62–63 pasos/s; la estimación de unas 3 h no sale de esos logs: si el nodo está menos cargado ahora, puede ser menos). A 128, **2.000.000 de pasos son ~8,8 h**, más las evaluaciones de `hall11`: 40 evaluaciones × 5 partidas × 3000 pasos = 600.000 pasos de evaluación (en `hall5` fueron 163.000), que suman ~1 h (suponiendo que un paso de evaluación cuesta como la mitad de uno de entrenamiento: no está medido). Total: **~9–10 h**. Con 256 serían ~16 h o más.
- Ese total **pasa de las 8 horas**. Por eso el comando usa `--time 08:00:00 --chain 2`: cada job se detiene limpio unos minutos antes de las 8 h, guarda el modelo y el buffer, y el segundo sigue desde ahí. Si las 8 h son el **total** por run y no por job, las alternativas son bajar el ritmo (`--extra env.train_ratio=64`, ~8 h, la mitad de actualizaciones por paso) o reducir los pasos (`--steps 1500000`); ninguna es gratis en aprendizaje.
- Se calibra en media hora: el log de Slurm (`~/logdir/r2dreamer/presentacion/slurm/<nombre>-<id>.out`) imprime cada pocos minutos `progress ... left ~H:MM:SS ... N steps/s`. Con ≥ 62 pasos/s a 128 va como lo calculado; con menos, hay que contar con más tiempo.

### Lo que se midió

- **Un agente al azar** (3 acciones, 320 a 650 episodios por mapa, sin obstáculos): gana una galleta en ~5 % de los episodios en los dos mapas (0,05 en `hall5`; 0,05–0,06 en `hall11`). Pero un episodio de `hall11` dura 2,6 veces más (3,7 con 3000 pasos), así que **por paso encuentra ~2,6 galletas cada 100.000 contra ~6 en `hall5`**. Y el mismo run ve menos episodios: ~480 en 1M de pasos (667 en 2M con partidas de 3000) contra 1225.
- **El run de `exp3-obstaculos`** (el real, de `barto`): las primeras galletas con regularidad llegaron hacia los 330.000 pasos (episodios 300–400: 0,05 → 0,66 galletas por partida) y siguió mejorando hasta el final (eval 12 al paso 1M; el experto de 5×5 saca ~20). Mientras tanto la **entropía de la política cayó de 1,10 a 0,16 antes de los 250.000 pasos**, es decir, se volvió casi determinista antes de encontrar galletas con regularidad.

### Qué se concluye (estimación, no medición del run)

1. **Entropía (cambiar).** Si en `hall5` la política se volvió casi determinista unos 80.000 pasos antes de encontrar galletas, en `hall11`, donde encontrarlas lleva ~2,5 veces más pasos, el riesgo de quedarse repitiendo una rutina sin galletas es mayor. 0,003 (10 veces) es lo que ya probaba el experimento `extra`, y sigue siendo un término pequeño frente a la recompensa. Cómo se mira: `train/action_entropy` en TensorBoard; con 3 acciones el máximo es 1,10.
2. **Pasos (2M).** Si el escalado por episodios vale, `hall11` necesita ~2,5 veces los pasos de `hall5` para el mismo punto: el despegue estaría hacia los 800.000 y 2M deja margen para aprender después.
3. **Buffer (dejarlo en 500.000).** Un buffer es un FIFO: guardar 1.000.000 de pasos costaría ~22 GB de RAM (más otros 22 GB de disco en cada guardado) y subir `--mem` a ~56 GB. No ayuda a lo que importa: lo valioso son los episodios con galleta que juega el actor *mismo*, que son recientes. Cuántas veces se repite cada paso lo da `train_ratio`, no el tamaño del buffer.
4. **`reward_slices` (no se activa).** Repite en cada lote secuencias que ya tuvieron recompensa. Ayudaría a la cabeza de recompensas con eventos raros, pero sesga el modelo hacia sobreestimar cuánto gana, y no hay una medición propia que lo respalde. Si el run no despega, es la primera cosa que probar (`--extra buffer.reward_slices=1`).

**Esto es razonado a partir de mediciones del ambiente y del run de `hall5`, no verificado entrenando `hall11`.** Para ver si la entropía importa, repite uno con la original en otra carpeta:

```bash
python3 server/launch.py run exp3-hall11-obstaculos --logroot ~/logdir/r2dreamer/base_hall11 --partition <PARTICION> --extra model.act_entropy=3e-4 trainer.eval_every=25000
```

## Qué mirar en las curvas

- `episode/eval_score`: lo principal. En `hall5` el despegue fue a los ~330k.
- `episode/presses` y `episode/cells`: el progreso antes de las primeras galletas. El azar da ~6 apretones de botón y ~49 casillas distintas por episodio (`progress.py` lo dibuja como referencia).
- `train/action_entropy`: si cae por debajo de ~0,3 antes de que `episode/score` suba, el run se quedó sin explorar.
- `episode/bumps` (solo con obstáculos).
- Los videos (`train_video`, `eval_video`) del mapa completo, a 8 px por casilla.
