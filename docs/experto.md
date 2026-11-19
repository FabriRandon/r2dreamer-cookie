# Entrenar con trayectorias de un agente "perfecto"

El ambiente de la galleta trae políticas escritas a mano (`envs/expert.py`) que lo resuelven casi sin errores, y el trainer puede mezclar episodios de tres fuentes distintas.

## Los agentes perfectos

Hay dos versiones de la política escrita a mano, según la vista de la que decide el experto. Se elige con `env.expert_view` (5 por defecto, que es lo que usaron los experimentos anteriores). Los dos conocen el mapa, recuerdan dónde están por los movimientos que hicieron (sin leer el estado del simulador) y **no saben dónde apareció la galleta**: tienen que verla. La galleta aparece siempre en una de las 4 esquinas exteriores de las salas laterales (2 por sala). Solo funcionan en el mapa `hall5` (pasillos de 5 casillas) y con `press_on_move`, y son independientes de la vista del agente (`view3`, `view5`...): esa solo cambia las imágenes que recibe el modelo.

### Experto de 3×3 (`env.expert_view=3`): el de `exp4-mix`

Decide solo con una vista de 3×3 casillas (dos adelante y una a cada lado), la misma que recibe un agente con `view3`. No alcanza a ver el interior de una sala desde su entrada, así que revisa las esquinas así:

1. va al botón y choca contra él (aparece la galleta en una esquina);
2. elige al azar una de las dos salas laterales, **entra por la fila del medio y sigue avanzando hasta ver el fondo** (la pared del fondo queda dos casillas adelante: a 6 casillas del inicio);
3. elige al azar si revisa primero **arriba o abajo** y gira hacia ese lado: en esa posición la esquina exterior queda dentro de la vista de 3×3 (la vista hacia un costado alcanza dos casillas por delante, así que **basta girar; no hace falta caminar más hacia la esquina**);
4. si ve la galleta, va por ella; si no, se da vuelta y revisa la otra esquina de la misma sala;
5. si no está en ninguna de las dos, está en la otra sala: repite lo mismo allá;
6. con la galleta, vuelve por el centro y presiona el botón de nuevo, hasta que termina el episodio.

Un ciclo (de un botón al siguiente) dura **35 pasos** si la galleta está en la primera esquina que mira, **37** si está en la segunda esquina de esa sala, **51** si está en la primera esquina de la otra sala y **53** si está en la segunda: las cuatro esquinas son igual de probables, así que cada duración sale ~25 % de las veces (medido en 60 episodios completos, unos 1.070 ciclos: 251, 268, 285 y 268). En `hall5` un episodio dura 816 pasos (4·12·17, el límite de MiniGrid; el `time_limit: 2088` de `configs/env/cookie.yaml` no llega a usarse): **≈ 18,2 galletas por episodio** (promedio de 60 episodios; entre 16 y 20). Es el techo razonable del `episode/eval_score` con vista 3×3.

Cada ciclo es el más corto posible dado el orden en que decide mirar (los tests lo comprueban contra una búsqueda de caminos independiente para las 32 combinaciones de esquina y decisiones al azar).

### Experto de 5×5 (`env.expert_view=5`, el original)

Decide con una vista de 5×5 propia. Cada ciclo:

1. va al botón y choca contra él;
2. vuelve al centro y mira una de las dos salas, elegida al azar, desde su entrada (ahí la vista de 5×5 cubre toda la sala);
3. si ve la galleta va por ella; si no, hace lo mismo en la otra sala;
4. vuelve por el centro y presiona el botón de nuevo, hasta que termina el episodio.

Un ciclo dura 35 pasos si acierta la sala a la primera y 43 si no (promedio ≈ 39), o sea ≈ 20,8 galletas por episodio (medido en 60 episodios: entre 20 y 22). Con la misma semilla hace exactamente las mismas acciones aunque el agente vea `view3`, `view5` o `view7`.

### Con obstáculos

Si el ambiente tiene obstáculos sólidos (`env.obstacles=1`, ver [`obstaculos.md`](obstaculos.md)), los dos expertos los recuerdan al verlos y los rodean (también al presionar el botón: lo hacen desde un costado si hay uno delante). El de 3×3 mira cada esquina desde la mejor posición alcanzable desde la que se ve; con la sala vacía es exactamente el recorrido de arriba. A cambio dejan de ser óptimos: 17,4 galletas por episodio el de 3×3 (18,2 sin obstáculos) y 20,1 el de 5×5 (20,8).

### Verlos en video

```bash
python demo_expert.py --expert-view 3 --showcase     # los 4 casos de un ciclo, uno tras otro (expert_demo_3x3/)
python demo_expert.py --expert-view 3 --steps 400    # jugando de forma natural
python demo_expert.py --expert-view 3 --stats 40     # solo números (galletas por episodio y duración de los ciclos)
python demo_expert.py                                # el experto de 5×5 (expert_demo/)
```

Cada video muestra el mapa completo, con lo que ve el experto aclarado y enmarcado en amarillo, y arriba el paso, las galletas, qué está haciendo («entra a la sala este hasta ver el fondo», «gira para mirar arriba en la sala oeste», «va por la galleta»...) y su acción. Escribe `.mp4`, `.gif` y una hoja de fotogramas del primer ciclo. En `--showcase` se fuerzan las decisiones al azar y la esquina de la galleta para que cada uno de los cuatro ciclos muestre un caso.

## Mezclar fuentes de datos

`trainer.data_mix` define qué fracción de los **episodios** juega cada fuente:

| Fuente | Quién elige las acciones |
|---|---|
| `actor` | el actor de Dreamer (lo normal) |
| `random` | acciones uniformemente al azar |
| `expert` | el agente perfecto (`env.expert_view` elige cuál) |

Los pesos se normalizan (solo importan las proporciones). Cada episodio se asigna a una fuente cuando empieza, al azar con esas probabilidades, y la mantiene hasta que termina. Si no se define ninguno, el actor juega todo, como siempre. Dreamer sigue entrenando online con el mismo `train_ratio`; lo único que cambia es quién generó los datos. **La evaluación siempre usa el actor**, para medir lo que Dreamer aprendió.

```bash
# solo trayectorias del experto
python train.py env=cookie env.task=cookie_partial_hall5_view5 trainer.data_mix.expert=1

# 50 % experto, 30 % actor, 20 % azar
python train.py env=cookie env.task=cookie_partial_hall5_view5 \
    trainer.data_mix.expert=5 trainer.data_mix.actor=3 trainer.data_mix.random=2
```

`trainer.random_policy=True` sigue funcionando (equivale a `data_mix.random=1`), pero no se puede combinar con `data_mix`.

En las métricas aparece, por episodio terminado, `episode/score_actor`, `episode/score_random` y `episode/score_expert` (el puntaje separado por fuente) y `data/episodes_*` (la fracción de episodios de cada fuente hasta ese momento).

## Videos del entorno completo

Por defecto `train_video` y `eval_video` de TensorBoard muestran lo que recibe el agente (su imagen de 64×64). Con `trainer.video_render=True` (solo en la tarea de la galleta) muestran **todo el entorno**, con lo que ve el agente aclarado y **enmarcado en amarillo** (el recuadro de 3×3 o 5×5 casillas delante de él, recortado donde sale del mapa), más el botón azul, la galleta y el agente (el triángulo rojo). Así se ve de un vistazo dónde está y por qué actúa como actúa.

```bash
python train.py env=cookie env.task=cookie_partial_hall5_view3 trainer.video_render=True trainer.train_video_every=10
```

- `eval_video`: el primer episodio completo de la evaluación, siempre jugado por el actor.
- `train_video`: un episodio de entrenamiento cada `train_video_every` (por defecto 1). Si el entrenamiento mezcla fuentes (`data_mix`), el video dice quién lo jugó en el nombre: `train_video_actor`, `train_video_random` y `train_video_expert`, y el contador es por fuente, de modo que cada una recibe sus propios videos.
- `trainer.video_tile_size` (16 por defecto) son los píxeles por casilla: el video mide 192×272 en el mapa `hall5`. Cada video pesa entre 0,5 y 0,7 MB en el archivo de eventos y tarda unos 5 s en codificarse (medido con un episodio del experto de 816 pasos); con `train_video_every=10` son unos 120 videos de entrenamiento más 41 de evaluación en un run de 1M pasos.
- No se cambia nada del entrenamiento ni del buffer: los cuadros se piden al entorno 0 solo mientras se graba.
