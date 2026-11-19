# Obstáculos abstractos en cada sala

En el mapa de pasillos (`hallN`) se puede poner **un obstáculo sólido en cada sala** (la del botón y las dos laterales donde aparecen las galletas). Son figuras abstractas de forma y color al azar, y **se sortean de nuevo en cada episodio**: otra figura, otro color y otro lugar en cada sala. Así Dreamer nunca los ve fijos, y tiene que aprender que *cualquier* figura nueva bloquea el paso.

```bash
python train.py env=cookie env.task=cookie_partial_hall5_view3 env.obstacles=1
```

`env.obstacles` es la cantidad de obstáculos **por sala** (`0` = ninguno, que es lo normal y no cambia nada de los experimentos anteriores). Hay dos experimentos del lanzador con obstáculos: `exp5-mix` es `exp4-mix` (visión 3×3, datos en tercios actor, azar y experto) con `env.obstacles=1`, y `exp3-obstaculos` es visión 5×5 con **solo Dreamer** juntando los datos:

```bash
python3 server/launch.py run exp5-mix
python3 server/launch.py run exp3-obstaculos
```

## Qué son

- **Figuras:** estrella, rombo (alto), cuadrado, cruz «+» y equis «×».
- **Colores:** amarillo, blanco, verde y magenta.
- **Las de un episodio son distintas entre sí:** como hay 5 figuras y 4 colores, los 3 obstáculos del episodio tienen figuras distintas y colores distintos.
- **Para que no se parezcan a lo importante** no hay círculos (el botón es un círculo azul y la galleta uno café con chispas) ni triángulos (el agente), ni azul, rojo o café. Los tests lo miden: la silueta de ninguna figura se parece mucho a la del botón, la galleta o el agente, y los colores quedan lejos del azul, el rojo y el café de la galleta.
- **Sólidos:** si el agente avanza contra uno, no se mueve. En la imagen del agente aparecen como una mancha del color de la figura; en la vista simbólica tienen el id `13` (el botón es `11` y la galleta `12`).

## Dónde pueden aparecer

En cualquier casilla libre de su sala, elegida al azar con la misma probabilidad (en la sala del botón 23 casillas; en cada sala lateral 22), **menos**:

- las dos esquinas exteriores de cada sala lateral, que es donde aparece la galleta;
- la puerta de la sala (la casilla por donde entra su pasillo): ahí sellaría la sala entera;
- el botón.

Además, un sorteo se descarta (y se repite) si deja alguna esquina de galleta o el botón inalcanzables desde la casilla de partida. Con la puerta excluida y uno solo por sala eso no ocurre nunca en la práctica (0 descartes en 3.000 sorteos); es una red de seguridad.

## Cómo se ve y se mide

- Los videos de entrenamiento y evaluación (`trainer.video_render=True`) los muestran en el mapa completo.
- Aparece una métrica nueva, **`episode/bumps`** (y `episode/eval_bumps`): cuántas veces el agente avanzó contra un obstáculo en la partida. Si Dreamer aprendió que bloquean, baja con el entrenamiento. `progress.py` la muestra como «choques con obstáculos».
- `compare_runs.py` agrega «con obstáculos» a la etiqueta de estos runs.

## Reproducibilidad

El sorteo tiene su propio generador de números aleatorios, sembrado con la semilla del run: el mismo `seed` da la misma sucesión de obstáculos, y otra semilla da otra. No usa los generadores de la galleta ni del experto, así que activar los obstáculos no altera lo demás.

## Los expertos con obstáculos

Los dos expertos (`env.expert_view=5` y `3`, ver [`experto.md`](experto.md)) los rodean. Los obstáculos son sólidos y **nada dice dónde están**, así que el experto:

1. recuerda los que va viendo en su vista (también los que quedan a los lados de su camino) y los olvida al terminar el episodio;
2. planifica el camino más corto evitando los que conoce, y vuelve a planificar en cada paso con lo nuevo que ve. Como la casilla de adelante siempre está en su vista, nunca avanza contra uno;
3. presiona el botón desde cualquiera de sus cuatro lados (si hay un obstáculo delante, lo hace desde un costado);
4. el experto de 3×3 mira cada esquina desde **la mejor posición alcanzable desde la que se ve**: con la sala vacía es exactamente la de siempre (entra por el medio hasta ver el fondo y gira), y con un obstáculo en el camino o justo en esa casilla, la siguiente mejor. Una esquina que quedó dentro de su vista sin galleta, aunque sea de pasada, cuenta como revisada.

Ya no son óptimos, porque no saben dónde están hasta verlos. Promedio de 60 episodios completos de cada caso (`demo_expert.py --stats 60`):

| | Sin obstáculos | Con obstáculos |
|---|---|---|
| Experto 3×3 | 18,2 galletas por episodio (16 a 20) | 17,4 (15 a 20) |
| Experto 5×5 | 20,8 (20 a 22) | 20,1 (17 a 22) |

En ninguno de los episodios chocaron ni perdieron su ubicación (lo comprueban los tests). Los tests además los prueban con 20 colocaciones elegidas para estorbarlos (delante del botón, justo donde el 3×3 se para a mirar, en la fila del medio de cada sala, junto a las puertas, rodeando el botón...).

## Verlo en video

```bash
python demo_expert.py --expert-view 3 --obstacles 1            # una partida natural, obstáculos nuevos en cada episodio
python demo_expert.py --expert-view 3 --obstacle-showcase      # tres colocaciones que le estorban, a propósito
```

Escriben en `expert_demo_3x3_obstaculos/`.
