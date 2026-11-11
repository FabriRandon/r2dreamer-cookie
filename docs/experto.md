# Entrenar con trayectorias de un agente "perfecto"

El ambiente de la galleta trae una política escrita a mano (`envs/expert.py`) que lo resuelve casi sin errores, y el trainer puede mezclar episodios de tres fuentes distintas.

## El agente perfecto

Solo ve la misma vista parcial de 5×5 que Dreamer (en su versión simbólica) y recuerda dónde está por los movimientos que hizo; conoce el mapa, pero no dónde está la galleta. Cada ciclo:

1. va al botón y choca contra él (aparece la galleta en una esquina de una sala lateral);
2. vuelve al centro y mira una de las dos salas, elegida al azar, desde su entrada (ahí la vista de 5×5 cubre toda la sala);
3. si ve la galleta va por ella; si no, hace lo mismo en la otra sala (también va directo por ella si la ve de paso);
4. vuelve por el centro y presiona el botón de nuevo, hasta que termina el episodio.

Un ciclo (de un botón al siguiente) dura 35 pasos si acierta la sala a la primera y 43 si no (promedio ≈ 39). En `hall5` un episodio dura 816 pasos (4·12·17, el límite de MiniGrid; el `time_limit: 2088` de `configs/env/cookie.yaml` no llega a usarse), o sea ≈ 20,7 galletas por episodio (medido en 60 episodios: entre 20 y 22). Ese es el techo del `episode/eval_score`; el run de referencia de visión completa llegó a ~21. Solo funciona con `cookie_*_hall5_view5` (o vista mayor): pasillos de 5 casillas y `press_on_move`.

Para verlo resolver el ambiente (escribe `expert_demo/` con video, GIF y una hoja de fotogramas) y para ver estadísticas:

```bash
python demo_expert.py --steps 120 --seed 4
python demo_expert.py --stats 20
```

## Mezclar fuentes de datos

`trainer.data_mix` define qué fracción de los **episodios** juega cada fuente:

| Fuente | Quién elige las acciones |
|---|---|
| `actor` | el actor de Dreamer (lo normal) |
| `random` | acciones uniformemente al azar |
| `expert` | el agente perfecto |

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
