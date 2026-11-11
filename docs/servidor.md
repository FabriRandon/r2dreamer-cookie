# Correr los experimentos en el servidor (SSH + task spooler)

Guía para entrenar R2-Dreamer / DreamerV3 (PyTorch) en el ambiente de la galleta en el servidor Linux con GPU, en vez de Colab. Todo se hace por SSH y con la cola `ts`, igual que con el DreamerV3 en JAX.

En `server/` están las herramientas; el código del entrenamiento es el mismo que usaste en Colab.

| Archivo | Para qué |
|---|---|
| `server/setup.sh` | instala el entorno (`.venv`, Python 3.11) y lo revisa |
| `server/check_env.py` | revisa GPU, ambiente de la galleta y `torch.compile` |
| `server/run_gpu.sh` | envoltorio para `ts` (como `run_gpu.sh` del otro repo) |
| `server/launch.py` | encola los 9 experimentos de la presentación con su configuración exacta |
| `server/status.py` | tabla con el avance de todos los runs |
| `server/sync_logs.sh` | (en tu PC) trae las métricas del servidor para analizarlas |

Lo que cambió respecto de Colab:

- **Se acabó Drive:** los runs quedan en `~/logdir/r2dreamer/presentacion/<run>/` del servidor, con los mismos nombres de carpeta que en Drive.
- **Detener un run con `ts -k` ahora guarda bien.** `ts -k` manda SIGTERM a *todo* el grupo de procesos del job, incluidos los procesos que simulan el ambiente. Antes esos procesos morían al instante y, si la señal caía mientras `train.py` esperaba al ambiente (por ejemplo en una evaluación), el entrenamiento terminaba con `Lost connection to worker` sin guardar. Ahora los procesos del ambiente ignoran SIGTERM (`envs/parallel.py`) y el run se detiene en el paso en que va y guarda el modelo y el buffer.
- No hay apagado automático de la sesión ni botón de detener: se usa la cola de `ts`.
- La verificación de configuración de los notebooks `random` ahora la hace `launch.py` para **todos** los experimentos antes de encolar.
- `matplotlib` pasó a ser un extra (`.[analysis]`), porque en un venv no viene instalado (en Colab sí). `setup.sh` lo instala.

---

## 1. Conectarse

Conviene tener un alias en `~/.ssh/config` de tu PC:

```text
Host barto
    HostName barto.ing.uc.cl
    User TU_USUARIO
```

```bash
ssh barto
```

Desde ahí todo lo que sigue es en el servidor, salvo lo que diga “en tu PC”. Si se corta la conexión no pasa nada: los jobs están en la cola de `ts`, no en tu terminal.

## 2. Llevar el código al servidor

El repo es un fork con cambios tuyos que quizá todavía no están en GitHub (por ejemplo `server/`). Dos opciones:

```bash
# A) con git: haz commit y push en tu PC, y en el servidor
git clone https://github.com/FabriRandon/r2dreamer-cookie.git ~/r2dreamer-cookie
cd ~/r2dreamer-cookie && git pull --ff-only          # para actualizar después
```

```bash
# B) sin git: copia la carpeta desde tu PC (sin .git, sin logs ni entornos)
rsync -avz --exclude .git --exclude .venv --exclude logdir ./ barto:~/r2dreamer-cookie/
```

El servidor necesita acceso a GitHub: el ambiente (`cookie-env`) se instala desde `github.com/pedropalmav/cookie-env`.

## 3. Instalar (una sola vez)

```bash
cd ~/r2dreamer-cookie
bash server/setup.sh
```

Crea `.venv` con `python3.11`, instala PyTorch 2.8.0 (el pin del repo) y el resto, y corre `server/check_env.py`. Tiene que terminar con `Todo en orden.`

**Sobre el venv:** el entrenamiento siempre usa `.venv`, sin que lo actives (`run_gpu.sh` lo hace por ti dentro de `ts`). `launch.py` y `status.py` solo usan la librería estándar y corren con el `python3` del sistema. Lo que sí necesita el venv (`progress.py --out`, `compare_runs.py`, `watch.py`, `tensorboard`) se corre con `.venv/bin/python ...`, o activándolo una vez por sesión, tras lo cual todos los comandos de esta guía funcionan tal cual:

```bash
cd ~/r2dreamer-cookie && source .venv/bin/activate
```

Problemas conocidos:

| Síntoma | Causa y solución |
|---|---|
| `torch ve una GPU: FALLA` | driver NVIDIA antiguo para CUDA 12.8. Mira `nvidia-smi` y reinstala con otro build: `rm -rf .venv && TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash server/setup.sh` |
| `existen los headers de Python (Python.h): FALLA` o `torch.compile falló` | `model.compile=True` (el valor por defecto, y con el que se hicieron los runs de Colab) necesita `gcc` y `python3.11-dev`. Si no puedes instalarlos, entrena con `--no-compile` (funciona igual y no cambia el checkpoint, pero es **más lento**). Otra salida sin permisos de administrador: un Python 3.11 de `uv python install 3.11` o de conda, que trae los headers (`PYTHON=/ruta/python3.11 bash server/setup.sh`) |
| `El venv ... no tiene pip` | falta el paquete `python3.11-venv`: `rm -rf .venv` y pide el paquete, o usa un Python de `uv`/conda (`PYTHON=...`) |
| `No hay python3.11` | pídelo al administrador, o usa `uv`/conda como arriba |
| falla instalar `cookie-env` | el servidor no llega a GitHub. Instálalo desde un clon que traigas por `scp` |

> `setup.sh` se puede volver a correr cuando quieras; no rehace lo que ya está.

## 4. Prueba corta (recomendado la primera vez)

Entrena 1.500 pasos de cada experimento con buffer chico (que no se guarda), en `~/logdir/r2dreamer/presentacion/_smoke/`. Al terminar deja ahí un `latest.pt`, por eso `launch.py --smoke` vacía esa carpeta del experimento antes de partir (si no, repetir la prueba solo diría `Nothing left to train`). No toca los runs de verdad. Sirve para comprobar que todo anda y estimar la velocidad (`pasos/s` en el log).

```bash
python3 server/launch.py run exp3-normal --smoke
ts                                  # ver el job en la cola
ts -t <id>                          # seguir su salida en vivo (Ctrl+C solo deja de mirar)
```

Si no tienes `gcc` o `Python.h`, agrégale `--no-compile`.

## 5. Lanzar los experimentos

```bash
python3 server/launch.py list                       # los 9 experimentos
python3 server/launch.py run exp3-normal --dry-run  # revisa la configuración y muestra el comando, sin encolar
python3 server/launch.py run exp3-normal exp3-random
python3 server/launch.py run all                    # los 9, uno tras otro según haya GPU libre
```

| Nombre | Qué es | Pasos |
|---|---|---|
| `exp1-normal` / `exp1-random` | mapa completo | 500.000 / 500.000 |
| `exp2-normal` / `exp2-random` | vista 7×7 | 500.000 / 500.000 |
| `exp3-normal` / `exp3-random` | vista 5×5 | 500.000 / 1.000.000 |
| `exp4-normal` / `exp4-random` | vista 3×3 | 500.000 / 750.000 |
| `extra` | 5×5, repitiendo las experiencias con galleta y entropía ×10 | 500.000 |
| `exp3-expert` | 5×5, **todos** los datos los juega el agente experto (`trainer.data_mix.expert=1`, ver [`experto.md`](experto.md)); el actor aprende de ellos y se evalúa solo | 1.000.000 |

Son los pasos que dicen los notebooks (`exp3-expert` no tiene notebook: es la misma configuración de `exp3-random` cambiando quién recoge los datos); el README de `EXPERIMENTOS PRESENTACIÓN/` cuenta los pasos que finalmente se usaron en algunos runs. Cámbialos con `--steps N`.

Cada experimento usa la misma configuración que su notebook: `env=cookie`, tarea `cookie_[full|partial]_hall5[_viewN]`, `train_ratio=128`, imagen de 64×64, decoder (`model.rep_loss=dreamer`), semilla 0, buffer de 500.000, evaluación cada 25.000 pasos con 5 partidas, y guardado del modelo cada 10.000. En los `random`, `trainer.random_policy=True` con `model.train_actor_critic=True`.

Antes de encolar, `launch.py` compone la configuración como lo hará `train.py` y comprueba que sea la del experimento. Si la carpeta del run ya existe con **otra** configuración (otra tarea, otra entropía, otra semilla…), se niega a encolar: continuar un checkpoint con otros ajustes lo estropearía en silencio. Ojo: solo compara esas claves (tarea, tamaño de imagen, `rep_loss`, `random_policy`, actor-crítico, `reward_slices`, entropía, semilla). Con `--extra` cambias ajustes sin que cambie el nombre de la carpeta, así que para una variante nueva usa otra `--logroot`.

Opciones útiles (van antes de `--extra`):

```bash
--seed 1                      # otra semilla: carpeta ..._s1_...
--logroot ~/logdir/r2dreamer/otra   # carpeta donde quedan los runs
--gpu-id 1                    # forzar la segunda GPU (ver nota)
--stop-at 200000              # parar y guardar en ese paso, para seguir otro día
--buffer-device cuda          # buffer en la GPU (~11 GB de VRAM) en vez de en RAM
--no-save-buffer              # no escribir el buffer en disco (ver sección 8)
--save-buffer-every 250000    # guardar el buffer menos seguido (por defecto 100.000)
--no-compile                  # sin torch.compile
--cpu                         # entrena en CPU (solo para pruebas; muy lento)
--no-ts                       # corre en primer plano, sin cola
--extra model.lr=2e-5 ...     # cualquier otro override de Hydra (siempre al final)
```

> **`--gpu-id`:** normalmente deja que `ts -G 1` elija la GPU libre. Con `--gpu-id N` se fija `CUDA_VISIBLE_DEVICES=N` y además `R2D_GPU=N`, porque el cliente de `ts` sobrescribe la primera variable con la GPU que le asigna el servidor. `ts` no se entera de que el job está en otra GPU, así que úsalo solo cuando haga falta. La primera línea del log (`ts -c <id>`) dice qué `CUDA_VISIBLE_DEVICES` tuvo el job.

## 6. Manejar la cola (`ts`)

```bash
ts                      # todos los jobs: estado, id, etiqueta (el nombre del experimento) y comando
ts -t <id>              # cola del log en vivo
ts -c <id>              # todo el log del job
ts -k <id>              # detener un job que corre: se detiene en el paso en que va y GUARDA (espera el mensaje "Saved at step ...")
ts -r <id>              # sacar de la cola un job que todavía no parte
nvidia-smi              # qué hay en las GPU
```

Después de `ts -k` el job aparece como terminado (a veces con error, a veces sin él: no importa). Para saber si guardó: `ts -c <id> | tail` debe decir `Stopped at step N. Saving ...` y `Saved at step N. Run the same command again to continue from there.`. Guardar el buffer completo puede tardar unos minutos. **No** uses `kill -9`: se pierde lo que haya desde el último guardado.

Cuántos jobs corren a la vez lo define `ts` (`ts -S` muestra el máximo de jobs simultáneos) y `-G 1` espera a que haya una GPU con la memoria casi libre. No cambies esas cosas si el servidor es compartido.

**Retomar un run** (porque lo detuviste, se cayó el servidor o llegó al `--stop-at`): corre exactamente el mismo comando `launch.py run ...`. Continúa desde `latest.pt` con su buffer. Para extender un run terminado, súbele `--steps`.

Un run cortado de golpe pierde solo lo recogido desde el último guardado del buffer (cada 100.000 pasos por defecto); el log dice qué tramo se perdió y se vuelve a recoger.

## 7. Ver cómo va

### Tabla de todos los runs

```bash
python3 server/status.py                       # carpeta por defecto: ~/logdir/r2dreamer/presentacion
python3 server/status.py --root ~/logdir/r2dreamer/otra
```

Muestra el paso, el % del total, las galletas por partida de entrenamiento (promedio de las últimas 20), la última evaluación, los pasos por segundo y hace cuánto se escribió la última métrica. No necesita el venv ni GPU, así que se puede correr en cualquier momento. Si un run “en la cola” lleva más de unos 15 minutos sin métricas, mira `ts -t <id>`.

### Un run en detalle

```bash
python3 progress.py --logdir ~/logdir/r2dreamer/presentacion/cookie_partial_hall5_view5_s0_base
```

Compara las últimas partidas con un agente al azar. Con `--out progreso.png` (usa el venv: `.venv/bin/python progress.py ...`; el `python3` del sistema puede no tener matplotlib) dibuja galletas, veces que aprieta el botón, casillas visitadas, pasos en el cuarto del botón, recompensa que predice el modelo y entropía del actor.

Cómo leerlo (de los notebooks y de `progress.py`): primero debería recorrer el mapa al menos como el azar y empezar a apretar el botón; después, las galletas por partida deberían despegarse del azar (~0,05). Como regla práctica (la de los notebooks de monitoreo: unos 300.000 pasos sin galletas sobre el azar en el mapa de pasillos cortos), conviene detener y revisar. En los runs `random`, `episode/score` es del recolector al azar; lo que mide el aprendizaje es `episode/eval_score` (el actor, sin azar).

### TensorBoard por túnel SSH

En el servidor (en una sesión de `tmux`/`screen`, o con `ts` si prefieres):

```bash
cd ~/r2dreamer-cookie && .venv/bin/tensorboard --logdir ~/logdir/r2dreamer/presentacion --port 6006
```

En tu PC:

```bash
ssh -N -L 6006:localhost:6006 barto
```

y abre <http://localhost:6006>. Las pestañas útiles son las mismas de los notebooks: `episode/score`, `episode/presses`, `episode/cells`, `episode/button_room`, `episode/eval_score`, `train/rew`, `train/action_entropy`, y en Images `train_video` y `open_loop`. (Si el puerto 6006 está ocupado por otra persona, cambia el número en ambos comandos.)

## 8. Disco, RAM y GPU

- **RAM:** con `buffer.storage_device=cpu` (lo normal) cada run usa ~11 GB de RAM para 500.000 pasos. Con varios runs a la vez, multiplica. Con `--buffer-device cuda` ese buffer va a la GPU (otros ~11 GB de VRAM).
- **Disco:** cada guardado del buffer escribe `replay/` completo (~11 GB por run; reemplaza al anterior, pero mientras se escribe hay dos copias). `latest.pt` y los eventos de TensorBoard son mucho más chicos. Antes de lanzar los 9 con la configuración por defecto, mira `df -h ~` y tu cuota. Si falta espacio: `--save-buffer-every 250000`, o `--no-save-buffer` (un run reanudado vuelve a juntar sus datos desde cero). Al detener con `ts -k` igual se guarda el buffer, salvo con `--no-save-buffer`.
- Los runs de Colab se hicieron en una L4 (24 GB) y tardaron de 5 a 6 horas por 500.000 pasos con `torch.compile`. La prueba corta te da la velocidad de estas GPU.

## 9. Analizar en tu PC

En tu PC, trae solo lo liviano (métricas, estado, configuración, videos de TensorBoard); nunca los buffers ni los checkpoints:

```bash
R2D_SSH=barto bash server/sync_logs.sh              # incluye los eventos de TensorBoard
R2D_SSH=barto bash server/sync_logs.sh --light      # solo json, yaml y log
# otra carpeta: R2D_REMOTE='~/logdir/r2dreamer/otra' R2D_LOCAL=./otra
```

Se puede repetir cuando quieras (rsync solo copia lo que cambió). Después, igual que con los notebooks:

```bash
python3 progress.py --logdir logdir_server/cookie_partial_hall5_view5_s0_base --out progreso.png
python3 compare_runs.py --root logdir_server --out logdir_server/comparacion
```

`compare_runs.py` junta todos los runs de la carpeta en un gráfico de galletas por experimento, más `resumen.csv`, `curvas.csv` y `lectura.json` (instala antes `pip install matplotlib PyYAML`; en el servidor, usa `.venv/bin/python compare_runs.py ...`). Los nombres de carpeta son los mismos de Drive, así que las etiquetas (“Vista local 5×5”, “repetir experiencias…”) salen igual.

También puedes lanzar `compare_runs`/`progress.py` directamente en el servidor, sin traer nada; solo hay que bajar después el PNG con `scp`.

### Video del agente jugando

`watch.py` carga `latest.pt` y graba una partida (mapa a la izquierda, lo que ve el agente a la derecha). En el servidor:

```bash
cd ~/r2dreamer-cookie
ts -G 1 bash server/run_gpu.sh python watch.py \
  --logdir ~/logdir/r2dreamer/presentacion/cookie_partial_hall5_view5_s0_base \
  --out ~/logdir/r2dreamer/presentacion/cookie_partial_hall5_view5_s0_base/agente_episodio0.mp4 --seed 0
```

(Sin `-G`, `ts` lo corre en CPU, que alcanza: tarda uno o dos minutos.) Se trae con `scp barto:~/logdir/r2dreamer/presentacion/<run>/agente_episodio0.mp4 .` o con `sync_logs.sh`, que incluye los `.mp4`. `compare_runs.find_video` los reconoce con ese nombre.

## 10. Correr algo a mano

Para un experimento distinto de los 9, usa el envoltorio con `ts`:

```bash
ts -G 1 bash server/run_gpu.sh python -u train.py env=cookie env.task=cookie_partial_hall5_view5 \
  env.train_ratio=128 model.rep_loss=dreamer buffer.storage_device=cpu \
  trainer.save_buffer_every=100000 \
  logdir=$HOME/logdir/r2dreamer/mio/prueba1
```

Usa **rutas absolutas** en `logdir` (`$HOME/...`): Hydra crea su carpeta de configuración (`.hydra/`) justo en `logdir`, y un `~` sin expandir crearía una carpeta llamada `~` dentro del repo. Ojo: fuera de `launch.py` rigen los valores de `configs/configs.yaml`, p. ej. el buffer se guardaría completo cada 10.000 pasos (~11 GB cada vez); por eso el ejemplo fija `trainer.save_buffer_every`. Y deja siempre el `python -u` y el `exec` de `run_gpu.sh`: sin eso, `ts -k` no guarda.

## 11. Problemas frecuentes

| Síntoma | Qué hacer |
|---|---|
| `ts` dice `command not found` | prueba `which ts`; con otro binario, `TS=/ruta/ts python3 server/launch.py ...` o `--ts-bin` |
| el job termina enseguida | `ts -c <id>`: la causa real suele estar al principio. Si es `CUDA out of memory`, otro proceso usa esa GPU: `nvidia-smi`; prueba `--extra batch_size=8` o `--buffer-device cpu` (si el buffer estaba en la GPU) |
| `size mismatch for ...` / `Missing key(s) in state_dict` / `Unexpected key(s)` al cargar | el checkpoint no calza con la configuración: casi siempre una carpeta reutilizada con otra vista o `rep_loss`. `launch.py` lo avisa antes; usa otra `--logroot` |
| error al compilar (`Python.h`, `gcc`, `InductorError`) | `--no-compile` |
| `Nothing left to train: the run is at step N` | ya llegó a `--steps`; súbelo para extender |
| no aparece `moviepy`/`ffmpeg` al grabar un video | `.venv/bin/pip install imageio-ffmpeg` (moviepy 1.0.3 la trae; revisa `check_env.py`) |
| una prueba o run quedó a medias con procesos colgados | `pgrep -af train.py` (te muestra los tuyos; incluye los procesos del ambiente); mátalos con `kill <pid>` (SIGTERM, guarda) y solo como último recurso `kill -9` |
