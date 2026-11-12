# Sacar los resultados del servidor

Guía corta para llevarte un run de `barto` a tu computador y: (1) verlo en TensorBoard, (2) probar el modelo de mundo que aprendió.

## Qué hay en la carpeta de un run

Cada run queda en `~/logdir/r2dreamer/presentacion/<nombre_del_run>/` del servidor:

| Archivo | Para qué sirve | Tamaño |
|---|---|---|
| `events.out.tfevents.*` | TensorBoard: curvas, videos e imágenes | grande (~650 MB a los 450k pasos; más en 1M) |
| `metrics.jsonl` | las mismas curvas en texto (`progress.py`, `compare_runs.py`, el notebook) | pequeño |
| `.hydra/config.yaml` | la configuración con la que se entrenó (hace falta para cargar el modelo) | pequeño |
| `latest.pt` | el agente aprendido: **el modelo de mundo, el actor y el crítico** | ~125 MB |
| `status.json`, `console.log` | en qué paso quedó y el log del entrenamiento | pequeño |
| `replay/` | el buffer de datos | **muchos GB (~11 GB con 500k pasos)** |

Para ver curvas y para probar el modelo **no** hace falta `replay/`. (El notebook de exploración usa `metrics.jsonl` en su lugar.)

## 0. Antes: ¿terminó el job de Slurm?

`barto` usa Slurm: el entrenamiento es un job (`squeue -u $USER`). Exporta cuando haya terminado, o antes si solo quieres mirar lo último guardado.

```bash
squeue -u $USER                                            # si ya no aparece, terminó
sacct -j <id_del_job> --format=JobID,State,Elapsed,ExitCode   # COMPLETED = terminó bien
python3 server/status.py                                   # el paso en que quedó el run (1.000.000 de 1.000.000)
```

El `<id_del_job>` lo imprime `launch.py` al encolar (`job N encolado`) y está en el nombre del log: `~/logdir/r2dreamer/presentacion/slurm/<nombre>-<id>.out`. Un job que llegó al límite de tiempo y se detuvo y guardó puede figurar como `COMPLETED` sin haber llegado al último paso: mira `status.py`, y si falta, repite el mismo `launch.py run ...` para continuar.

## 1. En el servidor: empaquetar el run

```bash
cd ~/r2dreamer-cookie
bash server/export_run.sh ~/logdir/r2dreamer/presentacion/cookie_partial_hall5_view5_s0_expert
```

Deja un `.tar.gz` en `~/exports/` (y su `.sha256`) con todo lo de la tabla menos `replay/`, más un `VERSION.txt` con el commit del repo **al momento de exportar** (si hiciste `git pull` después de lanzar el run, ese commit puede ser posterior al de entrenamiento). Cambia el nombre de la carpeta por el del run que quieras (por ejemplo `..._s0_mix`).

Comprimir ~0,5 a 1 GB tarda unos 30 segundos con un solo núcleo, así que se puede correr directo en la sesión de entrada. Si prefieres que lo contabilice Slurm (o el run es muy grande), pásalo por un job; puede esperar en la cola si el nodo está lleno:

```bash
srun --cpus-per-task=1 --mem=4G --time=00:30:00 bash server/export_run.sh ~/logdir/r2dreamer/presentacion/<run>
```

Variantes:

| Comando | Qué incluye | Tamaño aprox. |
|---|---|---|
| `bash server/export_run.sh <run>` | todo: TensorBoard **y** modelo | ~500 MB a 450k pasos |
| `... <run> --sin-eventos` | solo para probar el modelo (sin TensorBoard) | ~115 MB |
| `... <run> --solo-tensorboard` | solo curvas y videos (sin `latest.pt`) | ~400 MB |
| `... <run> --con-replay` | además el buffer (para diagnósticos que lo leen) | decenas de GB |

Si el run sigue entrenando, se exporta lo último guardado y avisa de que el final de los archivos puede quedar cortado (es inofensivo). Para ver la carpeta de runs: `ls ~/logdir/r2dreamer/presentacion/`.

## 2. En tu PC: bajarlo

```bash
rsync -avP barto:~/exports/cookie_partial_hall5_view5_s0_expert_completo.tar.gz .
```

`rsync -P` retoma si se corta la conexión; `scp barto:~/exports/<archivo> .` también sirve. (`barto` es el alias de `~/.ssh/config` de la sección 1 de `servidor.md`.) Para comprobar que llegó bien, baja también el `.sha256` y corre `sha256sum -c <archivo>.sha256`.

Después, extrae:

```bash
tar -xzf cookie_partial_hall5_view5_s0_expert_completo.tar.gz
```

(En Windows 10/11, `tar` viene en PowerShell.) Queda una carpeta con el nombre del run.

## 3. Ver el run en TensorBoard

```bash
pip install tensorboard
tensorboard --logdir cookie_partial_hall5_view5_s0_expert
```

Abre <http://localhost:6006>. Para comparar varios runs, ponlos en carpetas separadas dentro de una carpeta padre y usa el padre en `--logdir`. Las etiquetas útiles: `episode/eval_score` (galletas del actor, la medida principal), `episode/score_actor`, `episode/score_random` y `episode/score_expert` (los puntajes por quién jugó el episodio), `data/episodes_*` (el reparto de episodios) y las pérdidas `train/loss/*`. Sin TensorBoard: `python3 progress.py --logdir <carpeta>` y `python3 compare_runs.py --root <carpeta_padre> --out comparacion`.

## 4. Probar el modelo de mundo aprendido

Necesitas, de la carpeta del run: **`latest.pt` y `.hydra/config.yaml`** (el paquete por defecto los trae), y el **código del repo en la misma versión** con la que se entrenó (`VERSION.txt` guarda el commit al exportar; si no, la rama que lanzaste; los archivos del modelo son `networks.py`, `rssm.py` y `dreamer.py`).

```bash
cd r2dreamer-cookie                     # tu copia del repo
git checkout <commit de VERSION.txt>    # o la misma rama
bash server/setup.sh                    # crea .venv e instala (una vez)
.venv/bin/pip install gradio            # solo para la página interactiva
```

**Explorador interactivo** (el de tu notebook `notebooks/explorar_modelo_mundo_expert5x5.ipynb`, extraído tal cual a `explorar_modelo.py` para correrlo sin Colab):

```bash
.venv/bin/python explorar_modelo.py --logdir /ruta/al/run_extraido
```

Abre una página (por defecto <http://localhost:7860>) donde mueves al agente con las flechas y ves la realidad, lo que el modelo entiende al ver, lo que sueña sin ver, y la recompensa, el valor y la política que asigna. Corre bien en CPU (`--device cpu`). Una prueba rápida sin interfaz, que guarda un PNG: `--prueba salida.png`.

**Video del agente jugando:**

```bash
.venv/bin/python watch.py --logdir /ruta/al/run_extraido --out agente.mp4
```

**En Colab:** sube la carpeta extraída a Drive y abre el notebook de exploración; cambia `NOMBRE_RUN` por el nombre de la carpeta.

> `explorar_modelo.py` es una copia de la celda 12 del notebook. Si cambias uno, cambia el otro. De él probé aquí el modo `--prueba`; la página de Gradio es el código de tu notebook sin modificar.

## Lista rápida

1. Servidor: `bash server/export_run.sh ~/logdir/r2dreamer/presentacion/<run>`
2. PC: `rsync -avP barto:~/exports/<archivo>.tar.gz .` y `tar -xzf <archivo>.tar.gz`
3. TensorBoard: `tensorboard --logdir <run>`
4. Modelo de mundo: mismo commit del repo + `explorar_modelo.py --logdir <run>`
