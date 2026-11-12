"""Recorrer el modelo de mundo de un run del dominio cookie.

Abre una página (Gradio) donde se mueve al agente con botones o con las
flechas del teclado. Cada acción avanza dos cosas a la vez:

- el entorno real, y la reconstrucción del modelo cuando SÍ ve la observación
  (posterior: muestra qué retiene la representación);
- un sueño que NO ve nada desde la última sincronización (prior: muestra qué
  predice la dinámica).

Al lado aparecen la recompensa, el "continuar", el valor y la política que el
modelo asigna a cada estado, y una tira con lo que imagina el actor.

    python3 explorar_modelo.py --logdir /ruta/al/run            # abre la página
    python3 explorar_modelo.py --logdir /ruta/al/run --prueba out.png  # sin interfaz
"""

import argparse
import base64
import inspect
import io
import os
import pathlib
import random
import sys
import warnings

# Colab deja MPLBACKEND apuntando al backend inline del notebook, que no existe
# en el .venv de este script; matplotlib falla al importarse si no se cambia antes.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))

from dreamer import Dreamer  # noqa: E402
from envs import make_env  # noqa: E402

ACCIONES = ["izquierda", "derecha", "avanzar"]
LADO = 256  # tamaño en pantalla de las imágenes de 64x64


def cargar(logdir, device):
    config = OmegaConf.load(logdir / ".hydra" / "config.yaml")
    config.device = device
    config.model.compile = False
    env = make_env(config.env, 0)
    agent = Dreamer(config.model, env.observation_space, env.action_space).to(device)
    checkpoint = torch.load(logdir / "latest.pt", map_location=device, weights_only=False)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()
    return config, env, agent, int(checkpoint.get("step", 0))


def agrandar(img, lado=LADO):
    return np.asarray(Image.fromarray(img).resize((lado, lado), Image.NEAREST))


class Explorador:
    """Estado del recorrido: entorno real, posterior y sueño."""

    def __init__(self, logdir, device):
        self.device = torch.device(device)
        self.config, self.env, self.agent, self.pasos_entrenados = cargar(pathlib.Path(logdir), device)
        self.rssm = self.agent.rssm
        self.cookie = self.env.unwrapped
        self.cookie._env.unwrapped.tile_size = 24
        self.n_acciones = self.agent.act_dim
        self.nombres = ACCIONES if self.n_acciones == len(ACCIONES) else [str(i) for i in range(self.n_acciones)]
        self.muestrear = False
        self.reiniciar(0)

    # --- piezas del modelo -------------------------------------------------

    def _onehot(self, i):
        return F.one_hot(torch.tensor([int(i)], device=self.device), self.n_acciones).float()

    def _stoch(self, logit):
        dist = self.rssm.get_dist(logit)
        return dist.rsample() if self.muestrear else dist.mode

    def _embed(self, obs):
        data = {"image": torch.as_tensor(np.asarray(obs["image"]), device=self.device)[None]}
        return self.agent.encoder(self.agent.preprocess(data))

    def _posterior(self, estado, accion, obs):
        """Un paso viendo la observación (como rssm.obs_step)."""
        stoch, deter = estado
        deter = self.rssm._deter_net(stoch, deter, accion)
        logit = self.rssm._obs_net(torch.cat([deter, self._embed(obs)], -1))
        return self._stoch(logit), deter

    def _prior(self, estado, accion):
        """Un paso sin ver nada (como rssm.img_step)."""
        stoch, deter = estado
        deter = self.rssm._deter_net(stoch, deter, accion)
        return self._stoch(self.rssm._img_net(deter)), deter

    @torch.no_grad()
    def _decodificar(self, estado):
        stoch, deter = estado
        img = self.agent.decoder(stoch[:, None], deter[:, None])["image"].mode()
        return (img[0, 0].clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8)

    @torch.no_grad()
    def _cabezas(self, estado):
        feat = self.rssm.get_feat(*estado)
        return {
            "r": float(self.agent.reward(feat).mode().item()),
            "c": float(self.agent.cont(feat).mean.item()),
            "v": float(self.agent.value(feat).mode().item()),
            "pi": self.agent.actor(feat).probs[0].float().cpu().numpy(),
        }

    # --- acciones del usuario ----------------------------------------------

    @torch.no_grad()
    def reiniciar(self, semilla):
        semilla = int(semilla)
        # cookie-env elige la esquina de la galleta con el random global.
        self.cookie._seed, self.cookie._seeded = semilla, False
        random.seed(semilla)
        self.obs = self.env.reset()
        cero = torch.zeros(1, self.n_acciones, device=self.device)
        self.post = self._posterior(self.rssm.initial(1), cero, self.obs)
        self.sueno = self.post
        self.paso_real, self.desde_sync, self.galletas, self.terminado = 0, 0, 0, False
        self.historia = []
        self._registrar(None, 0.0)

    @torch.no_grad()
    def sincronizar(self):
        self.sueno = self.post
        self.desde_sync = 0
        self._registrar(None, float("nan"))

    @torch.no_grad()
    def paso(self, i, mover_real=True):
        accion = self._onehot(i)
        self.sueno = self._prior(self.sueno, accion)
        self.desde_sync += 1
        recompensa = float("nan")
        if mover_real and not self.terminado:
            self.obs, recompensa, hecho, _ = self.env.step(accion[0].cpu().numpy())
            recompensa = float(recompensa)
            self.post = self._posterior(self.post, accion, self.obs)
            self.paso_real += 1
            self.galletas += int(recompensa > 0)
            self.terminado = bool(hecho)
        self._registrar(i, recompensa)

    @torch.no_grad()
    def accion_actor(self, mover_real=True):
        """La acción más probable del actor, desde el estado que se está mirando."""
        estado = self.post if mover_real else self.sueno
        return int(self._cabezas(estado)["pi"].argmax())

    @torch.no_grad()
    def imaginar_actor(self, horizonte=15):
        """Lo que entrena al actor: desde el posterior actual, el actor juega dentro del sueño."""
        estado, cuadros = self.post, []
        for _ in range(int(horizonte)):
            h = self._cabezas(estado)
            i = int(h["pi"].argmax())
            cuadros.append((self._decodificar(estado), h, i))
            estado = self._prior(estado, self._onehot(i))
        return self._tira(cuadros)

    # --- registro y dibujo -------------------------------------------------

    @torch.no_grad()
    def _registrar(self, accion, recompensa_real):
        hp, hs = self._cabezas(self.post), self._cabezas(self.sueno)
        real = np.asarray(self.obs["image"], dtype=np.float32) / 255
        error = float(((self._decodificar(self.sueno) / 255 - real) ** 2).mean())
        self.historia.append({"accion": accion, "r_real": recompensa_real, "post": hp, "sueno": hs, "error": error})

    @torch.no_grad()
    def vistas(self):
        real = np.asarray(self.obs["image"])
        recon = self._decodificar(self.post)
        sueno = self._decodificar(self.sueno)
        error = np.abs(sueno.astype(np.int16) - real.astype(np.int16)).astype(np.uint8)
        return {
            "mapa": self.cookie.render(),
            "real": agrandar(real),
            "recon": agrandar(recon),
            "sueno": agrandar(sueno),
            "error": agrandar(error),
            "info": self.resumen(),
            "grafico": self.grafico(),
        }

    def resumen(self):
        h = self.historia[-1]
        fila = lambda nombre, x: (  # noqa: E731
            f"| {nombre} | {x['r']:+.3f} | {x['c']:.2f} | {x['v']:+.2f} | "
            + " / ".join(f"{p:.2f}" for p in x["pi"])
            + " |"
        )
        ultima = "—" if h["accion"] is None else self.nombres[h["accion"]]
        r_real = "—" if np.isnan(h["r_real"]) else f"{h['r_real']:+.2f}"
        estado = "**terminó el episodio** (reinicia)" if self.terminado else ""
        return "\n".join([
            f"**Paso real {self.paso_real}** · galletas {self.galletas} · última acción: {ultima} · "
            f"recompensa real: {r_real} · el sueño lleva **{self.desde_sync} pasos sin ver** · "
            f"modelo entrenado {self.pasos_entrenados:,} pasos {estado}",
            "",
            "| estado | recompensa pred. | continuar | valor | actor π (" + " / ".join(self.nombres) + ") |",
            "|---|---|---|---|---|",
            fila("viendo (posterior)", h["post"]),
            fila(f"soñando ({self.desde_sync} sin ver)", h["sueno"]),
            "",
            f"Error del sueño frente a la vista real (MSE): {h['error']:.4f}",
        ])

    def grafico(self):
        hist = self.historia
        x = np.arange(len(hist))
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True)
        a1.plot(x, [h["post"]["r"] for h in hist], label="recompensa pred. (viendo)")
        a1.plot(x, [h["sueno"]["r"] for h in hist], label="recompensa pred. (soñando)", linestyle="--")
        reales = [h["r_real"] for h in hist]
        a1.scatter(x, reales, s=12, color="black", label="recompensa real", zorder=3)
        a1.legend(fontsize=7, loc="best")
        a1.set_ylabel("recompensa")
        a2.plot(x, [h["error"] for h in hist], color="tab:red", label="error del sueño (MSE)")
        a2.plot(x, [h["post"]["v"] for h in hist], color="tab:green", label="valor (viendo)")
        a2.legend(fontsize=7, loc="best")
        a2.set_xlabel("acción del recorrido")
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img

    def _tira(self, cuadros):
        # Filas de 5 cuadros, para que el texto se lea al ajustar la imagen al ancho.
        lado, alto_txt, por_fila = 160, 60, 5
        columnas = min(por_fila, len(cuadros))
        filas = -(-len(cuadros) // por_fila)
        tira = Image.new("RGB", (lado * columnas, (lado + alto_txt) * filas), (20, 20, 20))
        dibujo = ImageDraw.Draw(tira)
        fuente = ImageFont.load_default(size=14)
        for k, (img, h, i) in enumerate(cuadros):
            x, y = (k % por_fila) * lado, (k // por_fila) * (lado + alto_txt)
            tira.paste(Image.fromarray(img).resize((lado - 4, lado - 4), Image.NEAREST), (x + 2, y + 2))
            texto = f"t+{k}: {self.nombres[i]}\nr {h['r']:+.2f}   c {h['c']:.2f}\nv {h['v']:+.2f}"
            dibujo.multiline_text((x + 4, y + lado + 2), texto, fill=(230, 230, 230), font=fuente, spacing=3)
        return np.asarray(tira)


def a_html(img, titulo, pixelado):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    estilo = "width:100%;height:auto;border-radius:4px;" + ("image-rendering:pixelated;" if pixelado else "")
    return (
        f'<figure style="margin:0"><figcaption style="font-size:13px;margin-bottom:4px">{titulo}</figcaption>'
        f'<img src="{src}" style="{estilo}"></figure>'
    )


def grilla(imagenes, columnas):
    celdas = "".join(a_html(*x) for x in imagenes)
    return f'<div style="display:grid;grid-template-columns:{columnas};gap:12px;align-items:start">{celdas}</div>'


def interfaz(exp):
    import gradio as gr

    # Las imágenes van incrustadas como data URI dentro de HTML. gr.Image las sirve
    # por URL, y esas URL se rompen detrás del proxy de Colab (se ven vacías).
    def mostrar():
        v = exp.vistas()
        arriba = grilla([
            (v["mapa"], "Mapa real (lo iluminado es lo que ve el agente)", False),
            (v["grafico"], "Historia del recorrido", False),
        ], columnas="1fr 1fr")
        vistas = grilla([
            (v["real"], "Vista real", True),
            (v["recon"], "Viendo: lo que entiende (reconstrucción)", True),
            (v["sueno"], f"Soñando: lo que predice sin ver ({exp.desde_sync} pasos)", True),
            (v["error"], "|sueño − real|", True),
        ], columnas="repeat(4, 1fr)")
        return [arriba, vistas, v["info"]]

    def mover(i, mover_real, muestrear):
        exp.muestrear = bool(muestrear)
        exp.paso(i, mover_real)
        return mostrar()

    def actor(mover_real, muestrear):
        exp.muestrear = bool(muestrear)
        exp.paso(exp.accion_actor(mover_real), mover_real)
        return mostrar()

    def actor_n(n, mover_real, muestrear):
        exp.muestrear = bool(muestrear)
        for _ in range(int(n)):
            if mover_real and exp.terminado:
                break
            exp.paso(exp.accion_actor(mover_real), mover_real)
        return mostrar()

    def sincronizar():
        exp.sincronizar()
        return mostrar()

    def reiniciar(semilla, muestrear):
        exp.muestrear = bool(muestrear)
        exp.reiniciar(semilla)
        return mostrar()

    teclado = """
<script>
document.addEventListener('keydown', (e) => {
  const tag = (document.activeElement || {}).tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;
  const ids = {ArrowLeft: 'btn-izq', ArrowRight: 'btn-der', ArrowUp: 'btn-avz', s: 'btn-sync', a: 'btn-actor'};
  const el = ids[e.key] && document.getElementById(ids[e.key]);
  if (el) { e.preventDefault(); el.click(); }
});
</script>
"""
    # Gradio 6 recibe head en launch(); las versiones anteriores, en Blocks().
    head_en_launch = "head" in inspect.signature(gr.Blocks.launch).parameters
    with gr.Blocks(title="Modelo de mundo", **({} if head_en_launch else {"head": teclado})) as demo:
        gr.Markdown(
            "### Recorrer el modelo de mundo (versión 3: imágenes incrustadas)\n"
            "Flechas ← → ↑ (o los botones) mueven al agente. **S** sincroniza el sueño con la realidad, "
            "**A** juega la acción del actor. *Viendo* = el modelo recibe la imagen real en cada paso; "
            "*soñando* = solo recibe las acciones desde la última sincronización."
        )
        panel_arriba = gr.HTML()
        panel_vistas = gr.HTML()
        with gr.Row():
            b_izq = gr.Button("⟲ izquierda", elem_id="btn-izq")
            b_avz = gr.Button("↑ avanzar", elem_id="btn-avz", variant="primary")
            b_der = gr.Button("⟳ derecha", elem_id="btn-der")
        with gr.Row():
            b_actor = gr.Button("Acción del actor (A)", elem_id="btn-actor")
            n_actor = gr.Number(value=20, precision=0, label="pasos", minimum=1, maximum=500)
            b_actor_n = gr.Button("El actor juega N pasos")
            b_sync = gr.Button("Sincronizar sueño con la realidad (S)", elem_id="btn-sync")
        with gr.Row():
            mover_real = gr.Checkbox(value=True, label="Mover también el entorno real (si no, solo se sueña)")
            muestrear = gr.Checkbox(value=False, label="Muestrear el estado estocástico (si no, el más probable)")
            semilla = gr.Number(value=0, precision=0, label="semilla del episodio")
            b_reset = gr.Button("Reiniciar episodio")
        info = gr.Markdown()
        with gr.Accordion("Imaginación del actor (lo que usa para aprender)", open=False):
            horizonte = gr.Slider(1, 40, value=15, step=1, label="horizonte")
            b_imag = gr.Button("Imaginar desde el estado actual")
            tira = gr.HTML()

        salidas = [panel_arriba, panel_vistas, info]
        opciones = [mover_real, muestrear]
        for boton, i in ((b_izq, 0), (b_der, 1), (b_avz, 2)):
            boton.click(lambda m, s, i=i: mover(i, m, s), opciones, salidas)
        b_actor.click(actor, opciones, salidas)
        b_actor_n.click(actor_n, [n_actor, *opciones], salidas)
        b_sync.click(sincronizar, None, salidas)
        b_reset.click(reiniciar, [semilla, muestrear], salidas)
        b_imag.click(lambda h: grilla([(exp.imaginar_actor(h), "El actor jugando dentro del sueño", True)], "1fr"),
                     horizonte, tira)
        demo.load(mostrar, None, salidas)
    return demo, ({"head": teclado} if head_en_launch else {})


def prueba(exp, salida):
    """Recorrido corto sin interfaz, para comprobar que todo carga."""
    rng = np.random.default_rng(0)
    for _ in range(12):
        exp.paso(int(rng.integers(exp.n_acciones)))
    v = exp.vistas()
    fila = np.concatenate([v["real"], v["recon"], v["sueno"], v["error"]], 1)
    Image.fromarray(fila).save(salida)
    Image.fromarray(exp.imaginar_actor(8)).save(pathlib.Path(salida).with_suffix(".actor.png"))
    print(v["info"])
    print(f"Imágenes en {salida} y {pathlib.Path(salida).with_suffix('.actor.png')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logdir", required=True, type=pathlib.Path, help="carpeta del run con latest.pt y .hydra/")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="crear además un link público de Gradio")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prueba", type=pathlib.Path, default=None, help="no abre la página: guarda un PNG de prueba")
    args = parser.parse_args()

    exp = Explorador(args.logdir, args.device)
    if args.prueba:
        prueba(exp, args.prueba)
        return
    demo, extra = interfaz(exp)
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port, share=args.share, **extra
    )


if __name__ == "__main__":
    main()
