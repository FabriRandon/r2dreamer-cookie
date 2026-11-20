"""Trae de Google Drive los archivos de un run para explorarlo, dado el link o el id de su carpeta.

Sirve con carpetas de otra cuenta (basta con que la tuya tenga acceso): no hace falta montar Drive ni
agregar un acceso directo. Solo baja lo necesario para `explorar_modelo.py`:

    latest.pt (el agente), .hydra/ (la configuración), status.json y metrics.jsonl

y se salta lo que pesa gigas: los eventos de TensorBoard (`events.out.tfevents*`) y el buffer (`replay/`).

En Colab, antes de usarlo:

    from google.colab import auth
    auth.authenticate_user()
    import drive_run
    carpeta = drive_run.descargar("https://drive.google.com/drive/folders/<id>", "/content/run")

Fuera de Colab usa las credenciales por defecto de Google (`gcloud auth application-default login`).
Solo usa la librería estándar y, al conectarse, `google-api-python-client` y `google-auth`.
"""

import argparse
import pathlib
import re
import sys

CARPETA = "application/vnd.google-apps.folder"
# Lo mínimo para explorar el modelo; `con_eventos` agrega los de TensorBoard.
ARCHIVOS = ("latest.pt", "status.json", "metrics.jsonl")
CARPETA_CONFIG = ".hydra"
GRANDES = ("replay",)  # nunca se bajan: son gigas
MB = 1 << 20


def id_de(texto):
    """El id de una carpeta de Drive, dado su link (con o sin basura al final) o el id mismo."""
    texto = str(texto).strip()
    for patron in (r"/folders/([A-Za-z0-9_-]{10,})", r"[?&]id=([A-Za-z0-9_-]{10,})", r"^([A-Za-z0-9_-]{10,})$"):
        encontrado = re.search(patron, texto)
        if encontrado:
            return encontrado[1]
    raise ValueError(f"No encuentro el id de una carpeta de Drive en {texto!r}")


def servicio():
    """El cliente de la API de Drive, con las credenciales de la sesión (en Colab: tras auth.authenticate_user())."""
    try:
        from googleapiclient.discovery import build
    except ImportError as error:
        raise SystemExit("Falta google-api-python-client: pip install google-api-python-client google-auth") from error
    return build("drive", "v3", cache_discovery=False)


def hijos(drive, carpeta):
    """Los archivos y carpetas dentro de una carpeta: lista de dicts con id, name, mimeType y size."""
    encontrados, pagina = [], None
    while True:
        respuesta = drive.files().list(
            q=f"'{carpeta}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageSize=1000,
            pageToken=pagina,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        encontrados += respuesta.get("files", [])
        pagina = respuesta.get("nextPageToken")
        if not pagina:
            return encontrados


def buscar_runs(drive, carpeta, nombre="", profundidad=3):
    """Los runs dentro de una carpeta (o ella misma): los que tienen latest.pt. Lista de (nombre, id)."""
    contenido = hijos(drive, carpeta)
    if any(f["name"] == "latest.pt" and f["mimeType"] != CARPETA for f in contenido):
        return [(nombre or carpeta, carpeta)]
    runs = []
    if profundidad > 0:
        for sub in contenido:
            if sub["mimeType"] == CARPETA and sub["name"] not in GRANDES and not sub["name"].startswith("."):
                runs += buscar_runs(drive, sub["id"], sub["name"], profundidad - 1)
    return runs


def _bajar(drive, id_archivo, destino):
    """Descarga un archivo por trozos (los de cientos de MB no caben de una vez en memoria)."""
    from googleapiclient.http import MediaIoBaseDownload

    pedido = drive.files().get_media(fileId=id_archivo, supportsAllDrives=True)
    with open(destino, "wb") as salida:
        descarga = MediaIoBaseDownload(salida, pedido, chunksize=32 * MB)
        listo = False
        while not listo:
            estado, listo = descarga.next_chunk()
            if estado:
                print(f"    {int(estado.progress() * 100)} %", end="\r", flush=True)


def descargar_run(drive, carpeta, destino, con_eventos=False, bajar=None):
    """Baja a `destino` lo necesario de la carpeta de un run. Devuelve la carpeta local."""
    bajar = bajar or _bajar
    destino = pathlib.Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    quiero = set(ARCHIVOS)
    for archivo in hijos(drive, carpeta):
        es_carpeta = archivo["mimeType"] == CARPETA
        if es_carpeta and archivo["name"] == CARPETA_CONFIG:
            (destino / CARPETA_CONFIG).mkdir(exist_ok=True)
            for config in hijos(drive, archivo["id"]):
                _traer(drive, config, destino / CARPETA_CONFIG / config["name"], bajar)
        elif not es_carpeta and (archivo["name"] in quiero or (con_eventos and archivo["name"].startswith("events.out.tfevents"))):
            _traer(drive, archivo, destino / archivo["name"], bajar)
    faltan = [n for n in ("latest.pt", f"{CARPETA_CONFIG}/config.yaml") if not (destino / n).exists()]
    if faltan:
        raise FileNotFoundError(f"En esa carpeta de Drive no está {', '.join(faltan)}")
    return destino


def _traer(drive, archivo, ruta, bajar):
    tamano = int(archivo.get("size", 0))
    if ruta.exists() and tamano and ruta.stat().st_size == tamano:
        print(f"  {ruta.name}: ya estaba ({tamano / MB:.1f} MB)")
        return
    print(f"  {ruta.name}: bajando {tamano / MB:.1f} MB ...")
    bajar(drive, archivo["id"], ruta)


def descargar(enlace, destino, nombre=None, con_eventos=False, drive=None):
    """Baja el run de la carpeta de Drive `enlace` (link o id) a `destino/<nombre del run>`. Si la carpeta
    tiene varios runs hay que decir `nombre`; con uno solo se toma ese. Devuelve la carpeta local del run."""
    drive = drive or servicio()
    raiz = id_de(enlace)
    runs = buscar_runs(drive, raiz)
    if not runs:
        raise FileNotFoundError("No encontré ningún run (una carpeta con latest.pt) en esa carpeta de Drive")
    if nombre:
        runs = [r for r in runs if r[0] == nombre]
        if not runs:
            raise FileNotFoundError(f"No hay un run llamado {nombre!r}")
    if len(runs) > 1:
        raise ValueError("Hay varios runs; elige uno con nombre=...: " + ", ".join(r[0] for r in runs))
    nombre_run, id_run = runs[0]
    print(f"Run {nombre_run}:")
    return descargar_run(drive, id_run, pathlib.Path(destino) / nombre_run, con_eventos)


def main(argumentos=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("carpeta", help="link o id de la carpeta de Drive (un run, o una carpeta con runs)")
    parser.add_argument("--destino", default="run", help="dónde dejarlo (se crea <destino>/<nombre del run>)")
    parser.add_argument("--run", default=None, help="nombre del run, si la carpeta tiene varios")
    parser.add_argument("--con-eventos", action="store_true", help="bajar también los eventos de TensorBoard (pesan cientos de MB)")
    args = parser.parse_args(argumentos)
    carpeta = descargar(args.carpeta, args.destino, args.run, args.con_eventos)
    print(f"Listo: {carpeta}")
    return carpeta


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
