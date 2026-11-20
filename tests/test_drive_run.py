"""drive_run.py: traer de Drive el run para explorarlo, probado con un Drive de mentira (sin red)."""

import re

import pytest

import drive_run

FOLDER = "application/vnd.google-apps.folder"
URL = "https://drive.google.com/drive/folders/11bZVL6RSUcWfm5tLzD28foWVQgtRHtMa"


def file(id, name, size=10, kind="application/octet-stream"):
    return {"id": id, "name": name, "mimeType": kind, "size": str(size)}


def folder(id, name):
    return {"id": id, "name": name, "mimeType": FOLDER}


# El árbol real de la carpeta de la corrida (nombres y estructura de lo que dejó el entrenamiento en Drive).
TREE = {
    "root": [folder("run", "distinct_s0")],
    "run": [
        file("log", "console.log", 331097), file("met", "metrics.jsonl", 470824),
        file("ev", "events.out.tfevents.1795065508.09fd9bad6c1b.2025.0", 792839101),
        file("st", "status.json", 115), file("pt", "latest.pt", 129420811),
        folder("hy", ".hydra"), folder("rp", "replay"),
    ],
    "hy": [file("hyd", "hydra.yaml", 3985), file("ov", "overrides.yaml", 531), file("cfg", "config.yaml", 4998)],
    "rp": [folder("sto", "storage"), file("x", "meta.json")],
    "sto": [file("img", "image.memmap", 6144000000)],
}


class FakeDrive:
    """Lo mínimo de la API de Drive que usa drive_run: files().list(q=...).execute(), paginando de a `page`."""

    def __init__(self, tree, page=3):
        self.tree, self.page, self.calls = tree, page, 0

    def files(self):
        return self

    def list(self, q, pageToken=None, **kwargs):
        parent = re.search(r"'([^']+)' in parents", q)[1]
        items = self.tree.get(parent, [])
        start = int(pageToken or 0)
        chunk = items[start:start + self.page]
        out = {"files": chunk}
        if start + self.page < len(items):
            out["nextPageToken"] = str(start + self.page)
        self.calls += 1
        return type("Request", (), {"execute": lambda self_: out})()


@pytest.mark.parametrize("text,expected", [
    (URL, "11bZVL6RSUcWfm5tLzD28foWVQgtRHtMa"),
    (URL + "??}", "11bZVL6RSUcWfm5tLzD28foWVQgtRHtMa"),  # el link tal como se pegó
    ("https://drive.google.com/drive/u/2/folders/11bZVL6RSUcWfm5tLzD28foWVQgtRHtMa?usp=sharing", "11bZVL6RSUcWfm5tLzD28foWVQgtRHtMa"),
    ("https://drive.google.com/open?id=11bZVL6RSUcWfm5tLzD28foWVQgtRHtMa", "11bZVL6RSUcWfm5tLzD28foWVQgtRHtMa"),
    ("  11bZVL6RSUcWfm5tLzD28foWVQgtRHtMa  ", "11bZVL6RSUcWfm5tLzD28foWVQgtRHtMa"),
])
def test_the_folder_id_comes_out_of_a_link_or_an_id(text, expected):
    assert drive_run.id_de(text) == expected


def test_a_text_without_an_id_is_refused():
    with pytest.raises(ValueError):
        drive_run.id_de("no es una carpeta")


def test_the_run_is_found_inside_the_folder_and_pagination_is_followed():
    drive = FakeDrive(TREE, page=2)
    assert drive_run.buscar_runs(drive, "root") == [("distinct_s0", "run")]
    assert drive_run.buscar_runs(drive, "run") == [("run", "run")]  # the folder of the run itself
    assert len(drive_run.hijos(drive, "run")) == len(TREE["run"]) and drive.calls > 3  # several pages


def test_it_downloads_only_what_the_explorer_needs(tmp_path):
    bajados = []

    def bajar(drive, id_archivo, destino):
        bajados.append(id_archivo)
        destino.write_bytes(b"x" * 10)

    local = drive_run.descargar_run(FakeDrive(TREE), "run", tmp_path / "distinct_s0", bajar=bajar)
    assert sorted(bajados) == ["cfg", "hyd", "met", "ov", "pt", "st"]  # sin eventos, sin replay, sin console.log
    assert (local / "latest.pt").exists() and (local / ".hydra" / "config.yaml").exists()
    assert not list(local.glob("events*")) and not (local / "replay").exists()


def test_tensorboard_events_only_if_asked(tmp_path):
    bajados = []
    drive_run.descargar_run(FakeDrive(TREE), "run", tmp_path, con_eventos=True,
                            bajar=lambda d, i, p: (bajados.append(i), p.write_bytes(b"x" * 10)))
    assert "ev" in bajados and "rp" not in bajados and "img" not in bajados  # the replay never


def test_what_is_already_there_is_not_downloaded_again(tmp_path):
    tree = {"run": [file("pt", "latest.pt", 10), folder("hy", ".hydra")], "hy": [file("cfg", "config.yaml", 10)]}
    bajados = []
    bajar = lambda d, i, p: (bajados.append(i), p.write_bytes(b"x" * 10))  # noqa: E731
    drive_run.descargar_run(FakeDrive(tree), "run", tmp_path, bajar=bajar)
    drive_run.descargar_run(FakeDrive(tree), "run", tmp_path, bajar=bajar)
    assert bajados == ["pt", "cfg"]  # the second time they are the same size: nothing


def test_a_folder_without_the_checkpoint_or_the_config_is_an_error(tmp_path):
    tree = {"run": [file("met", "metrics.jsonl")]}
    with pytest.raises(FileNotFoundError, match="latest.pt"):
        drive_run.descargar_run(FakeDrive(tree), "run", tmp_path, bajar=lambda d, i, p: p.write_bytes(b"x"))


def test_descargar_picks_the_only_run_or_asks_which(tmp_path, monkeypatch):
    monkeypatch.setattr(drive_run, "_bajar", lambda d, i, p: p.write_bytes(b"x" * 10))
    local = drive_run.descargar(URL, tmp_path, drive=FakeDrive({**TREE, "11bZVL6RSUcWfm5tLzD28foWVQgtRHtMa": TREE["root"]}))
    assert local == tmp_path / "distinct_s0"
    two = {"root": [folder("a", "run_a"), folder("b", "run_b")], "a": [file("p", "latest.pt")], "b": [file("q", "latest.pt")]}
    with pytest.raises(ValueError, match="run_a, run_b"):
        drive_run.descargar("root0000000000", tmp_path, drive=FakeDrive({**two, "root0000000000": two["root"]}))
    with pytest.raises(FileNotFoundError):
        drive_run.descargar("root0000000000", tmp_path, nombre="otro", drive=FakeDrive({**two, "root0000000000": two["root"]}))
