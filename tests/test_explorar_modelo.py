"""explorar_modelo.py: encontrar el run, avisar bien cuando algo falta y recorrer el modelo de mundo."""

from pathlib import Path

import pytest

pytest.importorskip("cookie_env")
pytest.importorskip("minigrid")

import explorar_modelo as em  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def make_run(folder, obstacles="none", task="cookie_partial_hall5_view5"):
    """Un run de mentira: la configuración de Hydra y un checkpoint con un modelo sin entrenar."""
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from dreamer import Dreamer
    from envs import make_env

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        config = compose(config_name="configs", overrides=[
            "env=cookie", f"env.task={task}", f"env.obstacles={obstacles}", "model.rep_loss=dreamer", "device=cpu",
        ])
    (folder / ".hydra").mkdir(parents=True)
    OmegaConf.save(config, folder / ".hydra" / "config.yaml")  # sin resolver, como lo deja Hydra
    config.model.compile = False
    env = make_env(config.env, 0)
    agent = Dreamer(config.model, env.observation_space, env.action_space)
    torch.save({"step": 1234, "agent_state_dict": agent.state_dict()}, folder / "latest.pt")
    return folder


def test_finding_the_run_inside_a_folder(tmp_path):
    (tmp_path / "obstaculos5x5" / "distinct_s0").mkdir(parents=True)
    (tmp_path / "obstaculos5x5" / "distinct_s0" / "latest.pt").write_bytes(b"x")
    assert em.resolver_run(tmp_path / "obstaculos5x5").name == "distinct_s0"  # el único que hay adentro
    assert em.resolver_run(tmp_path).name == "distinct_s0"  # también a dos niveles
    assert em.resolver_run(tmp_path / "obstaculos5x5" / "distinct_s0").name == "distinct_s0"  # el run mismo
    assert em.resolver_run(tmp_path, "distinct_s0").name == "distinct_s0"
    (tmp_path / "obstaculos5x5" / "same_s0").mkdir()
    (tmp_path / "obstaculos5x5" / "same_s0" / "latest.pt").write_bytes(b"x")
    with pytest.raises(SystemExit, match="distinct_s0, same_s0"):
        em.resolver_run(tmp_path / "obstaculos5x5")  # varios: hay que elegir
    assert em.resolver_run(tmp_path / "obstaculos5x5", "same_s0").name == "same_s0"
    with pytest.raises(SystemExit, match="no encuentro ningún run"):
        em.resolver_run(tmp_path / "obstaculos5x5", "otro")
    with pytest.raises(SystemExit, match="No existe"):
        em.resolver_run(tmp_path / "nada")


def test_a_run_without_its_files_says_what_is_missing(tmp_path):
    (tmp_path / "latest.pt").write_bytes(b"x")
    with pytest.raises(SystemExit, match="config.yaml"):
        em.cargar(tmp_path, "cpu")


def test_an_environment_this_code_cannot_build_says_so(tmp_path):
    run = make_run(tmp_path / "run")
    config = (run / ".hydra" / "config.yaml").read_text().replace("obstacles: none", "obstacles: laberinto")
    (run / ".hydra" / "config.yaml").write_text(config)
    with pytest.raises(SystemExit, match="No pude construir el ambiente.*laberinto"):
        em.cargar(run, "cpu")


def test_a_checkpoint_of_another_model_says_so(tmp_path):
    import torch

    run = make_run(tmp_path / "run")
    torch.save({"step": 1, "agent_state_dict": {"extra": torch.zeros(1)}}, run / "latest.pt")
    with pytest.raises(SystemExit, match="no calza con el modelo"):
        em.cargar(run, "cpu")


@pytest.mark.parametrize("obstacles,text", [("none", "ninguno"), ("distinct", "distinct"), ("1", "1 por sala")])
def test_walking_through_the_world_model_of_a_run(tmp_path, obstacles, text):
    # `distinct` es el modo del run de Drive (obstaculos5x5/distinct_s0); el 1 son los aleatorios.
    run = make_run(tmp_path / "run", obstacles)
    exp = em.Explorador(em.resolver_run(tmp_path), "cpu")
    assert text in exp.descripcion() and "vista 5×5" in exp.descripcion() and "1,234 pasos" in exp.descripcion()
    exp.reiniciar(3)
    for accion in (2, 2, 0, 2, 1, 2):
        exp.paso(accion)
    exp.sincronizar()
    vistas = exp.vistas()
    assert vistas["mapa"].shape[2] == 3 and vistas["real"].shape == vistas["recon"].shape == vistas["sueno"].shape
    assert exp.imaginar_actor(4).ndim == 3 and "Paso real 6" in vistas["info"]
    assert run.exists()


def test_the_page_can_be_built_if_gradio_is_installed(tmp_path):
    pytest.importorskip("gradio")
    make_run(tmp_path / "run", "distinct")
    exp = em.Explorador(tmp_path / "run", "cpu")
    demo, extra = em.interfaz(exp)
    assert demo is not None and isinstance(extra, dict)
