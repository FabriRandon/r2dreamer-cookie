"""Check that this machine can train the cookie domain.

    python server/check_env.py [--no-compile]

Run it with the repo's venv (server/setup.sh does). Exits with an error code if
something that training needs is missing.
"""

import argparse
import os
import pathlib
import shutil
import sys
import sysconfig

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

problems = []


def report(ok, text, hint=""):
    print(("  ok  " if ok else " FALLA") + f" {text}")
    if not ok:
        problems.append(text + (f" -> {hint}" if hint else ""))
        if hint:
            print(f"        {hint}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-compile", action="store_true", help="skip the torch.compile test")
    args = parser.parse_args()

    import torch

    print(f"torch {torch.__version__}, CUDA del build: {torch.version.cuda}")
    cuda = torch.cuda.is_available()
    report(
        cuda,
        "torch ve una GPU",
        "Revisa `nvidia-smi` y el driver. Con un driver antiguo reinstala con TORCH_INDEX=https://download.pytorch.org/whl/cu126",
    )
    if cuda:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"      GPU visible {i}: {p.name}, {p.total_memory / 2**30:.0f} GiB")
        a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        report(bool(torch.isfinite((a @ a).sum())), "float16 en la GPU (el entrenamiento usa autocast float16)")

    from envs.cookie import Cookie

    for task, view in (("full_hall5", None), ("partial_hall5_view5", 5), ("partial_hall5_view3", 3)):
        env = Cookie(task, size=(64, 64))
        obs = env.reset()
        grid = env._env.unwrapped
        ok = obs["image"].shape == (64, 64, 3) and env.action_space.n == 3 and (view is None or grid.agent_view_size == view)
        report(ok, f"cookie_{task}: mapa {grid.height}x{grid.width}, partida de {grid.max_steps} pasos, 3 acciones")
        env.step(2)

    try:
        from moviepy.editor import ImageSequenceClip  # noqa: F401

        report(True, "moviepy importa (videos de TensorBoard y watch.py)")
    except Exception as e:  # noqa: BLE001
        report(False, f"moviepy no importa: {e}")

    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    free = shutil.disk_usage(os.path.expanduser("~")).free / 2**30
    print(f"      RAM del servidor: {ram:.0f} GiB (cada run con buffer en RAM usa ~11 GiB). Disco libre en ~: {free:.0f} GiB")

    if not args.no_compile and cuda:
        # model.compile=True builds kernels with Triton, which needs a C compiler
        # and the Python headers.
        report(shutil.which("gcc") is not None or shutil.which("cc") is not None, "hay compilador de C (gcc)", "Sin él, usa model.compile=False (launch.py --no-compile)")
        header = pathlib.Path(sysconfig.get_paths()["include"]) / "Python.h"
        report(header.exists(), "existen los headers de Python (Python.h)", "Falta python3.11-dev; o usa model.compile=False (launch.py --no-compile)")
        try:
            f = torch.compile(lambda x: torch.sin(x) * 2 + 1, mode="reduce-overhead")
            x = torch.randn(1024, device="cuda")
            torch.testing.assert_close(f(x), torch.sin(x) * 2 + 1)
            report(True, "torch.compile funciona en la GPU")
        except Exception as e:  # noqa: BLE001
            report(False, f"torch.compile falló: {type(e).__name__}: {str(e)[:200]}", "Usa model.compile=False (launch.py --no-compile)")

    print()
    if problems:
        print("Problemas:\n  - " + "\n  - ".join(problems))
        sys.exit(1)
    print("Todo en orden.")


if __name__ == "__main__":
    main()
