"""The "only Dreamer" experiments (obstacles or not, 5x5 or 3x3) must collect every episode with the actor:
no random agent and no expert. The Hydra config the launcher builds is composed the way train.py does."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT))

import launch  # noqa: E402
from test_server_launch import args_for  # noqa: E402

try:
    from hydra import compose, initialize_config_dir
    import trainer
except ImportError:  # no hydra / torch: the launcher's own tests still cover the overrides
    compose = None

ACTOR_ONLY = ("exp3-obstaculos", "exp3-sin-obstaculos", "exp4-obstaculos", "exp4-sin-obstaculos",
              "exp3-hall11-obstaculos", "exp3-hall11-sin-obstaculos")


@unittest.skipIf(compose is None, "needs hydra and torch")
class ActorOnlyRuns(unittest.TestCase):
    def config(self, name):
        overrides, *_ = launch.build(name, args_for())
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
            return compose(config_name="configs", overrides=overrides)

    def test_every_episode_belongs_to_the_actor(self):
        for name in ACTOR_ONLY:
            with self.subTest(name):
                t = self.config(name).trainer
                self.assertFalse(t.random_policy)
                self.assertEqual([t.data_mix[s] for s in ("actor", "random", "expert")], [0.0, 0.0, 0.0])
                mix = trainer.DataMix(t.data_mix, random_policy=bool(t.random_policy))
                self.assertEqual(mix.probs.tolist(), [1.0, 0.0, 0.0])  # all the actor's
                self.assertEqual(set(mix.sample(1000).tolist()), {trainer.ACTOR})

    def test_the_actors_data_is_what_the_obstacles_flag_changes(self):
        # With and without obstacles the collection is the same; only the environment differs.
        for view in (3, 5):
            with self.subTest(view):
                con, sin = self.config(f"exp{'3' if view == 5 else '4'}-obstaculos"), self.config(f"exp{'3' if view == 5 else '4'}-sin-obstaculos")
                self.assertEqual(con.trainer.data_mix, sin.trainer.data_mix)
                self.assertEqual((con.env.obstacles, sin.env.obstacles), (1, 0))


if __name__ == "__main__":
    unittest.main()
