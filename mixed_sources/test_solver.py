import math
import unittest
from unittest.mock import patch
import numpy as np
import torch

from .belief import ChannelBelief
from .coverage import directional_cover, project_to_positive_hull
from .environment import SearchEnvironment, heuristic_index
from .simulator import TaskSimulator, make_scenario
from .policy import CandidateActorCritic, tensor_observations, load_policy


def source(point=(100.0, 0.0), heading=180.0):
    return dict(
        pos=point, channel=1, kind="dir", dir=heading, reception=1000.0, cleared=False
    )


class CoverageTests(unittest.TestCase):
    def test_boundary_and_arbitrary_headings_have_visible_station(self):
        stations, triangles = directional_cover(950.0)
        self.assertEqual(len(stations), 31)
        rng = np.random.default_rng(32)
        angles = np.concatenate(
            (np.linspace(0, 2 * np.pi, 720), rng.uniform(0, 2 * np.pi, 600))
        )
        radii = np.concatenate((np.full(720, 1800.0), 1800 * np.sqrt(rng.random(600))))
        points = radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
        headings = np.column_stack(
            (np.cos(np.arange(48) * np.pi / 24), np.sin(np.arange(48) * np.pi / 24))
        )
        delta = stations[None, :, :] - points[:, None, :]
        visible = (np.linalg.norm(delta, axis=-1) <= 1000)[:, :, None] & (
            np.einsum("psd,hd->psh", delta, headings) >= -1e-8
        )
        self.assertTrue(visible.any(axis=1).all())
        vertices = stations[triangles]
        self.assertLessEqual(
            np.linalg.norm(vertices - np.roll(vertices, 1, axis=1), axis=-1).max(),
            950 + 1e-8,
        )

    def test_unsafe_spacing_rejected(self):
        with self.assertRaises(ValueError):
            directional_cover(1001)

    def test_positive_hull_remains_detectable(self):
        sim = TaskSimulator([source(heading=180.0)])
        sim.enter()
        sites = [(-100.0, -100.0), (-100.0, 100.0), (0.0, 0.0)]
        for point in sites:
            self.assertEqual(sim.measure(point, 1)["type"], "direction")
        p = project_to_positive_hull(np.array([100.0, 80.0]), sites)
        self.assertNotEqual(sim.measure(p, 1)["type"], "no_signal")


class ObservationTests(unittest.TestCase):
    def test_backside_near_and_clear(self):
        sim = TaskSimulator([source(heading=90.0)])
        sim.enter()
        self.assertEqual(sim.measure((100.0, -1.0), 1)["type"], "no_signal")
        self.assertEqual(sim.measure((99.0, 0.0), 1)["type"], "near")
        self.assertTrue(sim.clear((100.0, -1.0), 1))

    def test_negative_shifts_heading_not_certified_position_set(self):
        sim = TaskSimulator([source()])
        sim.enter()
        b = ChannelBelief()
        for point in ((0.0, 0.0), (0.0, 100.0)):
            b.update(point, sim.measure(point, 1))
        previous = b.polygon.copy()
        b.update((200.0, 0.0), sim.measure((200.0, 0.0), 1))
        self.assertTrue(np.array_equal(previous, b.polygon))
        probability, pdir, direction, empty = b.reception_features(
            [(0.0, 50.0), (250.0, 0.0)]
        )
        self.assertFalse(empty)
        self.assertAlmostEqual(pdir, 1.0, places=6)
        self.assertGreater(probability[0], 0.99)
        self.assertLess(probability[1], 0.01)

    def test_continuous_radius_interval_not_three_discrete_radii(self):
        b = ChannelBelief()
        b.seen = True
        b.polygon = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        b.positive = [((1100.0, 0.0), 180.0)]
        b.negative = [(1200.0, 0.0)]
        b._refresh()
        c = b.cloud
        self.assertFalse(c["empty"])
        omni = ~c["directional"]
        self.assertTrue(omni.any())
        self.assertTrue(np.all(c["lower"][omni] < 1150))
        self.assertTrue(np.all(c["upper"][omni] > 1150))

    def test_empty_sample_cloud_is_not_a_clearance_certificate(self):
        b = ChannelBelief()
        b.seen = True
        b.positive = [((0.0, 0.0), 0.0)]
        b.negative = [(0.0, 0.0)]
        probability, _, _, empty = b.reception_features([(100.0, 0.0)])
        self.assertTrue(empty)
        self.assertEqual(probability[0], 0.5)
        self.assertGreater(b.radius, 20)


class EndToEndTests(unittest.TestCase):
    def test_mixed_stress_no_hidden_truth_and_no_failed_clears(self):
        for index, distribution in enumerate(
            (
                "uniform",
                "boundary_outward",
                "cluster_away",
                "collinear",
                "high_directional",
            )
        ):
            env = SearchEnvironment()
            env.reset(122000 + index, distribution=distribution, error_mode="extreme")
            with patch.object(
                env.sim, "outcome", side_effect=AssertionError("truth leaked")
            ):
                while not env.done:
                    for action in env.actions:
                        if action.kind == "clear":
                            self.assertLess(
                                np.linalg.norm(
                                    env.beliefs[action.channel].polygon - action.point,
                                    axis=1,
                                ).max(),
                                20,
                            )
                    env.step(heuristic_index(env.actions))
            self.assertTrue(env.success_certificate, env.summary())
            self.assertTrue(env.sim.outcome()["all_clear"])
            self.assertEqual(env.sim.metrics["clear_failure"], 0)

    def test_missing_channels_not_removed_by_particle_estimates(self):
        env = SearchEnvironment()
        env.reset(122007)
        env.step(heuristic_index(env.actions))
        self.assertFalse(env.success_certificate)
        self.assertGreater(len(env.unknown_channels()), 0)
        self.assertFalse(env.visited.all())

    def test_q4_policy_shape_mask_and_checkpoint_isolation(self):
        torch.set_num_threads(1)
        env = SearchEnvironment()
        obs = env.reset(122001)
        self.assertEqual(obs["candidates"].shape, (96, 29))
        model = CandidateActorCritic()
        t = tensor_observations(obs)
        distribution, value = model(t)
        self.assertEqual(float(distribution.probs[~t["mask"]].sum()), 0.0)
        (
            -distribution.log_prob(torch.tensor([heuristic_index(env.actions)])).mean()
            + value.square().mean()
        ).backward()
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )
        with self.assertRaises(ValueError):
            load_policy("omnidirectional/models/seed0/best_ppo.pt")


if __name__ == "__main__":
    unittest.main()
