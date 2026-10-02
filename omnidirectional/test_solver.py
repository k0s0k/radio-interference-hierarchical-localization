"""Run with: python -m unittest omnidirectional.test_solver -v"""

import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .geometry import outer_disk, contains, observe_direction, enclosing_circle
from .simulator import TaskSimulator, make_scenario
from .environment import SearchEnvironment, heuristic_index
from .policy import CandidateActorCritic, tensor_observations


def source(point=(100, 0), channel=1):
    return dict(pos=point, channel=channel, kind="omni", reception=1000, cleared=False)


class GeometryTests(unittest.TestCase):
    def test_outer_disk_contains_boundary(self):
        polygon = outer_disk()
        for angle in np.linspace(0, 2 * math.pi, 2000):
            self.assertTrue(
                contains(polygon, 1800 * np.array([math.cos(angle), math.sin(angle)]))
            )

    def test_circle_including_collinear_and_triangle(self):
        for points in (
            np.array([[0.0, 0], [10.0, 0], [3.0, 0]]),
            np.array([[0.0, 0], [2.0, 0], [1.0, math.sqrt(3)]]),
        ):
            center, radius = enclosing_circle(points)
            self.assertTrue(np.all(np.linalg.norm(points - center, axis=1) <= radius))
        self.assertAlmostEqual(radius, 2 / math.sqrt(3), places=5)

    def test_quantized_bounded_measurements_keep_truth(self):
        rng = np.random.default_rng(5)
        for _ in range(60):
            angle = rng.uniform(0, 2 * math.pi)
            truth = rng.uniform(0, 1800) * np.array([math.cos(angle), math.sin(angle)])
            polygon = outer_disk()
            for _ in range(4):
                angle = rng.uniform(0, 2 * math.pi)
                point = truth + rng.uniform(30, 1499) * np.array(
                    [math.cos(angle), math.sin(angle)]
                )
                true_angle = math.degrees(math.atan2(*(truth - point)[::-1]))
                measured = round((true_angle + rng.choice([-1, 1])) % 360, 2) % 360
                polygon = observe_direction(polygon, point, measured)
                self.assertTrue(contains(polygon, truth))
                center, radius = enclosing_circle(polygon)
                self.assertLessEqual(np.linalg.norm(center - truth), radius + 1e-6)


class SimulatorTests(unittest.TestCase):
    def test_fixed_error_independent_of_visit_order(self):
        a, b = TaskSimulator([source()], seed=99), TaskSimulator([source()], seed=99)
        a._error((100, 200))
        a._error((300, 400))
        b._error((300, 400))
        b._error((100, 200))
        self.assertEqual(a._error((100, 200)), b._error((100, 200)))

    def test_time_and_clear_channel_semantics(self):
        sim = TaskSimulator([source(channel=2)])
        sim.enter()
        self.assertTrue(sim.clear((80, 0), 2))
        self.assertAlmostEqual(sim.time, 21)
        self.assertEqual(sim.robot_channel, 1)
        sim.measure((80, 0), 2)
        self.assertEqual(sim.robot_channel, 2)
        self.assertAlmostEqual(sim.time, 27)

    def test_same_point_reading_and_near(self):
        sim = TaskSimulator([source()], seed=8)
        sim.enter()
        a, b = sim.measure((0, 0), 1), sim.measure((0, 0), 1)
        self.assertEqual(a, b)
        self.assertEqual(sim.measure((95, 0), 1)["type"], "near")

    def test_wall_time_uses_clock_and_budget_is_enforced(self):
        with patch("omnidirectional.simulator.time.perf_counter", return_value=10.0):
            sim = TaskSimulator([source()], max_real=2)
            sim.enter()
        with patch("omnidirectional.simulator.time.perf_counter", return_value=13.0):
            self.assertEqual(sim.real_time, 3)
            self.assertEqual(sim.measure((0, 0), 1)["type"], "closed")
        sim = TaskSimulator([source()], max_virtual=4)
        sim.enter()
        self.assertEqual(sim.measure((0, 0), 1)["type"], "closed")
        self.assertEqual(sim.time, 0)


class EnvironmentTests(unittest.TestCase):
    def test_no_truth_access_and_success_on_difficult_layouts(self):
        for distribution in ("uniform", "boundary", "cluster", "collinear"):
            env = SearchEnvironment()
            env.reset(seed=45, distribution=distribution, error_mode="extreme")
            with patch.object(
                env.sim, "outcome", side_effect=AssertionError("truth leaked")
            ):
                while not env.done:
                    env.step(heuristic_index(env.actions))
            self.assertTrue(env.success_certificate, (distribution, env.summary()))
            self.assertTrue(env.sim.outcome()["all_clear"])
            self.assertEqual(env.sim.metrics["clear_failure"], 0)

    def test_certified_clear_contains_all_vertices(self):
        env = SearchEnvironment()
        env.reset(seed=134)
        while not env.done:
            for a in env.actions:
                if a.kind == "clear":
                    distances = np.linalg.norm(
                        env.beliefs[a.channel].polygon - a.point, axis=1
                    )
                    self.assertTrue(np.all(distances < 20))
            env.step(heuristic_index(env.actions))

    def test_padding_is_not_an_action(self):
        env = SearchEnvironment()
        observation = env.reset(seed=3)
        self.assertEqual(int(observation["mask"].sum()), len(env.actions))
        with self.assertRaises(ValueError):
            env.step(len(env.actions))

    def test_policy_mask_and_candidate_permutation(self):
        torch.set_num_threads(1)
        env = SearchEnvironment()
        observation = env.reset(seed=4)
        model = CandidateActorCritic()
        data = tensor_observations(observation)
        distribution, value = model(data)
        self.assertEqual(float(distribution.probs[~data["mask"]].sum()), 0)
        permutation = torch.randperm(data["mask"].shape[1])
        reordered = dict(
            global_features=data["global_features"],
            candidates=data["candidates"][:, permutation],
            mask=data["mask"][:, permutation],
        )
        second, second_value = model(reordered)
        self.assertTrue(
            torch.allclose(distribution.probs[:, permutation], second.probs, atol=1e-6)
        )
        self.assertTrue(torch.allclose(value, second_value, atol=1e-6))
        loss = (
            -distribution.log_prob(torch.tensor([heuristic_index(env.actions)])).mean()
            + value.square().mean()
        )
        loss.backward()
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )


if __name__ == "__main__":
    unittest.main()
