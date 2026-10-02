"""Frozen Q4 models, paired held-out cases and official-definition metrics."""

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from omnidirectional.baseline_adapter import load_baseline
from .environment import SearchEnvironment, heuristic_index
from .policy import load_policy
from .simulator import TaskSimulator, make_scenario


def run_baseline(module, sim, corrected=False):
    previous_order, previous_fine = module.nn_order, module.fine_search_and_clear
    if corrected:

        def dynamic_order(points, _start):
            todo = list(points)
            while todo:
                i = min(
                    range(len(todo)),
                    key=lambda i: np.linalg.norm(np.asarray(todo[i]) - sim.robot_pos),
                )
                yield todo.pop(i)

        module.nn_order = dynamic_order
        code = inspect.getsource(previous_fine)
        if "90.0 * k" not in code:
            raise RuntimeError("Original local-search code changed; review bridge")
        code = code.replace("90.0 * k", "(360.0 / FINE_AZIM) * k")
        exec(compile(code, "baseline_corrected_fine", "exec"), module.__dict__)
    try:
        module.solve(sim, cover=350.0, directional=True)
        sim.exit()
    finally:
        module.nn_order, module.fine_search_and_clear = previous_order, previous_fine


def run_policy(env, seed, policy=None, distribution="uniform", error_mode="hash"):
    observation = env.reset(seed, distribution=distribution, error_mode=error_mode)
    inference = 0.0
    while not env.done:
        if policy is None:
            index = heuristic_index(env.actions)
        else:
            started = time.perf_counter()
            index, _ = policy.choose(observation)
            inference += time.perf_counter() - started
        observation, _, _, _ = env.step(index)
    return dict(**env.sim.outcome(), **env.summary(), inference_s=inference)


def summarize(rows):
    summary = []
    for group in dict.fromkeys(r["group"] for r in rows):
        for method in dict.fromkeys(r["method"] for r in rows):
            items = [r for r in rows if r["group"] == group and r["method"] == method]
            if not items:
                continue
            defined = [
                r["avg_localization_clear_s"]
                for r in items
                if r["avg_localization_clear_s"] is not None
            ]
            summary.append(
                dict(
                    group=group,
                    method=method,
                    n=len(items),
                    all_clear_count=sum(r["all_clear"] for r in items),
                    mean_cleared=float(np.mean([r["n_cleared"] for r in items])),
                    mean_source_fraction=float(
                        np.mean([r["n_cleared"] / r["n_sources"] for r in items])
                    ),
                    mean_virtual_s=float(np.mean([r["virtual_s"] for r in items])),
                    p95_virtual_s=float(
                        np.quantile([r["virtual_s"] for r in items], 0.95)
                    ),
                    max_virtual_s=max(r["virtual_s"] for r in items),
                    mean_per_source_s=float(np.mean(defined)) if defined else None,
                    undefined_per_source=len(items) - len(defined),
                    mean_wall_s=float(np.mean([r["wall_s"] for r in items])),
                    max_wall_s=max(r["wall_s"] for r in items),
                    mean_inference_s=float(np.mean([r["inference_s"] for r in items])),
                    mean_measurements=float(
                        np.mean([r["measurements"] for r in items])
                    ),
                    mean_no_signal=float(np.mean([r["no_signal"] for r in items])),
                    total_failed_clears=sum(r["clear_failure"] for r in items),
                    fallback_episodes=sum(r["fallback_steps"] > 0 for r in items),
                )
            )
    return summary


def compare(rows, reference, method):
    left = {
        r["seed"]: r
        for r in rows
        if r["method"] == reference and r["group"] == "uniform"
    }
    right = {
        r["seed"]: r for r in rows if r["method"] == method and r["group"] == "uniform"
    }
    seeds = sorted(set(left) & set(right))
    good = [s for s in seeds if left[s]["all_clear"] and right[s]["all_clear"]]
    result = dict(
        reference=reference,
        method=method,
        paired_success_n=len(good),
        total_n=len(seeds),
    )
    if good:
        indices = np.random.default_rng(9404).integers(0, len(good), (2000, len(good)))
        for metric in ("virtual_s", "avg_localization_clear_s"):
            x = np.array([left[s][metric] for s in good])
            y = np.array([right[s][metric] for s in good])
            samples = 100 * (1 - y[indices].mean(1) / x[indices].mean(1))
            result[metric] = dict(
                reduction_pct=float(100 * (1 - y.mean() / x.mean())),
                ci95_pct=np.quantile(samples, [0.025, 0.975]).tolist(),
                faster_count=int((y < x).sum()),
            )
    return result


# 报告排版源码已注释保存在“辅助材料/注释代码”。


def evaluate(args):
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.models).glob("seed*/best_ppo.pt"))
    if not paths:
        raise FileNotFoundError("Train Q4 checkpoints first")
    models, metadata = {}, {}
    for path in paths:
        name = "ppo_" + path.parent.name
        models[name], metadata[name] = load_policy(path)
    selected = min(
        models,
        key=lambda n: (1 - metadata[n]["validation"]["all_clear"]) * 1e6
        + metadata[n]["validation"]["mean_per_source_s"],
    )
    primary_path = next(p for p in paths if "ppo_" + p.parent.name == selected)
    models["bc"], _ = load_policy(primary_path.parent / "bc.pt")
    cases = [
        dict(
            seed=args.first_seed + i,
            group="uniform",
            distribution="uniform",
            error_mode="hash",
        )
        for i in range(args.n)
    ]
    for j, (distribution, error) in enumerate(
        (
            ("boundary_outward", "extreme"),
            ("cluster_away", "smooth"),
            ("collinear", "extreme"),
            ("high_directional", "smooth"),
        )
    ):
        cases += [
            dict(
                seed=150000 + 1000 * j + i,
                group=distribution,
                distribution=distribution,
                error_mode=error,
            )
            for i in range(args.stress_n)
        ]
    source_paths = list(Path(__file__).parent.glob("*.py")) + [
        Path(__file__).resolve().parent.parent / "omnidirectional" / name
        for name in (
            "geometry.py",
            "simulator.py",
            "environment.py",
            "baseline_adapter.py",
        )
    ]
    manifest = dict(
        selected=selected,
        selection="validation_all_clear_then_mean_per_source",
        cases=cases,
        checkpoints={
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        source_files={
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        runtime=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            torch=str(torch.__version__),
            device="cpu",
            threads=args.threads,
        ),
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    methods = [
        "baseline_fixed",
        "baseline_dynamic_fixed",
        "geometry_no_visibility",
        "geometry_visibility",
        "bc",
    ] + sorted(n for n in models if n != "bc")
    baseline = load_baseline()
    env = SearchEnvironment()
    no_visibility = SearchEnvironment(use_visibility=False)
    rows, examples = [], []
    started = time.perf_counter()
    for i, case in enumerate(cases):
        for method in methods:
            if method.startswith("baseline"):
                sim = TaskSimulator(
                    make_scenario(case["seed"], case["distribution"]),
                    seed=case["seed"],
                    error_mode=case["error_mode"],
                    record=False,
                )
                run_baseline(
                    baseline, sim, corrected=method == "baseline_dynamic_fixed"
                )
                result = dict(
                    **sim.outcome(),
                    steps=0,
                    fallback_steps=0,
                    certificate=None,
                    failure_reason="",
                    inference_s=0.0,
                )
            else:
                active = no_visibility if method == "geometry_no_visibility" else env
                active.record = bool(method == selected and i < 3)
                result = run_policy(
                    active,
                    case["seed"],
                    models.get(method),
                    case["distribution"],
                    case["error_mode"],
                )
                if active.record:
                    demo = dict(
                        seed=case["seed"],
                        policy=selected,
                        outcome=result,
                        macro_actions=active.macro_log,
                        requests=active.sim.log,
                    )
                    (output / f"case_{case['seed']}.json").write_text(
                        json.dumps(demo, indent=2), encoding="utf-8"
                    )
                    if i == 0:
                        (output / "demo.json").write_text(
                            json.dumps(demo, indent=2), encoding="utf-8"
                        )
                    examples.append(
                        dict(
                            local_case=f"Q4-local-{case['seed']}",
                            n_cleared=result["n_cleared"],
                            avg_localization_clear_s=result["avg_localization_clear_s"],
                            wall_s=result["wall_s"],
                        )
                    )
            rows.append(dict(**case, method=method, **result))
        if i % 20 == 0 or i == len(cases) - 1:
            with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(
                f"Q4: {i+1}/{len(cases)} cases, elapsed {time.perf_counter()-started:.1f}s",
                flush=True,
            )
    summary = summarize(rows)
    comparisons = [
        compare(rows, ref, selected)
        for ref in (
            "baseline_fixed",
            "baseline_dynamic_fixed",
            "geometry_no_visibility",
            "geometry_visibility",
            "bc",
        )
    ]
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "paired_comparisons.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    with (output / "case_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(examples[0]))
        writer.writeheader()
        writer.writerows(examples)
    #     write_report(output,summary,comparisons,selected,examples)
    print(
        json.dumps(dict(selected=selected, comparisons=comparisons), indent=2),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="mixed_sources/models")
    parser.add_argument("--output", default="mixed_sources/results")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--first-seed", type=int, default=140000)
    parser.add_argument("--stress-n", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    evaluate(parser.parse_args())
