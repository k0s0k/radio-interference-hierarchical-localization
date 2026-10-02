"""Paired held-out comparisons, with evaluation-only truth and frozen checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .environment import SearchEnvironment, heuristic_index
from .baseline_adapter import load_baseline, run_baseline
from .policy import load_policy
from .simulator import TaskSimulator, make_scenario


def run_policy(
    env, seed, policy=None, distribution="uniform", error_mode="hash", record=False
):
    observation = env.reset(seed, distribution=distribution, error_mode=error_mode)
    inference_s = 0.0
    while not env.done:
        if policy is None:
            index = heuristic_index(env.actions)
        else:
            started = time.perf_counter()
            index, _ = policy.choose(observation)
            inference_s += time.perf_counter() - started
        observation, _, _, _ = env.step(index)
    result = dict(**env.sim.outcome(), **env.summary(), inference_s=inference_s)
    return result


def paired_comparison(rows, reference, method):
    ref = {
        r["seed"]: r
        for r in rows
        if r["method"] == reference and r["group"] == "uniform"
    }
    test = {
        r["seed"]: r for r in rows if r["method"] == method and r["group"] == "uniform"
    }
    seeds = sorted(set(ref) & set(test))
    # Exclude no failures silently: list the paired success count explicitly.
    good = [s for s in seeds if ref[s]["all_clear"] and test[s]["all_clear"]]
    if not good:
        return dict(
            reference=reference, method=method, paired_success_n=0, total_n=len(seeds)
        )
    x = np.array([ref[s]["virtual_s"] for s in good])
    y = np.array([test[s]["virtual_s"] for s in good])
    indices = np.random.default_rng(9401).integers(0, len(good), (2000, len(good)))
    ratios = 100 * (1 - y[indices].mean(1) / x[indices].mean(1))
    return dict(
        reference=reference,
        method=method,
        paired_success_n=len(good),
        total_n=len(seeds),
        reduction_pct=float(100 * (1 - y.mean() / x.mean())),
        paired_bootstrap_ci95_pct=np.quantile(ratios, [0.025, 0.975]).tolist(),
        faster_count=int((y < x).sum()),
    )


def summarize(rows):
    result = []
    for group in dict.fromkeys(r["group"] for r in rows):
        for method in dict.fromkeys(r["method"] for r in rows):
            subset = [r for r in rows if r["group"] == group and r["method"] == method]
            if not subset:
                continue
            success = [r for r in subset if r["all_clear"]]
            result.append(
                dict(
                    group=group,
                    method=method,
                    n=len(subset),
                    all_clear_count=len(success),
                    all_clear_rate=len(success) / len(subset),
                    mean_virtual_s=float(np.mean([r["virtual_s"] for r in subset])),
                    mean_success_virtual_s=(
                        float(np.mean([r["virtual_s"] for r in success]))
                        if success
                        else None
                    ),
                    p95_virtual_s=float(
                        np.quantile([r["virtual_s"] for r in subset], 0.95)
                    ),
                    max_virtual_s=max(r["virtual_s"] for r in subset),
                    mean_wall_s=float(np.mean([r["wall_s"] for r in subset])),
                    max_wall_s=max(r["wall_s"] for r in subset),
                    mean_inference_s=float(np.mean([r["inference_s"] for r in subset])),
                    mean_requests=float(np.mean([r["requests"] for r in subset])),
                    total_clear_failures=sum(r["clear_failure"] for r in subset),
                )
            )
    return result


# 报告排版源码已注释保存在“辅助材料/注释代码”。


def evaluate(args):
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.models).glob("seed*/best_ppo.pt"))
    if not paths:
        raise FileNotFoundError("Train at least one checkpoint before evaluation")
    models, metadata = {}, {}
    for path in paths:
        name = "ppo_" + path.parent.name
        models[name], metadata[name] = load_policy(path)
    selected = min(
        models,
        key=lambda name: (1 - metadata[name]["validation"]["all_clear"]) * 1e6
        + metadata[name]["validation"]["mean_s"],
    )
    primary_path = next(p for p in paths if "ppo_" + p.parent.name == selected)
    models["bc"], _ = load_policy(primary_path.parent / "bc.pt")
    baseline = load_baseline()
    env = SearchEnvironment()
    methods = ["baseline_fixed", "baseline_dynamic", "geometry_greedy", "bc"] + sorted(
        name for name in models if name != "bc"
    )
    cases = [
        dict(
            seed=args.first_seed + i,
            group="uniform",
            distribution="uniform",
            error_mode="hash",
        )
        for i in range(args.n)
    ]
    for index, (distribution, error_mode) in enumerate(
        [
            ("boundary", "extreme"),
            ("cluster", "smooth"),
            ("collinear", "extreme"),
            ("uniform", "smooth"),
        ]
    ):
        cases.extend(
            dict(
                seed=50000 + 1000 * index + i,
                group=f"{distribution}_{error_mode}",
                distribution=distribution,
                error_mode=error_mode,
            )
            for i in range(args.stress_n)
        )
    manifest = dict(
        selected=selected,
        selection="validation_only",
        cases=cases,
        checkpoints={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        source_files={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("*.py")
        },
        runtime=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            torch=str(torch.__version__),
            torch_threads=args.threads,
            device="cpu",
        ),
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    rows = []
    started = time.perf_counter()
    for case_index, case in enumerate(cases):
        for method in methods:
            if method.startswith("baseline"):
                sim = TaskSimulator(
                    make_scenario(case["seed"], case["distribution"]),
                    seed=case["seed"],
                    error_mode=case["error_mode"],
                    record=False,
                )
                run_baseline(baseline, sim, dynamic=method == "baseline_dynamic")
                result = dict(
                    **sim.outcome(),
                    steps=0,
                    fallback_steps=0,
                    certificate=None,
                    failure_reason="",
                    inference_s=0.0,
                )
            else:
                result = run_policy(
                    env,
                    case["seed"],
                    models.get(method),
                    distribution=case["distribution"],
                    error_mode=case["error_mode"],
                )
            rows.append(dict(**case, method=method, **result))
        if case_index % 20 == 0 or case_index == len(cases) - 1:
            with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(
                f"Evaluated {case_index+1}/{len(cases)} paired scenarios in {time.perf_counter()-started:.1f}s",
                flush=True,
            )
    summary = summarize(rows)
    comparisons = [
        paired_comparison(rows, reference, selected)
        for reference in ("baseline_fixed", "baseline_dynamic", "geometry_greedy", "bc")
    ]
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "paired_comparisons.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    #     write_report(output, summary, comparisons, selected)
    # 绘图用的示例轨迹导出已注释，见辅助材料。
    print(
        json.dumps(dict(selected=selected, comparisons=comparisons), indent=2),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="omnidirectional/models")
    parser.add_argument("--output", default="omnidirectional/results")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--first-seed", type=int, default=40000)
    parser.add_argument("--stress-n", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    evaluate(parser.parse_args())
