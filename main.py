"""四问统一求解入口；第三、四问加载已训练的种子 0 策略。"""

import argparse
import json
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent


def solve_search(question, seed):
    """执行完整搜索闭环，结束后按题目定义汇总三个指标。"""
    import torch

    torch.set_num_threads(2)
    if question == 3:
        from omnidirectional.environment import SearchEnvironment
        from omnidirectional.evaluate import run_policy
        from omnidirectional.policy import load_policy

        package = "omnidirectional"
        scenario_seed = 40000 if seed is None else seed
    else:
        from mixed_sources.environment import SearchEnvironment
        from mixed_sources.evaluate import run_policy
        from mixed_sources.policy import load_policy

        package = "mixed_sources"
        scenario_seed = 140000 if seed is None else seed

    checkpoint = CODE_ROOT / package / "models/seed0/best_ppo.pt"
    policy, _ = load_policy(checkpoint)
    environment = SearchEnvironment(record=True)
    outcome = run_policy(environment, scenario_seed, policy, "uniform", "hash")
    cleared_count = outcome["n_cleared"]
    outcome["avg_localization_clear_s"] = (
        outcome["virtual_s"] / cleared_count if cleared_count else None
    )
    # 真值仅用于回合结束后的评估指标；求解输出不导出绘图用源坐标。
    return {
        "question": question,
        "seed": scenario_seed,
        "outcome": outcome,
        "macro_actions": environment.macro_log,
        "requests": environment.sim.log,
    }


def main():
    parser = argparse.ArgumentParser(description="无线电干扰源定位与清除 四问求解")
    parser.add_argument(
        "--question", type=int, choices=[1, 2, 3, 4], required=True, help="题目编号"
    )
    parser.add_argument("--seed", type=int, help="第三、四问测试场景的随机种子")
    parser.add_argument("--output", type=Path, help="结果 JSON 的保存路径")
    args = parser.parse_args()

    if args.question in (1, 2):
        from static_geometry.models import q1_example, q2_candidate

        result = q1_example() if args.question == 1 else q2_candidate()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        result = solve_search(args.question, args.seed)
        outcome = result["outcome"]
        average_time = outcome["avg_localization_clear_s"]
        average_text = (
            f"{average_time:.6f} 秒/个"
            if average_time is not None
            else "未定义（清除数为零）"
        )
        print(f"清除干扰源个数：{outcome['n_cleared']}")
        print(f"平均定位清除时间：{average_text}")
        print(f"程序运行时间：{outcome['wall_s']:.6f} 秒")
        print(f"总虚拟时间：{outcome['virtual_s']:.6f} 秒")

    output = (
        args.output or CODE_ROOT / "outputs" / f"question_{args.question}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("结果文件：", output.resolve())


if __name__ == "__main__":
    main()
