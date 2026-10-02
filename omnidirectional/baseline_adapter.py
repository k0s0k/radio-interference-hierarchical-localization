"""对照算法适配；动态巡检次序仅在本次比较期间生效。"""

from importlib import import_module


def load_baseline():
    """加载覆盖巡检基线；基线模块不执行绘图与报告导出。"""
    return import_module("baselines.question_three")


def run_baseline(module, simulator, dynamic=False):
    previous = module.nn_order
    if dynamic:

        def dynamic_order(points, _start):
            import numpy as np

            todo = list(points)
            while todo:
                index = min(
                    range(len(todo)),
                    key=lambda i: np.linalg.norm(
                        np.asarray(todo[i]) - simulator.robot_pos
                    ),
                )
                yield todo.pop(index)

        module.nn_order = dynamic_order
    try:
        module.solve(simulator, cover=650.0, directional=False)
        simulator.exit()
    finally:
        module.nn_order = previous
