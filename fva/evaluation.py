"""评测模块：复用课程提供的 `dataset/evaluation.py` 口径，并补充等级换算与可读报表。

课程评测按「相邻两条记录为一对」统计四类：
    P-C 成对正确 / P-V 均判有洞 / P-B 均判无洞 / P-R 完全反转
等级按 P-C 对数划分：A ≥ 30、B 25~29、C 20~24、D < 20；
若 P-R > P-C / 2 则降一级（最低 D）。

本模块不重复实现统计逻辑，而是直接加载课程脚本，保证与官方口径一致。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from . import config

LEVELS = [("A", 30), ("B", 25), ("C", 20), ("D", 0)]


def load_official_module():
    """动态加载课程提供的 evaluation.py（仅依赖标准库）。"""
    path = Path(config.EVALUATE_SCRIPT)
    if not path.exists():
        raise FileNotFoundError(f"缺少评测脚本：{path}")
    spec = importlib.util.spec_from_file_location("course_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def evaluate(result_path: Path | str = config.RESULT_PATH,
             dataset_path: Path | str = config.TEST_PATH) -> dict:
    """返回评测结果字典（含四类组数、比率与等级）。"""
    module = load_official_module()
    sample_count, unique_count, counts = module.evaluate(Path(dataset_path), Path(result_path))
    pairs = sample_count // 2
    evaluated = sum(counts.values())
    level, base = _grade(counts["P-C"])
    demoted = False
    if counts["P-R"] > counts["P-C"] / 2:
        level = _demote(level)
        demoted = True
    report = {
        "dataset": str(dataset_path),
        "result": str(result_path),
        "samples": sample_count,
        "unique_idx": unique_count,
        "pairs": pairs,
        "evaluated_pairs": evaluated,
        "skipped_pairs": pairs - evaluated,
        "counts": counts,
        "rates": {
            "P-C": round(counts["P-C"] / pairs, 4) if pairs else 0.0,
            "P-V": round(counts["P-V"] / pairs, 4) if pairs else 0.0,
            "P-B": round(counts["P-B"] / pairs, 4) if pairs else 0.0,
            "P-R": round(counts["P-R"] / pairs, 4) if pairs else 0.0,
        },
        "level": level,
        "base_level": base,
        "demoted": demoted,
        "demotion_rule": "P-R 组数超过 P-C 组数的一半则降一级" if demoted else "",
    }
    return report


def _grade(pair_correct: int) -> tuple[str, str]:
    for level, threshold in LEVELS:
        if pair_correct >= threshold:
            return level, level
    return "D", "D"


def _demote(level: str) -> str:
    order = ["A", "B", "C", "D"]
    index = order.index(level)
    return order[min(index + 1, len(order) - 1)]


def render_markdown(report: dict) -> str:
    counts = report["counts"]
    rates = report["rates"]
    lines = [
        "## 评测结果",
        "",
        f"- 测试集：`{report['dataset']}`（{report['samples']} 条 / {report['pairs']} 对）",
        f"- 结果文件：`{report['result']}`（实际评估 {report['evaluated_pairs']} 对，"
        f"跳过 {report['skipped_pairs']} 对）",
        f"- 等级：**{report['level']}**" + (f"（因 P-R 惩罚由 {report['base_level']} 降级）"
                                          if report["demoted"] else ""),
        "",
        "| 类别 | 含义 | 组数 | 占比 |",
        "|---|---|---|---|",
        f"| P-C | 成对正确（漏洞版→1、修复版→0） | {counts['P-C']} | {rates['P-C']:.1%} |",
        f"| P-V | 两版均判「有漏洞」 | {counts['P-V']} | {rates['P-V']:.1%} |",
        f"| P-B | 两版均判「无漏洞」 | {counts['P-B']} | {rates['P-B']:.1%} |",
        f"| P-R | 完全反转（漏洞版→0、修复版→1） | {counts['P-R']} | {rates['P-R']:.1%} |",
    ]
    return "\n".join(lines)
