# results/ 目录说明

本目录保存检测产物与评测记录。**请先读这里，再使用 `result.jsonl`。**

| 文件 | 内容 | 生成方式 |
|---|---|---|
| `result.jsonl` | 300 条样本的 `{"idx", "prediction"}` | `python -m fva.cli batch` |
| `eval.json` | 按课程口径的评测结果（四类组数、等级） | `python -m fva.cli eval` |
| `reports/<idx>.json` | 每条样本的完整报告（判定、理由、证据、工具轨迹） | 批量运行时自动落盘 |
| `traces/<idx>.json` | 每条样本的完整事件流（可回放分析过程） | 批量运行时自动落盘 |
| `result_llm_sample.jsonl` | 25 对抽样的大模型判定结果（抽样验证，非全量） | `python -m fva.cli import` |
| `eval_sample.json` | 上述抽样结果的评测 | `python -m fva.cli eval --result …` |
| `export_sample/` | 抽样评测的提示词批次（`batch_*.json`）与外部作答（`answer_*.json`） | `python -m fva.cli export --sample 25` |

## 当前 `result.jsonl` 的来源（重要）

> **当前 `result.jsonl` 由「离线降级模式」（`--mode offline`）生成，不是大模型 Agent 的正式成绩。**
>
> - 生成命令：`python -m fva.cli batch --mode offline --no-repo --workers 4`
> - 评测结果：P-C 9 / P-V 72 / P-B 66 / P-R 3，等级 **D**
> - 用途：作为「无 LLM 消融基线」，用于说明纯静态特征无法区分同源两版（详见项目报告第 6 章）。

**正式提交前请用真实模型复跑**（会覆盖本文件）：

```bash
set FVA_LLM_API_KEY=sk-xxx
python -m fva.cli batch --workers 2      # 断点续跑，可中断
python -m fva.cli eval
```

`result_llm_sample.jsonl` 是抽样验证（25 对，单轮 LLM + 领域知识，逐样本隔离）：

- P-C 5 / P-V 3 / P-B 15 / P-R 2（成对正确率 20.0%，样本级准确率 56%）
- 用途：量化「语义推理相对静态特征的增益」（是离线基线 6.0% 的 3.3 倍），不作为正式成绩提交。

## 复现说明

- 结果文件的顺序、条数与测试集一致；`evaluation.py` 允许只包含部分 `idx`（缺预测的对会被跳过）。
- 若更换模型或修改提示词，建议写到新的结果文件名（如 `result_<model>.jsonl`）以保留对照。
- `reports/` 与 `traces/` 是「分析记录」的证据来源，报告中每条结论都能在此回查。
