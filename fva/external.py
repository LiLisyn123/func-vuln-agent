"""外部模型批量推理：导出提示词 → 由任意可用模型作答 → 回填为 result.jsonl。

适用场景：开发机没有可直接调用的 API Key，但可以通过网页/会话界面使用同一模型。
流程与 `--mode llm` 完全一致（同样的系统提示词、同样的知识检索上下文、逐样本隔离），
只是「模型调用」由外部完成，产物可以回到本项目的评测流水线。

约定
----
* 导出：`python -m fva.cli export --per-batch 5`
  生成 `results/export/batch_XX.json`（每批若干样本，**保证同一对的两个版本不在同一批**）
  与 `results/export/manifest.json`。
* 作答：外部模型按 `results/export/template.md` 的格式，把答案写到 `answer_XX.json`。
* 回填：`python -m fva.cli import --answers results/export --result results/result_llm.jsonl`
  校验 idx 完整性、verdict 合法性后写出标准结果文件。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import config
from .knowledge import KnowledgeBase
from .textutil import dump_json, function_name, load_json, read_jsonl, truncate

SYSTEM_PROMPT_ONESHOT = """你是「函数漏洞检测智能体」的分析内核，专门判断单个 C/C++ 函数是否存在安全漏洞。

# 可用材料（仅限以下两类，禁止使用其它外部知识或联网搜索）
1. 用户给出的函数源码（本次只给出一个函数，没有它的其它版本，也不存在 diff 可供对比）；
2. 领域知识库中检索到的历史 CVE 相似案例（含漏洞描述与修复要点）。

# 分析要求
1. 先逐个列出函数内部的危险动作：内存拷贝与长度计算、数组下标与指针运算、整数运算与类型转换、
   字符串函数、内存分配与释放、错误返回值的使用、外部命令与格式化输出。
2. 对每个危险动作，判断函数内是否存在有效缓解：上界比较、SIZE_MAX/INT_MAX 检查、长度封顶、
   安全替代 API（snprintf/strncpy 等）、空指针检查、错误返回处理。
3. 参考历史案例的「修复要点」，判断当前代码是否符合「已修复」的形态。
4. 只因为代码风格不完美、理论上可能失败，不足以判有漏洞；但信息不足也不能默认无漏洞，
   应倾向选择证据更充分的一方，并在理由中说明依据。

# 判定口径
- 有漏洞：至少存在一个可被触发、且在函数内部看不到缓解措施的缺陷。
- 无漏洞：所有危险动作都有明确缓解证据，或代码路径不具备触发条件。

# 重要先验
待检样本中约有半数是**已经修复过的版本**。修复的典型形态是：在危险操作前后补上校验语句
（长度/边界比较、SIZE_MAX 或 INT_MAX 检查、空指针判断、改用 snprintf/strncpy/strlcpy 等安全 API、
出错时提前 return）。因此——
* 若函数已经对关键风险点（长度、下标、指针有效性）做了校验，应当判「无漏洞」，
  不要因为代码里仍然出现 memcpy / 指针运算就判有漏洞；
* 只有当风险动作在当前代码中确实**没有**任何缓解，且能被调用者以可控方式触发时，才判「有漏洞」。
* 对同一函数的两个版本，修复版与漏洞版在字面上往往只差几行校验，差别就在「有没有这一步校验」。

# 输出格式（必须是严格 JSON，不要输出其它内容）
{"idx": <整数>, "verdict": "有漏洞" 或 "无漏洞", "confidence": <0~1 的小数>,
 "cwe": ["CWE-xxx"], "reason": "≤200 字的判定理由，引用具体行号或代码片段"}"""


def build_prompt(sample: dict, kb: KnowledgeBase, kb_limit: int = 4) -> dict:
    """为一条样本构造单轮提示词（与 API 模式共用同一套知识检索与判定口径）。"""
    code = sample.get("func") or ""
    hits = kb.search_function(code, top_k=kb_limit)
    kb_text = KnowledgeBase.render_hits(hits, per_case_chars=500) if hits else \
        "（知识库中未检索到结构相似的历史案例）"
    user = f"""请判断下面这个 C/C++ 函数是否存在安全漏洞。

## 待检测函数（函数名：{function_name(code) or '未知'}）
```c
{truncate(code, 9000)}
```

## 所属仓库与提交
- 修复提交链接：{sample.get('commit_url') or '（无）'}
- 说明：分析对象是**该提交对应版本**的函数本身，不提供任何变更信息

## 历史相似案例（{len(hits)} 条，来自领域知识库）
{kb_text}

## 输出
只输出一行 JSON，字段：idx、verdict、confidence、cwe、reason。"""
    return {
        "idx": sample.get("idx"),
        "system": SYSTEM_PROMPT_ONESHOT,
        "user": user,
        "kb_hits": [{"rank": h.rank, "cve": h.case.cve, "cwe": h.case.cwe, "score": h.score} for h in hits],
    }


def export_sample_prompts(pairs: int = 25, per_batch: int = 5, seed: int = 20261122,
                          out_dir: Path | None = None) -> dict:
    """导出「抽样评测」用的提示词：每批混入漏洞版与修复版，且同一对的两个版本不同批。

    抽样方式：等间隔抽取若干对（覆盖面均匀），每批按 3:2 混入两种标签。
    """
    import random

    out_dir = Path(out_dir or (config.RESULTS_DIR / "export_sample"))
    out_dir.mkdir(parents=True, exist_ok=True)
    kb = KnowledgeBase.load()
    samples = list(read_jsonl(config.TEST_PATH))
    total_pairs = len(samples) // 2
    pairs = max(1, min(pairs, total_pairs))
    step = max(1, total_pairs // pairs)
    chosen = [i for i in range(0, total_pairs, step)][:pairs]

    vuln_stream, fixed_stream = [], []
    for pair_index in chosen:
        first, second = samples[2 * pair_index], samples[2 * pair_index + 1]
        vulnerable, fixed = (first, second) if first["target"] == 1 else (second, first)
        vuln_stream.append(build_prompt(vulnerable, kb))
        fixed_stream.append(build_prompt(fixed, kb))
    random.Random(seed).shuffle(vuln_stream)
    random.Random(seed + 1).shuffle(fixed_stream)

    batches, batch_number = [], 0
    while vuln_stream or fixed_stream:
        batch_number += 1
        batch = vuln_stream[:3] + fixed_stream[:2]
        vuln_stream, fixed_stream = vuln_stream[3:], fixed_stream[2:]
        if len(batch) < per_batch and (vuln_stream or fixed_stream):
            batch += vuln_stream[:1] + fixed_stream[:1]
            vuln_stream, fixed_stream = vuln_stream[1:], fixed_stream[1:]
        if batch:
            batches.append(batch)

    manifest = {"per_batch": per_batch, "batches": [], "samples": 0, "pairs": len(chosen)}
    for number, batch in enumerate(batches, 1):
        payload = {"batch_id": f"{number:03d}", "samples": batch}
        path = out_dir / f"batch_{number:03d}.json"
        dump_json(path, payload)
        manifest["batches"].append({"batch_id": payload["batch_id"], "path": str(path),
                                    "idx": [s["idx"] for s in batch]})
        manifest["samples"] += len(batch)
    dump_json(out_dir / "manifest.json", manifest)
    (out_dir / "template.md").write_text(_template(len(batches)), encoding="utf-8")
    return manifest


def export_prompts(per_batch: int = 5, limit_pairs: int | None = None,
                   out_dir: Path | None = None) -> dict:
    """导出分批提示词；同一对的两个版本一定落在不同批次（严格成对隔离）。"""
    out_dir = Path(out_dir or (config.RESULTS_DIR / "export"))
    out_dir.mkdir(parents=True, exist_ok=True)
    kb = KnowledgeBase.load()
    samples = list(read_jsonl(config.TEST_PATH))
    if limit_pairs:
        samples = samples[: limit_pairs * 2]

    # 偶数位 / 奇数位分别成批：测试集中相邻两条为一对，因此两种批次天然隔离
    streams = [samples[0::2], samples[1::2]]
    batches = []
    for stream in streams:
        for start in range(0, len(stream), per_batch):
            batches.append(stream[start:start + per_batch])

    manifest = {"per_batch": per_batch, "batches": [], "samples": len(samples)}
    for number, batch in enumerate(batches, 1):
        payload = {
            "batch_id": f"{number:03d}",
            "samples": [build_prompt(sample, kb) for sample in batch],
        }
        path = out_dir / f"batch_{number:03d}.json"
        dump_json(path, payload)
        manifest["batches"].append({
            "batch_id": payload["batch_id"],
            "path": str(path),
            "idx": [s["idx"] for s in payload["samples"]],
        })
    dump_json(out_dir / "manifest.json", manifest)
    (out_dir / "template.md").write_text(_template(len(batches)), encoding="utf-8")
    return manifest


def _template(batch_count: int) -> str:
    return f"""# 外部模型批量作答说明

本目录由 `python -m fva.cli export` 生成，共 {batch_count} 个批次文件 `batch_XXX.json`。
每个批次包含若干条**相互独立**的样本（同一对的两个版本不会出现在同一批次），
字段 `system` / `user` 即需要发送给模型的提示词。

## 作答方式

对每个 `batch_XXX.json`：

1. 把 `system` 作为系统提示词、`user` 作为用户消息发给模型（可逐条发送，也可把多条 `user`
   用 `=== SAMPLE idx=<idx> ===` 分隔后一起发送）；
2. 把结果写到同目录的 `answer_XXX.json`，格式为数组：

```json
[
  {{"idx": 194963, "verdict": "有漏洞", "confidence": 0.72, "cwe": ["CWE-787"],
    "reason": "第 245 行 memcpy 的长度来自属性长度字段，缺少与 dst 容量的比较……"}},
  {{"idx": 217569, "verdict": "无漏洞", "confidence": 0.65, "cwe": [],
    "reason": "第 250 行已用 if (length > extent) 截断……"}}
]
```

3. 全部批次完成后执行：

```bash
python -m fva.cli import --answers results/export --result results/result_llm.jsonl
```

`import` 会校验：idx 是否覆盖全部导出样本、verdict 是否为「有漏洞/无漏洞」，
然后写出标准 `result.jsonl`（每行 `{{"idx":…, "prediction":0|1}}`）与完整报告。
"""


ANSWER_RE = re.compile(r"\{[^{}]*\"idx\"[^{}]*\}", re.S)


def _extract_answers(text: str) -> list[dict]:
    """优先按完整 JSON 解析，失败时退化为逐个对象提取（兼容带说明文字的作答文件）。"""
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass
    items = []
    for match in ANSWER_RE.finditer(text):
        try:
            items.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    return items


def import_answers(answers_dir: Path | str, result_path: Path | str) -> dict:
    """把外部模型的作答回填为标准结果文件。"""
    answers_dir = Path(answers_dir)
    records: dict[int, dict] = {}
    files = sorted(answers_dir.glob("answer_*.json"))
    if not files:
        raise FileNotFoundError(f"{answers_dir} 下没有 answer_*.json")
    for path in files:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for item in _extract_answers(text):
            idx = item.get("idx")
            if isinstance(idx, str) and idx.strip().isdigit():
                idx = int(idx.strip())
            if not isinstance(idx, int):
                continue
            records[idx] = item

    manifest = load_json(answers_dir / "manifest.json", default={}) or {}
    expected = {idx for batch in manifest.get("batches", []) for idx in batch.get("idx", [])}
    missing = expected - set(records)
    invalid = [idx for idx, item in records.items()
               if str(item.get("verdict", "")).strip() not in ("有漏洞", "无漏洞")]

    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as handle:
        for idx in sorted(records):
            item = records[idx]
            verdict = str(item["verdict"]).strip()
            handle.write(json.dumps({
                "idx": idx,
                "prediction": 1 if verdict == "有漏洞" else 0,
                "verdict": verdict,
                "confidence": item.get("confidence"),
                "cwe": item.get("cwe") or [],
                "reason": item.get("reason") or "",
            }, ensure_ascii=False) + "\n")
    return {
        "answers": len(records),
        "expected": len(expected),
        "missing": sorted(missing),
        "invalid": sorted(invalid),
        "result": str(result_path),
    }
