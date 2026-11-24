# 外部模型批量作答说明

本目录由 `python -m fva.cli export` 生成，共 11 个批次文件 `batch_XXX.json`。
每个批次包含若干条**相互独立**的样本（同一对的两个版本不会出现在同一批次），
字段 `system` / `user` 即需要发送给模型的提示词。

## 作答方式

对每个 `batch_XXX.json`：

1. 把 `system` 作为系统提示词、`user` 作为用户消息发给模型（可逐条发送，也可把多条 `user`
   用 `=== SAMPLE idx=<idx> ===` 分隔后一起发送）；
2. 把结果写到同目录的 `answer_XXX.json`，格式为数组：

```json
[
  {"idx": 194963, "verdict": "有漏洞", "confidence": 0.72, "cwe": ["CWE-787"],
    "reason": "第 245 行 memcpy 的长度来自属性长度字段，缺少与 dst 容量的比较……"},
  {"idx": 217569, "verdict": "无漏洞", "confidence": 0.65, "cwe": [],
    "reason": "第 250 行已用 if (length > extent) 截断……"}
]
```

3. 全部批次完成后执行：

```bash
python -m fva.cli import --answers results/export --result results/result_llm.jsonl
```

`import` 会校验：idx 是否覆盖全部导出样本、verdict 是否为「有漏洞/无漏洞」，
然后写出标准 `result.jsonl`（每行 `{"idx":…, "prediction":0|1}`）与完整报告。
