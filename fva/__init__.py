"""函数漏洞检测智能体（Function Vulnerability Agent, FVA）。

面向 C/C++ 单函数的漏洞检测 Agent：以历史 CVE 案例构成领域知识库（RAG），
在分析过程中按需读取代码仓库上下文，最终给出「有漏洞 / 无漏洞」判定、
判定理由与分析记录。

包结构：
    config      配置与路径
    textutil    代码/文本处理工具（分词、截断、签名提取）
    llm         大模型客户端（OpenAI 兼容协议）与离线策略引擎
    knowledge   领域知识库：成对案例索引构建与混合检索
    repo        仓库快照与上下文挖掘工具（grep/find/调用关系）
    agent       自研 ReAct 智能体框架（工具调用 + 预算 + 留痕）
    pipeline    单样本检测编排与报告生成
    batch       批量检测与 result.jsonl 汇总
    offline     离线静态证据分析器（无 LLM 时的降级大脑 / 评测基线）
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
