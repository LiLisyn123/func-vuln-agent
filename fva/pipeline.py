"""单样本检测流水线：把「知识检索 → 初步分析 → 按需读仓库 → 报告」串起来。

Pipeline 的每一步都会产生事件（Event），供界面实时展示、供报告留存分析记录：

    ① 输入函数与仓库链接
    ② 领域知识检索（RAG）           → kb_hit 事件
    ③ 智能体初步分析与自主取证       → thought / tool_call / tool_result 事件
    ④ 按需读取仓库                   → snapshot 事件（惰性、可降级）
    ⑤ 输出结构化报告                 → verdict 事件

本模块不关心模型是真实的还是离线的，因此离线模式与真实模型模式的
事件流、报告结构、可追溯性完全一致。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import config
from .agent import AgentReport, ReActAgent, ToolContext, build_registry, build_user_prompt
from .knowledge import Hit, KnowledgeBase
from .llm import LLMClient, LLMError, OpenAICompatibleLLM
from .textutil import dump_json, function_name, truncate

VULN, SAFE = "有漏洞", "无漏洞"


class EventLog:
    """事件收集器：统一加上序号与相对时间，便于回放与断言。"""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self._start = time.time()

    def __call__(self, event: dict) -> None:
        record = dict(event)
        record.setdefault("type", "info")
        record["seq"] = len(self.events) + 1
        record["ts"] = round(time.time() - self._start, 3)
        self.events.append(record)

    def of_type(self, kind: str) -> list[dict]:
        return [event for event in self.events if event.get("type") == kind]

    def append(self, event: dict) -> None:
        self.__call__(event)


@dataclass
class DetectionResult:
    """单样本检测的完整产物（机器可读部分 + 人类可读分析记录）。"""

    idx: int | None
    prediction: int
    verdict: str
    confidence: float
    cwe: list[str] = field(default_factory=list)
    reason: str = ""
    key_evidence: list[str] = field(default_factory=list)
    func_name: str = ""
    commit_url: str = ""
    target: int | None = None
    mode: str = "offline"
    model: str = ""
    kb_hits: list[dict] = field(default_factory=list)
    snapshot: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    steps: int = 0
    n_tool_calls: int = 0
    tokens: dict = field(default_factory=dict)
    elapsed: float = 0.0
    finished_by: str = ""
    error: str = ""
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "prediction": self.prediction,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "cwe": self.cwe,
            "reason": self.reason,
            "key_evidence": self.key_evidence,
            "func_name": self.func_name,
            "commit_url": self.commit_url,
            "target": self.target,
            "mode": self.mode,
            "model": self.model,
            "kb_hits": self.kb_hits,
            "snapshot": self.snapshot,
            "tool_calls": self.tool_calls,
            "steps": self.steps,
            "n_tool_calls": self.n_tool_calls,
            "tokens": self.tokens,
            "elapsed": self.elapsed,
            "finished_by": self.finished_by,
            "error": self.error,
        }

    def analysis_record(self, max_events: int = 60) -> str:
        """报告第四项「分析记录」的文本化版本（界面与文档共用）。"""
        lines = []
        if self.kb_hits:
            lines.append("【历史案例检索】")
            for hit in self.kb_hits[:6]:
                lines.append(
                    f"  #{hit.get('rank')} {hit.get('cve')} ({'/'.join(hit.get('cwe') or [])}) "
                    f"@ {hit.get('project')} 相似度 {hit.get('score')}"
                )
        if self.snapshot:
            lines.append(
                "【仓库上下文】"
                f" {self.snapshot.get('slug')} @ {str(self.snapshot.get('snapshot_sha', ''))[:12]}"
                f"（{self.snapshot.get('rule')} 规则，来源 {self.snapshot.get('source')}）"
            )
        elif self.events:
            errors = [e for e in self.events if e.get("type") == "snapshot" and e.get("status") == "error"]
            if errors:
                lines.append(f"【仓库上下文】不可用：{errors[-1].get('detail')}")
        if self.tool_calls:
            lines.append("【工具调用轨迹】")
            for index, call in enumerate(self.tool_calls[:max_events], 1):
                args = call.get("args") or {}
                arg_text = ", ".join(f"{k}={truncate(str(v), 60, '…')}" for k, v in list(args.items())[:3])
                lines.append(f"  {index:>2}. {call.get('tool')}({arg_text}) → {truncate(call.get('preview', ''), 160, '…')}")
        return "\n".join(lines) if lines else "（本次检测没有产生可展示的分析记录）"


class Detector:
    """面向单个函数的检测器；批量任务复用一个实例即可。"""

    def __init__(
        self,
        kb: KnowledgeBase | None = None,
        *,
        mode: str = "auto",
        settings: config.Settings | None = None,
        repo_enabled: bool | None = None,
    ):
        self.settings = settings or config.Settings.load()
        self.kb = kb or KnowledgeBase.load()
        self.mode = mode
        self.repo_enabled = self.settings.repo.enabled if repo_enabled is None else repo_enabled
        self._llm: LLMClient | None = None
        self._offline = None
        self.mode_used = self._resolve_mode()

    # ---------------------------------------------------------- 运行模式
    def _resolve_mode(self) -> str:
        if self.mode == "offline":
            return "offline"
        if self.mode == "llm":
            if not self.settings.llm.configured:
                raise LLMError(
                    "指定了 --mode llm，但未配置大模型密钥。"
                    "请设置 FVA_LLM_API_KEY / DEEPSEEK_API_KEY，或改用 --mode offline。"
                )
            return "llm"
        return "llm" if self.settings.llm.configured else "offline"

    @property
    def model_name(self) -> str:
        return self.settings.llm.model if self.mode_used == "llm" else "offline-analyzer"

    # ---------------------------------------------------------- 主体
    def detect(
        self,
        func_code: str,
        commit_url: str,
        *,
        idx: int | None = None,
        target: int | None = None,
        on_event: Callable[[dict], None] | None = None,
        save_trace: bool = False,
    ) -> DetectionResult:
        log = EventLog()
        listener = on_event or (lambda event: None)

        def emit(event: dict) -> None:
            log.append(event)
            try:
                listener(event)
            except Exception:  # 界面回调异常不应影响检测
                pass

        started = time.time()
        func_code = (func_code or "").strip("\n")
        if not func_code.strip():
            result = self._error_result(idx, target, commit_url, "输入函数为空", log)
            return result

        emit({"type": "stage", "stage": "input", "detail": (
            f"函数 {function_name(func_code) or '?'}；{len(func_code.splitlines())} 行；"
            f"模式 {self.mode_used}（{self.model_name}）"
        )})

        # ② 领域知识检索（Pipeline 第②步，先于推理执行一次）
        hits: list[Hit] = []
        if self.settings.agent.prefetch_kb:
            hits = self.kb.search_function(func_code, top_k=self.settings.agent.max_kb_hits)
            emit({
                "type": "kb_hit",
                "count": len(hits),
                "query": truncate(function_name(func_code) or "（函数源码）", 120, ""),
                "stage": "prefetch",
                "data": [
                    {"rank": hit.rank, "cve": hit.case.cve, "cwe": hit.case.cwe, "score": hit.score,
                     "project": hit.case.project, "func": hit.case.func_name}
                    for hit in hits
                ],
            })
        kb_text = KnowledgeBase.render_hits(hits, per_case_chars=900) if hits else \
            "（知识库中未检索到结构相似的历史案例）"

        # ③ 智能体：自主分析 + 按需取证
        context = ToolContext(
            func_code=func_code,
            commit_url=commit_url,
            target=1 if target is None else int(target),
            kb=self.kb,
            settings=self.settings,
            idx=idx,
            on_event=emit,
        )
        registry = build_registry()
        llm = self._build_llm(context, repo_enabled=self.repo_enabled)
        user_prompt = build_user_prompt(
            func_code,
            commit_url,
            kb_text,
            target_name=function_name(func_code),
            repo_available=self.repo_enabled,
            repo_note="按仓库体量与网络情况自动降级" if self.repo_enabled else "已关闭",
            kb_count=len(hits),
        )
        agent = ReActAgent(llm, context, self.settings.agent, registry, on_event=emit)
        report: AgentReport = agent.run(user_prompt)

        elapsed = round(time.time() - started, 2)
        result = DetectionResult(
            idx=idx,
            prediction=report.prediction,
            verdict=report.verdict,
            confidence=report.confidence,
            cwe=report.cwe,
            reason=report.reason or "（未给出判定理由）",
            key_evidence=report.key_evidence,
            func_name=function_name(func_code),
            commit_url=commit_url,
            target=target,
            mode=self.mode_used,
            model=self.model_name,
            kb_hits=[{
                "rank": hit.rank, "cve": hit.case.cve, "cwe": hit.case.cwe, "score": hit.score,
                "project": hit.case.project, "func": hit.case.func_name, "commit_url": hit.case.commit_url,
            } for hit in hits],
            snapshot=context.snapshot_meta,
            tool_calls=[{
                "tool": event.get("tool"),
                "args": event.get("args") or {},
                "preview": event.get("preview", ""),
                "ok": event.get("ok"),
                "elapsed": event.get("elapsed"),
            } for event in log.of_type("tool_result")],
            steps=report.steps,
            n_tool_calls=report.tool_calls,
            tokens=report.tokens,
            elapsed=elapsed,
            finished_by=report.finished_by,
            error=report.error,
            events=log.events,
        )
        emit({"type": "final", "prediction": result.prediction, "verdict": result.verdict,
              "elapsed": elapsed})
        if save_trace and idx is not None:
            self.save_artifacts(result)
        return result

    # ---------------------------------------------------------- 辅助
    def _build_llm(self, context: ToolContext, repo_enabled: bool) -> LLMClient:
        if self.mode_used == "llm":
            if self._llm is None:
                self._llm = OpenAICompatibleLLM(self.settings.llm)
            return self._llm
        from .offline import OfflineAnalyzer, OfflineLLM

        if self._offline is None:
            self._offline = OfflineAnalyzer.load()
        return OfflineLLM(
            self._offline,
            repo_enabled=repo_enabled,
            prefetched_kb=self.settings.agent.prefetch_kb,
        )

    def _error_result(self, idx, target, commit_url, message: str, log: EventLog) -> DetectionResult:
        log.append({"type": "error", "detail": message})
        return DetectionResult(
            idx=idx, prediction=0, verdict=SAFE, confidence=0.0, reason=f"输入无效：{message}",
            commit_url=commit_url, target=target, mode=self.mode_used, model=self.model_name,
            error=message, events=log.events,
        )

    @staticmethod
    def save_artifacts(result: DetectionResult) -> None:
        """把报告与事件流落盘，便于批量复盘与报告取证。"""
        if result.idx is None:
            return
        config.ensure_dirs()
        dump_json(config.REPORT_DIR / f"{result.idx}.json", result.to_dict())
        dump_json(config.TRACE_DIR / f"{result.idx}.json", {
            "idx": result.idx,
            "prediction": result.prediction,
            "mode": result.mode,
            "events": result.events,
        })


def detect_one(
    func_code: str,
    commit_url: str,
    *,
    idx: int | None = None,
    target: int | None = None,
    mode: str = "auto",
    repo_enabled: bool | None = None,
    kb: KnowledgeBase | None = None,
    on_event: Callable[[dict], None] | None = None,
    save_trace: bool = False,
    settings: config.Settings | None = None,
) -> DetectionResult:
    """便捷入口：构造检测器并检测单条样本。"""
    detector = Detector(kb=kb, mode=mode, settings=settings, repo_enabled=repo_enabled)
    return detector.detect(func_code, commit_url, idx=idx, target=target,
                           on_event=on_event, save_trace=save_trace)


def new_run_id() -> str:
    return uuid.uuid4().hex[:8]
