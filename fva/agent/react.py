"""自研 ReAct 智能体框架：让模型在「推理 → 工具调用 → 观察」之间自主循环。

为什么不直接串行调用 LLM：本框架把**决策权**交给模型——调哪些工具、调几次、
何时停止，都由模型依据证据自行决定；框架只负责三件事：
1. 能力边界：只暴露受控工具（知识检索 / 仓库只读查询），没有别的网络与执行能力；
2. 预算约束：轮数、工具次数、观察长度上限，保证单样本成本可控、不会死循环；
3. 过程留痕：每一步思考、调用与观察都写入事件流，供界面展示与事后复盘。

与 LangGraph 等图编排框架的差异：这里用显式状态机（THINK → ACT → OBSERVE）
实现同一套「自主决策 + 工具调用」语义，代价是不支持并行分支，收益是零依赖、
可完全离线运行，且每一步都可注入回调与断言（便于测试）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

from ..config import AgentSettings
from ..llm import AssistantTurn, LLMClient, LLMError, ToolCall
from ..textutil import truncate
from .prompts import REPORT_REQUEST, SYSTEM_PROMPT
from .tools import ToolContext, ToolSpec, build_registry, normalize_verdict, tool_schemas


@dataclass
class AgentReport:
    """Agent 的最终产物：结构化判定 + 过程统计。"""

    verdict: str
    confidence: float = 0.5
    cwe: list[str] = field(default_factory=list)
    reason: str = ""
    key_evidence: list[str] = field(default_factory=list)
    steps: int = 0
    tool_calls: int = 0
    tokens: dict = field(default_factory=dict)
    finished_by: str = "submit"        # submit / forced / fallback / error
    error: str = ""
    elapsed: float = 0.0

    @property
    def prediction(self) -> int:
        return 1 if self.verdict == "有漏洞" else 0


class ReActAgent:
    """通用 ReAct 循环（与具体模型/工具解耦，便于替换与测试）。"""

    def __init__(
        self,
        llm: LLMClient,
        context: ToolContext,
        settings: AgentSettings | None = None,
        registry: dict[str, ToolSpec] | None = None,
        on_event: Callable[[dict], None] | None = None,
    ):
        self.llm = llm
        self.ctx = context
        self.settings = settings or AgentSettings()
        self.registry = registry or build_registry()
        self.on_event = on_event or context.on_event
        self.ctx.on_event = self.on_event

    # ---------------------------------------------------------- 事件
    def _emit(self, kind: str, **payload) -> None:
        self.on_event({"type": kind, **payload})

    # ---------------------------------------------------------- 主循环
    def run(self, user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> AgentReport:
        started = time.time()
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        repo_available = self.ctx.settings.repo.enabled
        schemas = tool_schemas(self.registry, include_repo=repo_available)

        steps = 0
        tool_calls = 0
        tokens = {"prompt": 0, "completion": 0}
        finished_by = "forced"
        report: dict | None = None
        error = ""
        self._emit("stage", stage="agent_start", detail=f"模型：{self.llm.name}；工具 {len(schemas)} 个")

        try:
            while steps < self.settings.max_steps:
                steps += 1
                turn: AssistantTurn = self.llm.chat(messages, tools=schemas, tool_choice="auto")
                self._accumulate(tokens, turn.usage)
                if turn.text.strip():
                    self._emit("thought", step=steps, text=truncate(turn.text.strip(), 1200))

                if not turn.tool_calls:
                    # 模型想直接下结论：先尝试「补取证」，再强制结构化提交
                    if steps == 1 and repo_available and self.settings.min_repo_checks > 0:
                        messages.append({"role": "assistant", "content": turn.text or ""})
                        messages.append({"role": "user", "content": (
                            "在下结论前，请至少做一次仓库核验：先调用 repo_overview 确认快照版本，"
                            "再用 find_callers / grep_code / get_function 之一定位关键上下文。"
                        )})
                        self._emit("stage", stage="nudge", detail="模型跳过取证，已要求其补充仓库核验")
                        continue
                    messages.append({"role": "assistant", "content": turn.text or ""})
                    forced = self._force_report(messages, schemas, tokens)
                    if forced is not None:
                        report = forced[0]
                        finished_by = "forced"
                        steps += forced[1]
                        break
                    report = self._fallback_report(turn.text)
                    finished_by = "fallback"
                    break

                messages.append(self._assistant_message(turn))
                terminal = False
                for call in turn.tool_calls:
                    tool_calls += 1
                    self.ctx.tool_calls = tool_calls
                    result, is_terminal = self._execute(call, steps)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": truncate(result, self.settings.max_observation_chars),
                    })
                    if is_terminal:
                        report = self._parse_report(call)
                        finished_by = "submit"
                        terminal = True
                if terminal:
                    break
                if tool_calls >= self.settings.max_tool_calls:
                    self._emit("stage", stage="budget", detail="工具调用次数达到上限，要求立即收敛")
                    forced = self._force_report(messages, schemas, tokens)
                    if forced is not None:
                        report = forced[0]
                        steps += forced[1]
                        finished_by = "forced"
                    else:
                        report = self._fallback_report("")
                        finished_by = "fallback"
                    break
        except LLMError as exc:
            error = str(exc)
            self._emit("error", detail=error)
            report = None
        except Exception as exc:  # 保证单个样本异常不拖垮批量任务
            error = f"{type(exc).__name__}: {exc}"
            self._emit("error", detail=error)
            report = None

        if report is None and not error:
            # 轮数用尽仍未提交：再强制收尾一次，保证每条样本都有结构化结论
            self._emit("stage", stage="budget", detail="轮数达到上限，要求立即收敛结论")
            forced = self._force_report(messages, schemas, tokens)
            if forced is not None:
                report = forced[0]
                steps += forced[1]
                finished_by = "forced"
            else:
                report = self._fallback_report("")
                finished_by = "fallback"

        if report is None:
            report = self._empty_report(error)
            finished_by = "error"

        result = AgentReport(
            verdict=report["verdict"],
            confidence=report["confidence"],
            cwe=report["cwe"],
            reason=report["reason"],
            key_evidence=report["key_evidence"],
            steps=steps,
            tool_calls=tool_calls,
            tokens=tokens,
            finished_by=finished_by,
            error=error,
            elapsed=round(time.time() - started, 2),
        )
        self._emit("verdict", verdict=result.verdict, prediction=result.prediction,
                   confidence=result.confidence, reason=truncate(result.reason, 1500),
                   finished_by=finished_by, steps=steps, tool_calls=tool_calls)
        return result

    # ---------------------------------------------------------- 工具执行
    def _execute(self, call: ToolCall, step: int) -> tuple[str, bool]:
        spec = self.registry.get(call.name)
        self._emit("tool_call", step=step, tool=call.name, args=_short_args(call.arguments))
        if spec is None:
            self._emit("tool_result", tool=call.name, ok=False, preview="未知工具")
            return f"[未知工具] {call.name}，可用工具：{', '.join(self.registry)}", False

        signature = f"{call.name}:{json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)}"
        count = self.ctx.seen_calls.get(signature, 0) + 1
        self.ctx.seen_calls[signature] = count
        if count > 2:
            message = f"[已跳过] 完全相同的调用已执行 {count - 1} 次，请换一种取证方式或直接提交结论。"
            self._emit("tool_result", tool=call.name, ok=False, preview=message)
            return message, False

        started = time.time()
        try:
            result = spec.handler(self.ctx, call.arguments or {})
        except Exception as exc:
            result = f"[工具执行异常] {type(exc).__name__}: {exc}"
        elapsed = round(time.time() - started, 2)
        ok = not result.startswith(("[", "（"))
        if spec.terminal and result.startswith("{"):
            ok = True
        preview = truncate(result.replace("\n", " ⏎ "), 260)
        self._emit("tool_result", tool=call.name, ok=ok, elapsed=elapsed,
                   chars=len(result), preview=preview)
        return result, spec.terminal

    # ---------------------------------------------------------- 结论解析
    def _parse_report(self, call: ToolCall) -> dict:
        args = call.arguments or {}
        verdict = normalize_verdict(args.get("verdict"))
        confidence = args.get("confidence")
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.6 if verdict == "有漏洞" else 0.55
        cwe = args.get("cwe") or []
        if isinstance(cwe, str):
            cwe = [cwe]
        evidence = args.get("key_evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        return {
            "verdict": verdict,
            "confidence": confidence,
            "cwe": [str(item) for item in cwe][:5],
            "reason": truncate(str(args.get("reason") or "").strip(), 4000),
            "key_evidence": [str(item) for item in evidence][:6],
        }

    def _force_report(self, messages: list[dict], schemas: list[dict],
                      tokens: dict) -> tuple[dict, int] | None:
        """预算耗尽或模型迟迟不提交时，强制其以结构化方式收尾。"""
        messages = messages + [{"role": "user", "content": REPORT_REQUEST}]
        try:
            turn = self.llm.chat(
                messages,
                tools=schemas,
                tool_choice={"type": "function", "function": {"name": "submit_report"}},
            )
        except LLMError as exc:
            self._emit("error", detail=f"强制收尾失败：{exc}")
            return None
        self._accumulate(tokens, turn.usage)
        for call in turn.tool_calls:
            if call.name == "submit_report":
                return self._parse_report(call), 1
        if turn.text.strip():
            return self._fallback_report(turn.text), 1
        return None

    def _fallback_report(self, text: str) -> dict:
        """模型未按格式提交时，从自然语言中抽取结论（保底不丢样本）。"""
        verdict = normalize_verdict(text)
        return {
            "verdict": verdict,
            "confidence": 0.3,
            "cwe": [],
            "reason": truncate(text.strip() or "（模型未给出结构化结论，按兜底规则判定）", 2000),
            "key_evidence": [],
        }

    @staticmethod
    def _empty_report(error: str) -> dict:
        return {
            "verdict": "有漏洞",
            "confidence": 0.0,
            "cwe": [],
            "reason": f"分析未能完成：{error}",
            "key_evidence": [],
        }

    # ---------------------------------------------------------- 辅助
    @staticmethod
    def _assistant_message(turn: AssistantTurn) -> dict:
        return {
            "role": "assistant",
            "content": turn.text or "",
            "tool_calls": [call.to_message() for call in turn.tool_calls],
        }

    @staticmethod
    def _accumulate(tokens: dict, usage: dict) -> None:
        if not usage:
            return
        tokens["prompt"] = tokens.get("prompt", 0) + int(usage.get("prompt_tokens") or 0)
        tokens["completion"] = tokens.get("completion", 0) + int(usage.get("completion_tokens") or 0)


def _short_args(args: dict) -> dict:
    """事件里只保留可读的短参数，避免把整段代码写进日志。"""
    short = {}
    for key, value in (args or {}).items():
        if isinstance(value, str) and len(value) > 160:
            short[key] = value[:160] + "…"
        else:
            short[key] = value
    return short
