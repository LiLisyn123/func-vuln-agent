"""大模型接入层：OpenAI Chat Completions 协议（含 Function Calling）。

设计目标
--------
* **与厂商解耦**：只要服务端兼容 OpenAI 的 `/chat/completions`（DeepSeek、通义、
  vLLM、Ollama 的 OpenAI 兼容模式、各类中转网关均可），改 `FVA_LLM_BASE_URL`
  与模型名即可切换，业务代码零改动。
* **只暴露必要的网络能力**：本模块只能向配置好的模型端点发请求，
  不支持网页搜索、也不接受任意 URL，遵守「禁止网络搜索」红线。
* **可测试**：`ScriptedLLM` 用固定脚本模拟模型返回，供单元测试与无网络演示使用。
"""

from __future__ import annotations

import dataclasses
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import LLMSettings


class LLMError(RuntimeError):
    """模型调用失败（网络、鉴权、限流、返回格式异常）。"""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class AssistantTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    """模型客户端统一接口：真实模型与离线策略引擎都实现它。"""

    name: str

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AssistantTurn: ...


class OpenAICompatibleLLM:
    """OpenAI 兼容的 Chat Completions 客户端（带重试与超时）。"""

    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or LLMSettings.from_env()
        self.name = f"openai-compatible:{self.settings.model}"
        if not self.settings.api_key:
            raise LLMError(
                "未检测到大模型密钥：请设置环境变量 FVA_LLM_API_KEY（或 DEEPSEEK_API_KEY / OPENAI_API_KEY），"
                "或改用离线模式（--mode offline）。"
            )

    # ---------------------------------------------------------- 请求
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AssistantTurn:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = f"{self.settings.base_url}/chat/completions"

        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.settings.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return self._parse(data)
            except urllib.error.HTTPError as exc:
                detail = _read_error(exc)
                last_error = LLMError(f"模型接口返回 HTTP {exc.code}：{detail}")
                if exc.code in (408, 409, 425, 429) or exc.code >= 500:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise last_error from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = LLMError(f"模型接口不可达：{exc}")
                time.sleep(min(2 ** attempt, 8))
            except json.JSONDecodeError as exc:
                raise LLMError(f"模型接口返回非法 JSON：{exc}") from exc
        raise last_error or LLMError("模型调用失败")

    # ---------------------------------------------------------- 解析
    @staticmethod
    def _parse(data: dict) -> AssistantTurn:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"模型返回缺少 choices：{str(data)[:300]}")
        message = choices[0].get("message") or {}
        text = _content_to_text(message.get("content"))
        calls: list[ToolCall] = []
        for index, item in enumerate(message.get("tool_calls") or []):
            function = item.get("function") or {}
            name = function.get("name") or ""
            raw_args = function.get("arguments")
            arguments = _parse_arguments(raw_args)
            calls.append(ToolCall(id=item.get("id") or f"call_{index}", name=name, arguments=arguments))
        return AssistantTurn(
            text=text,
            tool_calls=calls,
            usage=data.get("usage") or {},
            raw=data,
        )


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in ("text", "output_text"):
                parts.append(item.get("text") or "")
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _parse_arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except (json.JSONDecodeError, TypeError):
        return {"_raw": str(raw)[:2000]}


def _read_error(exc: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8", "replace"))
    except Exception:
        return exc.reason or "未知错误"
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)[:300]
    return str(error or payload)[:300]


class ScriptedLLM:
    """脚本化「假模型」：按顺序返回预置回合，用于单元测试与演示。

    它不参与真实判定，只用于验证 Agent 循环、工具调度与事件流是否正确。
    """

    name = "scripted"

    def __init__(self, turns: list[AssistantTurn]):
        self.turns = list(turns)
        self.calls = 0
        self.history: list[list[dict]] = []

    def chat(self, messages, tools=None, tool_choice=None, max_tokens=None, temperature=None):
        self.history.append(list(messages))
        self.calls += 1
        if not self.turns:
            return AssistantTurn(text="（脚本已用尽）")
        return self.turns.pop(0)


def build_llm(mode: str = "auto", settings: LLMSettings | None = None) -> LLMClient:
    """按模式构造模型客户端：auto 在缺少密钥时抛出明确错误，由上层选择离线模式。"""
    settings = settings or LLMSettings.from_env()
    if mode == "offline":
        raise ValueError("离线模式由 fva.offline 提供，请通过 pipeline 构造")
    return OpenAICompatibleLLM(settings)


# ---------------------------------------------------------------- 连通性自检


_HINTS = {
    401: "密钥无效或已过期：请到模型控制台确认 API Key 是否复制完整（注意首尾空格）。",
    403: "密钥没有该模型的访问权限，或账号被限制。",
    404: "接口路径不对：BASE_URL 应填服务根地址（如 https://api.deepseek.com），不要带 /v1/chat/completions。",
    422: "请求被拒绝：模型名可能不存在，检查 FVA_LLM_MODEL。",
    429: "触发限流或余额不足：稍后重试或检查账户余额。",
}


def check_connection(settings: LLMSettings | None = None) -> tuple[bool, str]:
    """用一次极小的对话请求验证「密钥 + 地址 + 模型」是否可用。

    返回 (是否成功, 人类可读的说明)。失败时尽量给出可操作的原因。
    """
    settings = settings or LLMSettings.from_env()
    if not settings.api_key:
        return False, "未配置密钥：请执行 python -m fva.cli setup，或在项目根目录的 .env 中填写 FVA_LLM_API_KEY。"
    probe = dataclasses.replace(settings, max_retries=0, timeout=min(settings.timeout, 30.0), max_tokens=16)
    started = time.time()
    try:
        client = OpenAICompatibleLLM(probe)
        turn = client.chat(
            [{"role": "user", "content": "只回复两个字：正常"}],
            max_tokens=16,
            temperature=0.0,
        )
    except LLMError as exc:
        message = str(exc)
        hint = ""
        for code, text in _HINTS.items():
            if f"HTTP {code}" in message:
                hint = f"\n  提示：{text}"
                break
        if not hint and ("不可达" in message or "超时" in message or "timed out" in message.lower()):
            hint = ("\n  提示：检查网络/代理；公司内网可能需要设置 HTTPS_PROXY，"
                    "或把 FVA_LLM_BASE_URL 指向可访问的中转地址。")
        return False, f"{message}{hint}"
    elapsed = time.time() - started
    reply = (turn.text or "").strip().replace("\n", " ")[:40]
    return True, (f"连接成功（{elapsed:.2f}s，模型回复：「{reply}」；"
                  f"密钥来源：{settings.key_source or '未知'}）")
