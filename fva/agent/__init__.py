"""智能体模块：自研 ReAct 框架、工具注册表与提示词。"""

from .prompts import REPORT_REQUEST, SYSTEM_PROMPT, build_user_prompt
from .react import AgentReport, ReActAgent
from .tools import ToolContext, ToolSpec, build_registry, normalize_verdict, tool_schemas

__all__ = [
    "AgentReport",
    "ReActAgent",
    "REPORT_REQUEST",
    "SYSTEM_PROMPT",
    "ToolContext",
    "ToolSpec",
    "build_registry",
    "build_user_prompt",
    "normalize_verdict",
    "tool_schemas",
]
