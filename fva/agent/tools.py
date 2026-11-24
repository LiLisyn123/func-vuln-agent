"""Agent 工具层：把「领域知识检索」与「仓库上下文挖掘」封装成 LLM 可自主调用的工具。

每个工具包含三部分：JSON Schema 描述（给模型看）、执行函数（真正干活）、
以及可观测的副作用（写事件、累计计数）。工具只提供事实与证据，不做判定，
判定由 Agent 综合证据后通过 `submit_report` 提交。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import config
from ..knowledge import KnowledgeBase
from ..repo import RepoTools, SnapshotError, SnapshotManager
from ..textutil import function_name, function_signature, truncate

VERDICT_VULN = "有漏洞"
VERDICT_SAFE = "无漏洞"


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    handler: Callable[["ToolContext", dict], str]
    requires_repo: bool = False
    terminal: bool = False


@dataclass
class ToolContext:
    """一次检测会话的共享上下文（知识库、仓库、预算与事件回调）。"""

    func_code: str
    commit_url: str
    target: int
    kb: KnowledgeBase
    settings: config.Settings
    idx: int | None = None
    snapshot_manager: SnapshotManager | None = None
    repo: RepoTools | None = None
    snapshot_meta: dict = field(default_factory=dict)
    repo_error: str = ""
    repo_attempted: bool = False
    tool_calls: int = 0
    seen_calls: dict[str, int] = field(default_factory=dict)
    kb_queries: list[dict] = field(default_factory=list)
    on_event: Callable[[dict], None] = lambda event: None
    target_name: str = ""

    def emit(self, kind: str, **payload) -> None:
        self.on_event({"type": kind, **payload})

    # ---------------------------------------------------------- 仓库惰性加载
    def ensure_repo(self) -> tuple[RepoTools | None, str]:
        """首次需要仓库证据时才下载快照（证据驱动，避免无谓开销）。"""
        if self.repo is not None:
            return self.repo, ""
        if not self.settings.repo.enabled:
            self.repo_error = "仓库快照能力已被配置关闭（FVA_REPO_ENABLED=0）"
            return None, self.repo_error
        if self.repo_attempted:
            return None, self.repo_error
        self.repo_attempted = True
        manager = self.snapshot_manager or SnapshotManager(self.settings.repo)
        self.snapshot_manager = manager
        rule_text = "父提交版本" if self.target == 1 else "提交后版本"
        self.emit("snapshot", status="start", detail=f"按 version 规则获取{rule_text}的仓库快照")
        try:
            snapshot = manager.ensure_snapshot(self.commit_url, self.target)
        except SnapshotError as exc:
            self.repo_error = str(exc)
            self.emit("snapshot", status="error", detail=self.repo_error)
            return None, self.repo_error
        except Exception as exc:  # 网络/磁盘等意外错误一律降级，不中断检测
            self.repo_error = f"快照获取异常：{exc}"
            self.emit("snapshot", status="error", detail=self.repo_error)
            return None, self.repo_error
        self.repo = RepoTools(snapshot.root)
        self.snapshot_meta = {
            "slug": snapshot.ref.slug,
            "snapshot_sha": snapshot.snapshot_sha,
            "rule": snapshot.rule,
            "source": snapshot.source,
            "cache_hit": snapshot.cache_hit,
        }
        self.emit("snapshot", status="ok", detail=(
            f"{snapshot.ref.slug} @ {snapshot.snapshot_sha[:12]}（{snapshot.rule} 规则，"
            f"来源 {snapshot.source}{'，命中缓存' if snapshot.cache_hit else ''}）"
        ), data=self.snapshot_meta)
        return self.repo, ""

    def repo_or_message(self) -> tuple[RepoTools | None, str]:
        repo, error = self.ensure_repo()
        if repo is None:
            return None, (
                f"[仓库上下文不可用] {error}\n"
                "请仅依据函数源码本身与历史案例作出判断，并在理由中说明该限制。"
            )
        return repo, ""


# ---------------------------------------------------------------- 工具实现


def _tool_search_knowledge(ctx: ToolContext, args: dict) -> str:
    query = str(args.get("query") or "").strip()
    top_k = int(args.get("top_k") or ctx.settings.agent.max_kb_hits)
    top_k = max(1, min(top_k, 8))
    if not query:
        query = function_signature(ctx.func_code) + "\n" + ctx.func_code
    hits = ctx.kb.search(query, top_k=top_k)
    ctx.kb_queries.append({"query": truncate(query, 200, ""), "hits": [
        {"cve": h.case.cve, "cwe": h.case.cwe, "score": h.score, "func": h.case.func_name}
        for h in hits
    ]})
    ctx.emit("kb_hit", count=len(hits), query=truncate(query, 120, ""), data=[
        {"rank": h.rank, "cve": h.case.cve, "cwe": h.case.cwe, "score": h.score,
         "project": h.case.project, "func": h.case.func_name} for h in hits
    ])
    return KnowledgeBase.render_hits(hits, per_case_chars=900)


def _tool_repo_overview(ctx: ToolContext, args: dict) -> str:
    repo, message = ctx.repo_or_message()
    if repo is None:
        return message
    meta = ctx.snapshot_meta
    head = (
        f"仓库：{meta.get('slug', '?')}\n"
        f"快照版本：{str(meta.get('snapshot_sha', ''))[:12]}"
        f"（version 规则：{meta.get('rule', '?')} = "
        f"{'修复提交的父提交' if meta.get('rule') == 'parent' else '修复提交本身'}）\n"
    )
    return head + repo.summary()


def _tool_grep_code(ctx: ToolContext, args: dict) -> str:
    repo, message = ctx.repo_or_message()
    if repo is None:
        return message
    pattern = str(args.get("pattern") or "").strip()
    if not pattern:
        return "（pattern 不能为空）"
    return repo.grep(pattern, glob=str(args.get("glob") or ""), max_hits=int(args.get("max_hits") or 40))


def _tool_read_file(ctx: ToolContext, args: dict) -> str:
    repo, message = ctx.repo_or_message()
    if repo is None:
        return message
    path = str(args.get("path") or "")
    if not path:
        return "（path 不能为空）"
    return repo.read_file(path, _int_or_none(args.get("start_line")), _int_or_none(args.get("end_line")))


def _tool_list_dir(ctx: ToolContext, args: dict) -> str:
    repo, message = ctx.repo_or_message()
    if repo is None:
        return message
    return repo.list_dir(str(args.get("path") or ""))


def _tool_find_function(ctx: ToolContext, args: dict) -> str:
    repo, message = ctx.repo_or_message()
    if repo is None:
        return message
    name = str(args.get("name") or "").strip()
    if not name:
        return "（name 不能为空）"
    return repo.find_function(name)


def _tool_get_function(ctx: ToolContext, args: dict) -> str:
    repo, message = ctx.repo_or_message()
    if repo is None:
        return message
    name = str(args.get("name") or "").strip()
    if not name:
        return "（name 不能为空）"
    return repo.get_function(name, args.get("path") or None)


def _tool_find_callers(ctx: ToolContext, args: dict) -> str:
    repo, message = ctx.repo_or_message()
    if repo is None:
        return message
    name = str(args.get("name") or "").strip()
    if not name:
        return "（name 不能为空）"
    return repo.find_callers(name)


def _tool_find_callees(ctx: ToolContext, args: dict) -> str:
    repo, message = ctx.repo_or_message()
    if repo is None:
        return message
    name = str(args.get("name") or "").strip()
    if not name:
        return "（name 不能为空）"
    return repo.find_callees(name, args.get("path") or None)


def _tool_search_symbol(ctx: ToolContext, args: dict) -> str:
    repo, message = ctx.repo_or_message()
    if repo is None:
        return message
    name = str(args.get("name") or "").strip()
    if not name:
        return "（name 不能为空）"
    return repo.search_symbol(name)


def _tool_glob_files(ctx: ToolContext, args: dict) -> str:
    repo, message = ctx.repo_or_message()
    if repo is None:
        return message
    pattern = str(args.get("pattern") or "").strip()
    if not pattern:
        return "（pattern 不能为空）"
    return repo.glob_files(pattern)


def _tool_submit_report(ctx: ToolContext, args: dict) -> str:
    """终止工具：把最终结论结构化提交（不返回给模型，直接结束循环）。"""
    verdict = normalize_verdict(args.get("verdict"))
    return json.dumps({"accepted": True, "verdict": verdict}, ensure_ascii=False)


# ---------------------------------------------------------------- 工具注册表


def _param(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


def build_registry() -> dict[str, ToolSpec]:
    string = {"type": "string"}
    integer = {"type": "integer"}
    specs = [
        ToolSpec(
            "search_knowledge",
            "在历史 CVE 漏洞案例知识库中检索与当前函数相似的真实案例，返回漏洞描述、"
            "修复要点与漏洞版代码节选。可用于建立「这类缺陷长什么样」的先验。",
            _param({"query": {**string, "description": "检索语句，建议写函数签名 + 关键 API 名，留空则用当前函数源码"},
                    "top_k": {**integer, "description": "返回条数，1-8，默认 5"}}, []),
            _tool_search_knowledge,
        ),
        ToolSpec(
            "repo_overview",
            "查看当前分析所对应的代码仓库快照概要（仓库名、固定版本、目录结构、源文件规模）。",
            _param({}, []),
            _tool_repo_overview, requires_repo=True,
        ),
        ToolSpec(
            "grep_code",
            "在仓库快照的 C/C++ 文件中按正则检索，返回「文件:行号: 内容」。"
            "适合查找某个 API 的所有使用点、某个宏/全局变量的定义与引用。",
            _param({"pattern": {**string, "description": "Python 正则表达式"},
                    "glob": {**string, "description": "可选，按路径过滤，如 src/*.c"},
                    "max_hits": integer}, ["pattern"]),
            _tool_grep_code, requires_repo=True,
        ),
        ToolSpec(
            "read_file",
            "读取仓库中某个文件的内容（建议配合行号区间精读）。",
            _param({"path": {**string, "description": "仓库内相对路径"},
                    "start_line": integer, "end_line": integer}, ["path"]),
            _tool_read_file, requires_repo=True,
        ),
        ToolSpec(
            "list_dir",
            "列出仓库中某个目录下的文件与子目录。",
            _param({"path": string}, []),
            _tool_list_dir, requires_repo=True,
        ),
        ToolSpec(
            "find_function",
            "在仓库中定位某个函数的定义位置（文件与行号范围）。",
            _param({"name": {**string, "description": "函数名"}}, ["name"]),
            _tool_find_function, requires_repo=True,
        ),
        ToolSpec(
            "get_function",
            "取出仓库中某个函数的完整实现源码（用于阅读被调用方的实际行为）。",
            _param({"name": {**string, "description": "函数名"}, "path": string}, ["name"]),
            _tool_get_function, requires_repo=True,
        ),
        ToolSpec(
            "find_callers",
            "在整个仓库中查找调用指定函数的调用方，返回「文件:行号 → 所在函数」。"
            "用于判断参数是否来自不可信输入、调用方是否已做校验。",
            _param({"name": {**string, "description": "被调用函数名"}}, ["name"]),
            _tool_find_callers, requires_repo=True,
        ),
        ToolSpec(
            "find_callees",
            "列出指定函数内部直接调用的其它函数，便于继续追查下游是否安全。",
            _param({"name": {**string, "description": "函数名"}, "path": string}, ["name"]),
            _tool_find_callees, requires_repo=True,
        ),
        ToolSpec(
            "search_symbol",
            "在仓库中查找符号的定义位置（函数/宏/typedef/全局变量）。"
            "例如查找 SIZE_MAX、结构体字段名、缓冲区大小的宏定义。",
            _param({"name": {**string, "description": "符号名"}}, ["name"]),
            _tool_search_symbol, requires_repo=True,
        ),
        ToolSpec(
            "glob_files",
            "按文件名/路径通配符查找文件（相当于 find 命令），如 `**/*exif*.c`。",
            _param({"pattern": {**string, "description": "通配符表达式"}}, ["pattern"]),
            _tool_glob_files, requires_repo=True,
        ),
        ToolSpec(
            "submit_report",
            "提交最终检测结论并结束分析。必须给出判定结果与判定理由；"
            "理由要引用具体证据（行号、函数名、宏、历史案例编号）。",
            _param(
                {
                    "verdict": {"type": "string", "enum": [VERDICT_VULN, VERDICT_SAFE],
                                "description": "判定结果"},
                    "confidence": {"type": "number", "description": "置信度 0-1"},
                    "cwe": {"type": "array", "items": string, "description": "最贴近的 CWE 编号"},
                    "reason": {**string, "description": "判定理由（证据链）"},
                    "key_evidence": {"type": "array", "items": string,
                                     "description": "支撑结论的关键证据条目"},
                },
                ["verdict", "reason"],
            ),
            _tool_submit_report, terminal=True,
        ),
    ]
    return {spec.name: spec for spec in specs}


def tool_schemas(registry: dict[str, ToolSpec], include_repo: bool = True) -> list[dict]:
    """转换为 OpenAI tools 字段。仓库不可用时不暴露仓库工具，避免模型空转。"""
    schemas = []
    for spec in registry.values():
        if spec.requires_repo and not include_repo:
            continue
        schemas.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        })
    return schemas


def normalize_verdict(value: Any) -> str:
    """把模型可能给出的多种写法归一为「有漏洞 / 无漏洞」。"""
    if isinstance(value, bool):
        return VERDICT_VULN if value else VERDICT_SAFE
    if isinstance(value, (int, float)):
        return VERDICT_VULN if int(value) == 1 else VERDICT_SAFE
    text = str(value or "").strip().lower()
    if text in {"1", "vuln", "vulnerable", "true", "是", "有漏洞"}:
        return VERDICT_VULN
    if text in {"0", "safe", "benign", "false", "否", "无漏洞"}:
        return VERDICT_SAFE
    # 否定表述优先：「未发现漏洞」不能被当作「有漏洞」
    if any(neg in text for neg in ("无漏洞", "没有漏洞", "未发现漏洞", "未找到漏洞",
                                   "不存在漏洞", "无任何漏洞", "safe", "benign")):
        return VERDICT_SAFE
    if "有漏洞" in text or "存在漏洞" in text or "vulnerab" in text:
        return VERDICT_VULN
    return VERDICT_VULN  # 兜底：无法解析时按「有漏洞」处理，避免漏报


def _int_or_none(value) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def guess_target_name(code: str) -> str:
    return function_name(code)


def extract_code_identifiers(code: str, limit: int = 12) -> list[str]:
    """提取函数中最可能代表「危险动作」的标识符，用于仓库检索。"""
    danger = re.compile(
        r"\b(memcpy|memmove|strcpy|strcat|sprintf|snprintf|vsnprintf|alloca|realloc|malloc|calloc|"
        r"free|sprintf|sscanf|fgets|gets|read|recv|send|system|popen|exec\w*|printf|fprintf|open|"
        r"fopen|memset|atoi|atol|strtol|strlen|strncpy|strncat)\b"
    )
    seen: list[str] = []
    for match in danger.finditer(code):
        if match.group(1) not in seen:
            seen.append(match.group(1))
        if len(seen) >= limit:
            break
    return seen
