"""全局配置：路径、大模型接入、Agent 预算、仓库快照策略。

配置读取优先级（高 → 低）：

1. **环境变量**（`FVA_LLM_API_KEY` 等）——适合 CI / 临时会话；
2. **项目根目录的 `.env` 文件**——适合日常使用：写一次，双击 `.bat`、命令行、
   界面都能读到（双击启动的进程继承不到你某个 cmd 窗口里 `set` 的变量，`.env` 不受此限制）；
3. 代码内置默认值。

主要环境变量 / `.env` 键：

    大模型        FVA_LLM_BASE_URL / FVA_LLM_API_KEY / FVA_LLM_MODEL
                  （兼容 OPENAI_BASE_URL / OPENAI_API_KEY / DEEPSEEK_API_KEY）
    运行预算      FVA_AGENT_MAX_STEPS / FVA_AGENT_MAX_TOOL_CALLS
    仓库快照      FVA_SNAPSHOT_MAX_MB / FVA_GITHUB_TOKEN

默认值针对「DeepSeek + GitHub」的常见组合，且不包含任何密钥（`.env` 已在 .gitignore 中）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- 路径

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FVA_DATA_DIR", ROOT / "dataset"))
CACHE_DIR = Path(os.environ.get("FVA_CACHE_DIR", ROOT / "cache"))
RESULTS_DIR = Path(os.environ.get("FVA_RESULTS_DIR", ROOT / "results"))

ENV_FILE = ROOT / ".env"          # 本地配置（含密钥，不入库）

KNOWLEDGE_PATH = DATA_DIR / "knowledge.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"
EVALUATE_SCRIPT = DATA_DIR / "evaluation.py"

KB_CORPUS_PATH = CACHE_DIR / "kb_corpus.jsonl"      # 成对案例语料（含截断源码）
KB_INDEX_PATH = CACHE_DIR / "kb_index.pkl.gz"       # BM25 倒排索引（pickle+gzip）
REPO_CACHE_DIR = CACHE_DIR / "repos"                # 仓库快照
TARBALL_CACHE_DIR = CACHE_DIR / "tarballs"          # 原始 tarball
SNAPSHOT_META_PATH = CACHE_DIR / "snapshots.json"   # 快照元数据（提交解析结果）
OFFLINE_MODEL_PATH = CACHE_DIR / "offline_model.json"

RESULT_PATH = RESULTS_DIR / "result.jsonl"
REPORT_DIR = RESULTS_DIR / "reports"                # 每条样本的完整报告
TRACE_DIR = RESULTS_DIR / "traces"                  # 每条样本的事件流


def ensure_dirs() -> None:
    """创建运行期需要的目录（幂等）。"""
    for path in (CACHE_DIR, REPO_CACHE_DIR, TARBALL_CACHE_DIR, RESULTS_DIR, REPORT_DIR, TRACE_DIR):
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- .env 文件


def parse_env_text(text: str) -> dict[str, str]:
    """解析 `.env` 文本：支持 `KEY=VALUE`、`export KEY=VALUE`、引号与 `#` 注释。"""
    values: dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value.endswith(" #") and " #" in value:
            value = value.split(" #", 1)[0].strip()
        if key:
            values[key] = value
    return values


_DOTENV_CACHE: dict[str, tuple[float, dict[str, str]]] = {}


def load_env_file(path: Path | str | None = None, *, reload: bool = False) -> dict[str, str]:
    """读取 `.env`（按「路径 + 修改时间」缓存，改文件后下一次调用即生效）。"""
    target = Path(path) if path else ENV_FILE
    try:
        mtime = target.stat().st_mtime
    except OSError:
        return {}
    key = str(target)
    cached = _DOTENV_CACHE.get(key)
    if cached and cached[0] == mtime and not reload:
        return cached[1]
    try:
        text = target.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return {}
    values = parse_env_text(text)
    _DOTENV_CACHE[key] = (mtime, values)
    return values


def write_env_file(updates: dict[str, str], path: Path | str | None = None) -> Path:
    """把键值合并写入 `.env`（保留原有其它键与注释），返回文件路径。"""
    target = Path(path) if path else ENV_FILE
    existing_text = ""
    if target.exists():
        try:
            existing_text = target.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            existing_text = ""
    lines = existing_text.splitlines()
    pending = dict(updates)
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        key = ""
        if stripped and not stripped.startswith(("#", ";")) and "=" in stripped:
            body = stripped[7:].strip() if stripped.lower().startswith("export ") else stripped
            key = body.split("=", 1)[0].strip()
        if key and key in pending:
            output.append(f"{key}={pending.pop(key)}")
        else:
            output.append(line)
    if pending:
        if output and output[-1].strip():
            output.append("")
        output.append("# 由 python -m fva.cli setup 写入")
        for key, value in pending.items():
            output.append(f"{key}={value}")
    target.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    load_env_file(target, reload=True)
    return target


# ---------------------------------------------------------------- 配置项


def _env(*names: str, default: str | None = None) -> str | None:
    """按优先级读取：环境变量 → `.env` 文件 → 默认值。"""
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    dotenv = load_env_file()
    for name in names:
        value = dotenv.get(name)
        if value and value.strip():
            return value.strip()
    return default


def env_source(*names: str) -> str:
    """返回配置项来自哪里（用于诊断「为什么还是离线模式」）。"""
    for name in names:
        if (os.environ.get(name) or "").strip():
            return f"环境变量 {name}"
    dotenv = load_env_file()
    for name in names:
        if (dotenv.get(name) or "").strip():
            return f".env 文件（{name}）"
    return ""


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, default=str(default)) or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, default=str(default)) or default)
    except (TypeError, ValueError):
        return default


@dataclass
class LLMSettings:
    """大模型接入配置（OpenAI Chat Completions 协议）。"""

    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    model: str = "deepseek-chat"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 120.0
    max_retries: int = 2
    key_source: str = ""          # 密钥来源（环境变量名 / .env），用于诊断
    base_url_source: str = ""
    env_file: str = ""            # 实际使用的 .env 路径

    @classmethod
    def from_env(cls) -> "LLMSettings":
        name_hint = ("FVA_LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_BASE_URL")
        key_hint = ("FVA_LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "LLM_TOKEN")
        return cls(
            base_url=_env(*name_hint, default="https://api.deepseek.com").rstrip("/"),
            api_key=_env(*key_hint, default=""),
            model=_env("FVA_LLM_MODEL", "OPENAI_MODEL", "LLM_MODEL", default="deepseek-chat"),
            temperature=_env_float("FVA_LLM_TEMPERATURE", 0.0),
            max_tokens=_env_int("FVA_LLM_MAX_TOKENS", 2048),
            timeout=_env_float("FVA_LLM_TIMEOUT", 120.0),
            max_retries=_env_int("FVA_LLM_MAX_RETRIES", 2),
            key_source=env_source(*key_hint),
            base_url_source=env_source(*name_hint),
            env_file=str(ENV_FILE),
        )

    @property
    def configured(self) -> bool:
        """是否具备调用真实大模型的条件。"""
        return bool(self.api_key and self.base_url and self.model)

    @property
    def masked_key(self) -> str:
        if not self.api_key:
            return ""
        if len(self.api_key) <= 10:
            return self.api_key[:2] + "***"
        return f"{self.api_key[:6]}...{self.api_key[-4:]}"

    def describe(self) -> str:
        if not self.configured:
            return f"{self.model} @ {self.base_url}（未配置密钥）"
        source = self.key_source or "未知来源"
        return f"{self.model} @ {self.base_url}（密钥 {self.masked_key}，来自 {source}）"


@dataclass
class AgentSettings:
    """智能体运行预算：控制自主决策的轮数与工具调用规模，避免无限循环与成本失控。"""

    max_steps: int = 12              # 最多「思考-行动」轮数
    max_tool_calls: int = 20         # 最多工具调用次数
    min_repo_checks: int = 1         # 证据不足时至少要求的仓库核验次数
    max_kb_hits: int = 5             # 单次知识检索返回案例数
    prefetch_kb: bool = True         # 进入 Agent 循环前先做一次知识检索（Pipeline 第②步）
    max_observation_chars: int = 3000  # 单条工具结果回填给模型的字符上限
    max_reason_chars: int = 900      # 判定理由展示上限

    @classmethod
    def from_env(cls) -> "AgentSettings":
        return cls(
            max_steps=_env_int("FVA_AGENT_MAX_STEPS", 12),
            max_tool_calls=_env_int("FVA_AGENT_MAX_TOOL_CALLS", 20),
            min_repo_checks=_env_int("FVA_AGENT_MIN_REPO_CHECKS", 1),
            max_kb_hits=_env_int("FVA_AGENT_MAX_KB_HITS", 5),
            prefetch_kb=_env("FVA_AGENT_PREFETCH_KB", default="1") != "0",
        )


@dataclass
class RepoSettings:
    """仓库快照策略。"""

    enabled: bool = True
    max_snapshot_mb: int = 300       # 解压后软上限，超过仍可分析但会提示
    download_timeout: float = 300.0
    github_token: str = ""

    @classmethod
    def from_env(cls) -> "RepoSettings":
        return cls(
            enabled=_env("FVA_REPO_ENABLED", default="1") != "0",
            max_snapshot_mb=_env_int("FVA_SNAPSHOT_MAX_MB", 300),
            download_timeout=_env_float("FVA_SNAPSHOT_TIMEOUT", 300.0),
            github_token=_env("FVA_GITHUB_TOKEN", "GITHUB_TOKEN", default=""),
        )


@dataclass
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings.from_env)
    agent: AgentSettings = field(default_factory=AgentSettings.from_env)
    repo: RepoSettings = field(default_factory=RepoSettings.from_env)
    root: Path = ROOT

    @classmethod
    def load(cls) -> "Settings":
        ensure_dirs()
        return cls()


SETTINGS = Settings.load()
