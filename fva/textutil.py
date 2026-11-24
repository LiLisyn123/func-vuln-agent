"""文本与 C/C++ 代码处理工具：分词、函数签名提取、截断、JSON 读写。

这些函数被知识库检索、仓库工具与评测脚本共同使用，保持无副作用、易测试。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator

# C/C++ 关键字与常见噪声词：它们不携带「这段代码像不像某类漏洞」的信息。
C_KEYWORDS = frozenset(
    """
    auto break case char const continue default do double else enum extern float for goto if
    inline int long register restrict return short signed sizeof static struct switch typedef
    union unsigned void volatile while bool true false nullptr NULL define include ifdef ifndef
    endif elif pragma typedef typename class public private protected virtual override namespace
    using template new delete this operator throw try catch const_cast static_cast
    """.split()
)

# 代码中高频但对「漏洞类型」判别力很弱的标识符。
NOISE_IDENTIFIERS = frozenset(
    """
    ret retval res result rc err error len length size count num n i j k x y z p q r s t
    buf buffer dst src str ptr data value val tmp temp out in arg args param params
    int char void const static unsigned long short float double bool
    """.split()
)

TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9_]+")
CAMEL_SPLIT = re.compile(r"[A-Z]+[a-z]*|[a-z]+|[0-9]+")
STRING_BLOCK = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
COMMENT_BLOCK = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)

_STRING_LITERAL = STRING_BLOCK


def strip_comments(code: str) -> str:
    """去掉注释但保留字符串字面量与换行，避免破坏后续行号。"""
    def repl(match: re.Match[str]) -> str:
        return "\n" * match.group(0).count("\n")

    # 先保护字符串，再删注释，最后还原
    holders: list[str] = []

    def hold(match: re.Match[str]) -> str:
        holders.append(match.group(0))
        return f"\x00{len(holders) - 1}\x00"

    protected = _STRING_LITERAL.sub(hold, code)
    cleaned = COMMENT_BLOCK.sub(repl, protected)
    return re.sub(r"\x00(\d+)\x00", lambda m: holders[int(m.group(1))], cleaned)


def tokenize(text: str, *, drop_noise: bool = False) -> list[str]:
    """代码/文本分词。

    规则：按非标识符字符切分，再按 camelCase / snake_case 拆词，小写化；
    过滤长度小于 2 的词、纯数字、C 关键字，可选过滤高频噪声标识符。
    """
    if not text:
        return []
    text = strip_comments(text)
    out: list[str] = []
    for word in TOKEN_SPLIT.split(text):
        if not word:
            continue
        for sub in CAMEL_SPLIT.findall(word):
            sub = sub.lower()
            if len(sub) < 2 or sub.isdigit() or sub in C_KEYWORDS:
                continue
            if drop_noise and sub in NOISE_IDENTIFIERS:
                continue
            out.append(sub)
    return out


def truncate(text: str, limit: int, tail: str = "\n...（已截断）") -> str:
    """按字符截断，避免把超长函数原样塞进提示词。"""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + tail


SIGNATURE_RE = re.compile(
    r"^\s*(?P<sig>[A-Za-z_][\w\s\*&:<>,]*(?:\b[A-Za-z_]\w*)\s*\([^;{}]*\))\s*(?:\{|$)",
    re.M,
)


def function_signature(code: str) -> str:
    """抽取函数首行签名（用于知识检索的查询构造与去重）。"""
    head = strip_comments(code).lstrip()
    match = SIGNATURE_RE.search(head)
    if match:
        return " ".join(match.group("sig").split())[:400]
    # 退化：取第一个 '{' 之前的文本
    idx = head.find("{")
    return " ".join(head[: idx if idx > 0 else 200].split())[:400]


NAME_RE = re.compile(r"([A-Za-z_]\w*)\s*\(")


def function_name(code: str) -> str:
    """从签名中抽取函数名；失败时返回空串。"""
    sig = function_signature(code)
    names = NAME_RE.findall(sig)
    if not names:
        return ""
    # 形如 `static int foo(...)`：最后一个左括号前的标识符是函数名
    return names[-1]


def normalize_function(code: str) -> str:
    """归一化函数源码：去注释、合并空白，用于判断两条记录是否同一函数。"""
    return re.sub(r"\s+", " ", strip_comments(code or "")).strip()


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """逐行读取 JSONL，跳过空行。"""
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    """写入 JSONL，返回写入条数。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def dump_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
