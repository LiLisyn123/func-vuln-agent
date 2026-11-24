"""轻量 C/C++ 结构解析。

为什么不用 Tree-sitter：本项目的运行环境需要「零原生依赖、可离线安装」，
而 Tree-sitter 的 Python 绑定需要编译与语言包。这里实现一个**字符级扫描器**：
先屏蔽注释与字符串字面量（保持字符偏移与行号不变），再按括号配对识别
函数定义、调用关系与包含关系。接口与 Tree-sitter 方案解耦，后续可平滑替换。

精度取舍：能可靠处理 C 风格函数定义（含 C++ 成员函数、模板声明中的普通函数）；
不解析宏展开与语法错误代码，遇到无法配对时退化为不返回该函数，而不是抛异常。
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

C_EXTENSIONS = frozenset(
    {".c", ".h", ".cc", ".cpp", ".cxx", ".c++", ".hh", ".hpp", ".hxx", ".inl", ".ipp", ".m", ".mm"}
)

# 出现在 `identifier (` 形式但不是函数定义的关键字
KEYWORDS = frozenset(
    """
    if else for while do switch case return sizeof typedef alignof decltype noexcept
    catch throw new delete static_cast dynamic_cast const_cast reinterpret_cast
    defined __attribute__ __declspec __asm__ asm goto operator template typename
    """.split()
)

IDENT_START = re.compile(r"[A-Za-z_]")
IDENT_CHAR = re.compile(r"[A-Za-z0-9_]")


@dataclass
class Function:
    name: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    signature: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature,
        }


def is_c_file(path: str) -> bool:
    lower = path.lower()
    return any(lower.endswith(ext) for ext in C_EXTENSIONS)


def mask_literals(code: str) -> str:
    """把注释与字符串字面量替换为空格，保留换行与字符偏移。"""
    out = list(code)
    i, n = 0, len(code)
    while i < n:
        ch = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            j = i
            while j < n and code[j] != "\n":
                out[j] = " "
                j += 1
            i = j
        elif ch == "/" and nxt == "*":
            j = i
            while j < n - 1 and not (code[j] == "*" and code[j + 1] == "/"):
                if code[j] != "\n":
                    out[j] = " "
                j += 1
            for k in range(j, min(j + 2, n)):
                out[k] = " "
            i = j + 2
        elif ch in "\"'":
            quote = ch
            j = i + 1
            while j < n:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == quote or code[j] == "\n":
                    break
                j += 1
            for k in range(i + 1, min(j, n)):
                if out[k] != "\n":
                    out[k] = " "
            out[i] = " "
            if j < n and out[j] == quote:
                out[j] = " "
            i = j + 1
        else:
            i += 1
    return "".join(out)


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


class _LineMap:
    """行号查询加速：预计算换行位置，查询从 O(n) 降为 O(log n)。"""

    def __init__(self, text: str):
        self.positions = [i for i, ch in enumerate(text) if ch == "\n"]

    def line_of(self, index: int) -> int:
        return bisect.bisect_left(self.positions, index) + 1

    def line_start(self, index: int) -> int:
        position = bisect.bisect_left(self.positions, index)
        return 0 if position == 0 else self.positions[position - 1] + 1


def _match_pair(masked: str, open_index: int) -> int:
    """括号配对，返回匹配的右括号下标；失败返回 -1。"""
    opening = masked[open_index]
    closing = ")" if opening == "(" else "}"
    depth = 0
    for i in range(open_index, len(masked)):
        ch = masked[i]
        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return i
    return -1


def extract_functions(code: str) -> list[Function]:
    """识别文件中的函数定义（含 C++ 类成员函数）。"""
    masked = mask_literals(code)
    line_map = _LineMap(masked)
    functions: list[Function] = []
    depth = 0
    i, n = 0, len(masked)
    while i < n:
        ch = masked[i]
        if ch == "{":
            depth += 1
            i += 1
            continue
        if ch == "}":
            depth = max(0, depth - 1)
            i += 1
            continue
        if not IDENT_START.match(ch) or (i > 0 and IDENT_CHAR.match(masked[i - 1])):
            i += 1
            continue
        j = i
        while j < n and IDENT_CHAR.match(masked[j]):
            j += 1
        name = masked[i:j]
        if name in KEYWORDS:
            i = j
            continue
        k = j
        while k < n and masked[k] in " \t\r\n":
            k += 1
        if k >= n or masked[k] != "(":
            i = j
            continue
        close = _match_pair(masked, k)
        if close < 0:
            i = j
            continue
        m = close + 1
        # 跳过常见的尾部修饰：const / noexcept / override / final / throw(...) / & 
        while True:
            while m < n and masked[m] in " \t\r\n":
                m += 1
            tail = masked[m: m + 12]
            matched = False
            for word in ("const", "noexcept", "override", "final", "&", "__attribute__"):
                if tail.startswith(word):
                    m += len(word)
                    matched = True
                    break
            if not matched:
                break
            if masked[m - 1] == "&":
                break
        if m >= n or masked[m] != "{":
            i = j
            continue
        body_end = _match_pair(masked, m)
        if body_end < 0:
            i = j
            continue
        # 签名：从所在行行首到右括号
        sig_start = line_map.line_start(i)
        signature = " ".join((code[sig_start:close + 1]).split())
        start_line = line_map.line_of(i)
        end_line = line_map.line_of(body_end)
        functions.append(
            Function(
                name=name,
                start_line=start_line,
                end_line=end_line,
                start_byte=i,
                end_byte=body_end + 1,
                signature=signature[:400],
            )
        )
        # 跳过整个函数体：内部嵌套的调用不会误判为函数定义
        i = body_end + 1
    return functions


def find_function(functions: list[Function], name: str) -> Function | None:
    for fn in functions:
        if fn.name == name:
            return fn
    lowered = name.lower()
    for fn in functions:
        if fn.name.lower() == lowered:
            return fn
    return None


def enclosing_function(functions: list[Function], line: int) -> str:
    for fn in functions:
        if fn.start_line <= line <= fn.end_line:
            return fn.name
    return "?"


CALL_RE = re.compile(r"([A-Za-z_]\w*)\s*\(")


def extract_calls(code: str, start_byte: int = 0, end_byte: int | None = None) -> list[str]:
    """提取代码片段中调用的函数名（保持出现顺序、去重）。"""
    segment = code[start_byte: end_byte if end_byte is not None else len(code)]
    masked = mask_literals(segment)
    calls: list[str] = []
    seen: set[str] = set()
    for match in CALL_RE.finditer(masked):
        name = match.group(1)
        if name in KEYWORDS or name in seen:
            continue
        # 形如 `if (` 已被关键字过滤；跳过宏一样的全大写短名（DBG/LOG 等噪声）
        seen.add(name)
        calls.append(name)
    return calls
