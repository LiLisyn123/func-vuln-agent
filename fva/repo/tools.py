"""仓库上下文挖掘工具。

Agent 通过这些工具「按需」取证：查调用方、被调用方、全局变量/宏/类型定义、
读取文件片段、在整个仓库中检索符号。所有工具都返回**紧凑文本**，并强制截断，
以免单次观察污染上下文预算。工具本身不做任何判定，只提供事实。

性能取舍：
* 首次调用时建立「C/C++ 文件清单 + 函数符号表」的内存索引（大仓库约数秒）；
* 每次 grep / 调用关系查询都有文件数、字节数与结果条数上限；
* 索引结果可落盘缓存，同一快照二次使用直接复用。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import cparse

DEFAULT_MAX_FILES = 20000
DEFAULT_MAX_BYTES = 120 * 1024 * 1024     # 单次全仓扫描的字节上限
MAX_FILE_BYTES = 2 * 1024 * 1024
SKIP_DIRS = {
    ".git", ".svn", ".hg", "node_modules", "third_party", "ThirdParty", "external",
    "build", "dist", "out", ".github", ".vscode", ".idea", "__pycache__",
}


class RepoTools:
    """封装一个仓库快照上的只读查询能力。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._files: list[Path] | None = None
        self._symbols: dict[str, list[tuple[str, int, int]]] | None = None
        self._text_cache: dict[str, str] = {}

    # ---------------------------------------------------------- 基础
    def _iter_files(self) -> list[Path]:
        if self._files is not None:
            return self._files
        files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for name in filenames:
                if cparse.is_c_file(name):
                    files.append(Path(dirpath) / name)
                    if len(files) >= DEFAULT_MAX_FILES:
                        break
            if len(files) >= DEFAULT_MAX_FILES:
                break
        files.sort(key=lambda p: str(p).lower())
        self._files = files
        return files

    def rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root)).replace("\\", "/")
        except ValueError:
            return str(path)

    def _read(self, path: Path) -> str:
        key = str(path)
        if key in self._text_cache:
            return self._text_cache[key]
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                text = ""
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        self._text_cache[key] = text
        return text

    def summary(self) -> str:
        files = self._iter_files()
        top: dict[str, int] = {}
        for path in files:
            try:
                rel = path.relative_to(self.root)
            except ValueError:
                continue
            top[rel.parts[0] if len(rel.parts) > 1 else "."] = top.get(
                rel.parts[0] if len(rel.parts) > 1 else ".", 0) + 1
        parts = [f"{k}（{v} 个文件）" for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:12]]
        return (
            f"仓库根目录：{self.root.name}\n"
            f"C/C++ 源文件：{len(files)} 个\n"
            f"主要目录：{', '.join(parts) if parts else '（无）'}"
        )

    # ---------------------------------------------------------- 符号索引
    def symbols(self) -> dict[str, list[tuple[str, int, int]]]:
        """全仓函数定义索引：函数名 → [(相对路径, 起始行, 结束行)]。"""
        if self._symbols is not None:
            return self._symbols
        cache_path = self.root / ".fva_symbols.json"
        if cache_path.exists():
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                self._symbols = {k: [tuple(x) for x in v] for k, v in raw.items()}
                return self._symbols
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        index: dict[str, list[tuple[str, int, int]]] = {}
        for path in self._iter_files():
            text = self._read(path)
            if not text:
                continue
            for fn in cparse.extract_functions(text):
                index.setdefault(fn.name, []).append((self.rel(path), fn.start_line, fn.end_line))
        # 只保留定义不超过 40 处的符号，避免宏同名噪声
        self._symbols = {k: v for k, v in index.items() if len(v) <= 40}
        try:
            cache_path.write_text(json.dumps(self._symbols, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        return self._symbols

    # ---------------------------------------------------------- 查询工具
    def grep(self, pattern: str, glob: str = "", max_hits: int = 40) -> str:
        """按正则/子串在全仓 C/C++ 文件中检索，返回「文件:行号: 内容」。"""
        try:
            regex = re.compile(pattern)
        except re.error:
            regex = re.compile(re.escape(pattern))
        glob_re = _glob_to_regex(glob) if glob else None
        hits: list[str] = []
        scanned = 0
        for path in self._iter_files():
            rel = self.rel(path)
            if glob_re and not glob_re.search(rel):
                continue
            text = self._read(path)
            if not text:
                continue
            scanned += len(text)
            if scanned > DEFAULT_MAX_BYTES:
                hits.append("…（已达到扫描字节上限，结果被截断）")
                break
            for line_no, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{rel}:{line_no}: {line.strip()[:220]}")
                    if len(hits) >= max_hits:
                        hits.append(f"…（共达 {max_hits} 条上限）")
                        return "\n".join(hits)
        return "\n".join(hits) if hits else "（无匹配）"

    def read_file(self, rel_path: str, start: int | None = None, end: int | None = None,
                  max_chars: int = 12000) -> str:
        path = (self.root / rel_path).resolve()
        if not str(path).startswith(str(self.root.resolve())):
            return "（非法路径）"
        if not path.is_file():
            return f"（文件不存在：{rel_path}）"
        text = self._read(path)
        lines = text.splitlines()
        if start or end:
            lo = max(0, (start or 1) - 1)
            hi = min(len(lines), (end or lo + 80))
            chunk = lines[lo:hi]
            header = f"# {rel_path} 第 {lo + 1}-{hi} 行（共 {len(lines)} 行）"
        else:
            chunk = lines
            header = f"# {rel_path}（共 {len(lines)} 行）"
        body = "\n".join(f"{lo + i + 1:>6}| {line}" for i, line in enumerate(chunk))
        if len(body) > max_chars:
            body = body[:max_chars] + "\n…（已截断）"
        return f"{header}\n{body}"

    def list_dir(self, rel_path: str = "", max_entries: int = 60) -> str:
        target = (self.root / rel_path).resolve()
        if not str(target).startswith(str(self.root.resolve())):
            return "（非法路径）"
        if not target.is_dir():
            return f"（目录不存在：{rel_path or '.'}）"
        dirs, files = [], []
        for entry in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if entry.name.startswith(".") or entry.name in SKIP_DIRS:
                continue
            (dirs if entry.is_dir() else files).append(entry.name)
        shown = [d + "/" for d in dirs[:max_entries]] + files[:max_entries]
        return "\n".join(shown) if shown else "（空目录）"

    def find_function(self, name: str, max_hits: int = 8) -> str:
        """定位函数定义（返回文件与行号，便于随后 read_file 精读）。"""
        defs = self.symbols().get(name)
        if not defs:
            # 退化为正则签名搜索
            return self.grep(rf"\b{re.escape(name)}\s*\([^;{{}}]*\)\s*\{{", max_hits=max_hits)
        return "\n".join(f"{path}:{start}-{end}" for path, start, end in defs[:max_hits])

    def get_function(self, name: str, rel_path: str | None = None, max_chars: int = 6000) -> str:
        """取出指定函数的完整源码（用于阅读被调用方实现）。"""
        defs = self.symbols().get(name) or []
        if rel_path:
            defs = [d for d in defs if d[0] == rel_path] or defs
        if not defs:
            return f"（未找到函数 {name} 的定义）"
        path, start, end = defs[0]
        text = self._read(self.root / path)
        lines = text.splitlines()[start - 1:end]
        body = "\n".join(lines)
        if len(body) > max_chars:
            body = body[:max_chars] + "\n…（已截断）"
        return f"# {path}:{start}-{end}\n{body}"

    def find_callers(self, name: str, max_hits: int = 20) -> str:
        """全仓查找调用方：返回「文件:行号 → 所在函数」。"""
        call_re = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\(")
        definitions = {(path, start, end) for path, start, end in (self.symbols().get(name) or [])}
        results: list[str] = []
        scanned = 0
        for path in self._iter_files():
            text = self._read(path)
            if not text or name not in text:
                continue
            scanned += len(text)
            if scanned > DEFAULT_MAX_BYTES:
                break
            rel = self.rel(path)
            own_ranges = [(start, end) for file, start, end in definitions if file == rel]
            funcs = cparse.extract_functions(text)
            for line_no, line in enumerate(text.splitlines(), 1):
                if not call_re.search(line):
                    continue
                if any(start <= line_no <= end for start, end in own_ranges):
                    continue  # 跳过函数自身的定义/实现体
                caller = cparse.enclosing_function(funcs, line_no)
                results.append(f"{rel}:{line_no} → {caller}()  |  {line.strip()[:140]}")
                if len(results) >= max_hits:
                    return "\n".join(results) + "\n…（已达到上限）"
        return "\n".join(results) if results else f"（未发现 {name} 的调用方）"

    def find_callees(self, name: str, rel_path: str | None = None) -> str:
        """列出指定函数内部直接调用的函数名。"""
        defs = self.symbols().get(name) or []
        if not defs:
            return f"（未找到函数 {name} 的定义，无法分析其调用）"
        path, start, end = defs[0]
        if rel_path:
            for candidate in defs:
                if candidate[0] == rel_path:
                    path, start, end = candidate
                    break
        text = self._read(self.root / path)
        lines = text.splitlines()
        body = "\n".join(lines[start - 1:end])
        calls = cparse.extract_calls(body)
        if not calls:
            return f"（{name} 内部没有直接调用）"
        return ", ".join(calls[:40])

    def search_symbol(self, name: str, max_hits: int = 25) -> str:
        """查找符号的定义位置（函数/宏/类型/全局变量）。"""
        pattern = rf"^\s*[^/\n]*\b{re.escape(name)}\b[^;\n]*[;{{=]"
        return self.grep(pattern, max_hits=max_hits)

    def glob_files(self, pattern: str, max_hits: int = 40) -> str:
        """按文件名/路径通配符查找文件（替代 find 命令）。"""
        regex = _glob_to_regex(pattern)
        matches = [self.rel(p) for p in self._iter_files() if regex.search(self.rel(p))]
        if not matches:
            return "（无匹配文件）"
        return "\n".join(matches[:max_hits])


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """把 * 与 ? 通配符转换为正则，支持 `**/` 前缀。"""
    pattern = pattern.strip()
    escaped = re.escape(pattern).replace(r"\*\*/", "(?:.*/)?").replace(r"\*\*", ".*")
    escaped = escaped.replace(r"\*", "[^/]*").replace(r"\?", "[^/]")
    return re.compile(escaped, re.I)
