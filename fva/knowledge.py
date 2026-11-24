"""领域知识库（RAG 模块）。

设计要点
--------
1. **成对建库**：课程允许历史案例成对使用，因此把同一提交里的「漏洞版」与
   「修复版」组织成一条案例，额外提炼「历史修复要点」（修复版相对漏洞版新增的
   实质性代码行）。这比只索引单个函数更接近「漏洞模式」本身，也为离线分析器
   提供可解释的依据。
2. **纯 Python BM25**：倒排索引 + 数组压缩，无需向量数据库与嵌入模型下载，
   离线可复现；检索目标是根据代码标识符找到结构相似的历史漏洞。
3. **防污染**：建库时剔除与测试集重合的样本（本课程数据集本身已隔离，
   这里再做一次校验并记录在案）。

对外接口：
    build_kb(force=False)         构建 / 复用语料与索引
    KnowledgeBase.load()          加载知识库
    kb.search(query, top_k)       检索相似案例
    kb.render(hits, limit)        渲染为提示词片段
"""

from __future__ import annotations

import difflib
import gzip
import math
import pickle
import re
from array import array
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from . import config
from .textutil import (
    function_name,
    normalize_function,
    read_jsonl,
    strip_comments,
    tokenize,
    truncate,
)

# 修复要点提炼时忽略的行：纯括号、纯注释、预处理噪声。
TRIVIAL_LINE = re.compile(r"^[\s{}()\[\];,]*$")


@dataclass
class Case:
    """一条历史漏洞案例（成对或单版本）。"""

    case_id: str
    cve: str
    cwe: list[str]
    project: str
    func_name: str
    commit_id: str
    commit_url: str
    file_name: str
    cve_desc: str
    commit_message: str
    vuln_idx: int | None = None
    fixed_idx: int | None = None
    code_vuln: str = ""
    code_fixed: str = ""
    fix_hints: list[str] = field(default_factory=list)

    @property
    def paired(self) -> bool:
        return self.vuln_idx is not None and self.fixed_idx is not None

    def to_json(self) -> dict:
        return {
            "case_id": self.case_id,
            "cve": self.cve,
            "cwe": self.cwe,
            "project": self.project,
            "func_name": self.func_name,
            "commit_id": self.commit_id,
            "commit_url": self.commit_url,
            "file_name": self.file_name,
            "cve_desc": self.cve_desc,
            "commit_message": self.commit_message,
            "vuln_idx": self.vuln_idx,
            "fixed_idx": self.fixed_idx,
            "code_vuln": self.code_vuln,
            "code_fixed": self.code_fixed,
            "fix_hints": self.fix_hints,
        }

    @classmethod
    def from_json(cls, raw: dict) -> "Case":
        return cls(**raw)

    @property
    def search_text(self) -> str:
        """参与 BM25 索引的文本：以漏洞版代码为主体，辅以描述与类型。"""
        parts = [
            f"cwe {' '.join(self.cwe)} cve {self.cve} {self.func_name}",
            self.cve_desc,
            self.commit_message,
            self.code_vuln,
        ]
        return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------- 建库


def _normalize_cwe(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value if v]
    return []


def _extract_fix_hints(vuln_code: str, fixed_code: str, limit: int = 6) -> list[str]:
    """提炼「修复版相对漏洞版新增的实质性代码行」。

    只保留有语义的行（跳过纯括号、纯注释），并按长度截断；这是历史案例中
    「修复动作」的最小表示，用于提示模型关注同类修复模式。
    """
    if not vuln_code or not fixed_code:
        return []
    a = [ln.strip() for ln in strip_comments(vuln_code).splitlines()]
    b = [ln.strip() for ln in strip_comments(fixed_code).splitlines()]
    hints: list[str] = []
    for line in difflib.unified_diff(a, b, lineterm="", n=0):
        if not line.startswith("+") or line.startswith("+++"):
            continue
        text = line[1:].strip()
        if len(text) < 4 or TRIVIAL_LINE.match(text):
            continue
        if text.startswith("#include"):
            continue
        if text not in hints:
            hints.append(truncate(text, 200, ""))
        if len(hints) >= limit:
            break
    return hints


def _group_records(records: list[dict]) -> list[Case]:
    """按 (cve, commit_id) 分组，再按函数名配对，生成案例列表。"""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record.get("cve") or "", record.get("commit_id") or "")].append(record)

    cases: list[Case] = []
    for (cve, commit_id), group in groups.items():
        buckets: dict[str, dict[int, list[dict]]] = defaultdict(lambda: {0: [], 1: []})
        for record in group:
            name = function_name(record.get("func") or "")
            target = int(record.get("target") or 0)
            buckets[name or "_"][target].append(record)

        counter = 0
        for name, bucket in buckets.items():
            paired = min(len(bucket[0]), len(bucket[1]))
            for index in range(paired):
                counter += 1
                cases.append(_make_case(cve, commit_id, bucket[1][index], bucket[0][index], counter))
            for target, extra in ((1, bucket[1][paired:]), (0, bucket[0][paired:])):
                for record in extra:
                    counter += 1
                    cases.append(_make_case(cve, commit_id, record, None, counter))
    cases.sort(key=lambda c: (c.cve, c.commit_id, c.case_id))
    return cases


def _make_case(cve: str, commit_id: str, vuln: dict | None, fixed: dict | None, counter: int) -> Case:
    primary = vuln or fixed or {}
    code_vuln = truncate((vuln or {}).get("func") or "", 3200)
    code_fixed = truncate((fixed or {}).get("func") or "", 2400)
    case_id = f"{cve or 'CVE'}@{commit_id[:10]}#{counter}"
    return Case(
        case_id=case_id,
        cve=cve or primary.get("cve") or "",
        cwe=_normalize_cwe(primary.get("cwe")),
        project=primary.get("project") or "",
        func_name=function_name(primary.get("func") or ""),
        commit_id=commit_id,
        commit_url=primary.get("commit_url") or "",
        file_name=str(primary.get("file_name") or ""),
        cve_desc=(primary.get("cve_desc") or "")[:1500],
        commit_message=(primary.get("commit_message") or "")[:1200],
        vuln_idx=int(vuln["idx"]) if vuln else None,
        fixed_idx=int(fixed["idx"]) if fixed else None,
        code_vuln=code_vuln,
        code_fixed=code_fixed,
        fix_hints=_extract_fix_hints(code_vuln, code_fixed),
    )


def _contamination_report(cases: list[Case], test_path: Path) -> dict:
    """校验知识库与测试集是否重合（红线：知识库不得包含测试样本）。"""
    if not test_path.exists():
        return {"checked": False}
    test_funcs = {normalize_function(r.get("func") or "") for r in read_jsonl(test_path)}
    test_urls = {r.get("commit_url") for r in read_jsonl(test_path)}
    overlap_funcs = sum(1 for c in cases if normalize_function(c.code_vuln) in test_funcs)
    overlap_urls = sum(1 for c in cases if c.commit_url in test_urls)
    return {
        "checked": True,
        "cases": len(cases),
        "test_samples": len(test_funcs),
        "overlap_by_code": overlap_funcs,
        "overlap_by_commit": overlap_urls,
    }


class BM25:
    """BM25 倒排索引：postings 用 array 压缩，避免纯 Python 对象的内存膨胀。"""

    k1 = 1.5
    b = 0.75

    def __init__(self, docs: list[list[str]]):
        self.n_docs = len(docs)
        self.doc_len = array("I")
        postings: dict[str, tuple[array, array]] = {}
        total = 0
        for doc_index, tokens in enumerate(docs):
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.doc_len.append(len(tokens))
            total += len(tokens)
            for token, count in tf.items():
                entry = postings.get(token)
                if entry is None:
                    postings[token] = (array("I", [doc_index]), array("I", [count]))
                else:
                    entry[0].append(doc_index)
                    entry[1].append(count)
        self.postings = postings
        self.avgdl = total / self.n_docs if self.n_docs else 0.0
        self.idf = {
            token: math.log(1.0 + (self.n_docs - len(pair[0]) + 0.5) / (len(pair[0]) + 0.5))
            for token, pair in postings.items()
        }

    def search(self, query_tokens: list[str], top_k: int = 5) -> list[tuple[float, int]]:
        if not query_tokens or not self.n_docs:
            return []
        scores = [0.0] * self.n_docs
        seen: set[str] = set()
        for token in query_tokens:
            if token in seen:
                continue
            seen.add(token)
            entry = self.postings.get(token)
            if entry is None:
                continue
            ids, tfs = entry
            idf = self.idf[token]
            for i in range(len(ids)):
                doc = ids[i]
                tf = tfs[i]
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc] / self.avgdl)
                scores[doc] += idf * tf * (self.k1 + 1) / (tf + norm)
        ranked = sorted(((s, i) for i, s in enumerate(scores) if s > 0), reverse=True)
        return ranked[:top_k]


@dataclass
class Hit:
    case: Case
    score: float
    rank: int


def build_kb(force: bool = False, verbose: bool = True) -> dict:
    """构建知识库语料与索引，返回构建统计信息。"""
    config.ensure_dirs()
    stats = {"corpus": str(config.KB_CORPUS_PATH), "index": str(config.KB_INDEX_PATH)}
    if not force and config.KB_CORPUS_PATH.exists() and config.KB_INDEX_PATH.exists():
        stats["reused"] = True
        return stats
    if not config.KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(f"缺少知识库数据源：{config.KNOWLEDGE_PATH}")

    records = list(read_jsonl(config.KNOWLEDGE_PATH))
    cases = _group_records(records)
    contamination = _contamination_report(cases, config.TEST_PATH)
    if contamination.get("overlap_by_code"):
        # 严格隔离：剔除与测试集代码完全一致的案例，避免泄漏答案
        test_funcs = {normalize_function(r.get("func") or "") for r in read_jsonl(config.TEST_PATH)}
        cases = [c for c in cases if normalize_function(c.code_vuln) not in test_funcs]

    with config.KB_CORPUS_PATH.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(_json_line(case.to_json()))

    docs = [tokenize(case.search_text) for case in cases]
    index = BM25(docs)
    payload = {
        "case_ids": [c.case_id for c in cases],
        "postings": index.postings,
        "doc_len": index.doc_len,
        "avgdl": index.avgdl,
        "idf": index.idf,
        "built_from": str(config.KNOWLEDGE_PATH),
        "contamination": contamination,
    }
    with gzip.open(config.KB_INDEX_PATH, "wb", compresslevel=5) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    stats.update(
        {
            "reused": False,
            "records": len(records),
            "cases": len(cases),
            "paired": sum(1 for c in cases if c.paired),
            "vocab": len(index.postings),
            "avg_doc_len": round(index.avgdl, 1),
            "contamination": contamination,
        }
    )
    if verbose:
        print(
            f"[KB] 原始记录 {len(records)} 条 → 案例 {len(cases)} 条"
            f"（成对 {stats['paired']}）；词表 {stats['vocab']}；平均长度 {stats['avg_doc_len']}"
        )
        print(f"[KB] 语料：{config.KB_CORPUS_PATH}")
        print(f"[KB] 索引：{config.KB_INDEX_PATH}（{config.KB_INDEX_PATH.stat().st_size / 1e6:.1f} MB）")
        print(f"[KB] 污染校验：{contamination}")
    return stats


def _json_line(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False) + "\n"


class KnowledgeBase:
    """知识库查询接口：加载语料 + 倒排索引，提供相似案例检索与提示词渲染。"""

    def __init__(self, cases: list[Case], index: BM25, stats: dict | None = None):
        self.cases = cases
        self.index = index
        self.stats = stats or {}

    @classmethod
    def load(cls, build_if_missing: bool = True) -> "KnowledgeBase":
        if build_if_missing and not (config.KB_CORPUS_PATH.exists() and config.KB_INDEX_PATH.exists()):
            build_kb()
        cases = [Case.from_json(raw) for raw in read_jsonl(config.KB_CORPUS_PATH)]
        with gzip.open(config.KB_INDEX_PATH, "rb") as handle:
            payload = pickle.load(handle)
        index = BM25([])
        index.postings = payload["postings"]
        index.doc_len = payload["doc_len"]
        index.avgdl = payload["avgdl"]
        index.idf = payload["idf"]
        index.n_docs = len(cases)
        if len(payload["case_ids"]) != len(cases):
            raise RuntimeError("知识库语料与索引不一致，请执行 python -m fva.cli build-kb --force 重建")
        return cls(cases, index, payload.get("contamination"))

    # ---------------------------------------------------------- 检索

    def search(self, query: str, top_k: int = 5, *, paired_only: bool = True) -> list[Hit]:
        tokens = tokenize(query)
        ranked = self.index.search(tokens, top_k=max(top_k * 4, 12))
        hits: list[Hit] = []
        for score, doc_index in ranked:
            case = self.cases[doc_index]
            if paired_only and not case.paired:
                continue
            hits.append(Hit(case=case, score=round(score, 2), rank=len(hits) + 1))
            if len(hits) >= top_k:
                break
        return hits

    def search_function(self, code: str, top_k: int = 5) -> list[Hit]:
        """以函数源码为查询检索相似历史案例（签名权重加倍，突出接口相似）。"""
        from .textutil import function_signature

        query = function_signature(code) + "\n" + code
        return self.search(query, top_k=top_k)

    # ---------------------------------------------------------- 渲染

    @staticmethod
    def render_hits(hits: list[Hit], per_case_chars: int = 900) -> str:
        if not hits:
            return "（知识库中未检索到结构相似的历史案例）"
        blocks = []
        for hit in hits:
            case = hit.case
            lines = [
                f"[{hit.rank}] {case.cve or '历史案例'} | {'/'.join(case.cwe) or 'CWE 未知'} | "
                f"{case.project}::{case.func_name or '?'} | 相似度 {hit.score}",
                f"  漏洞披露：{truncate(case.cve_desc, 260, '…')}",
            ]
            if case.commit_message:
                lines.append(f"  修复提交说明：{truncate(' '.join(case.commit_message.split()), 200, '…')}")
            if case.fix_hints:
                hints = "；".join(case.fix_hints[:4])
                lines.append(f"  该案例修复要点（修复版新增，仅供类比）：{truncate(hints, 320, '…')}")
            if case.code_vuln:
                lines.append("  漏洞版代码（节选）：\n```c\n" + truncate(case.code_vuln, per_case_chars, "\n...") + "\n```")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def by_cwe(self, cwe_prefix: str, limit: int = 20) -> list[Case]:
        return [c for c in self.cases if any(x.startswith(cwe_prefix) for x in c.cwe)][:limit]
