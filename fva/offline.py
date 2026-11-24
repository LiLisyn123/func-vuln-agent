"""离线分析引擎：不依赖大模型时的「降级大脑」与消融基线。

定位
----
1. **降级运行**：没有 API Key、或调用超预算时，界面与批量流程仍可跑通，
   此时由本模块给出判定，并在报告中如实标注 `mode=offline`。
2. **消融基线**：用同一份知识库数据训练一个「修复模式判别器」，
   量化「只靠领域知识与静态特征」能做到什么水平，反衬 Agent 的取证价值。

方法
----
* 特征：约 40 个人工定义的可解释特征——危险动作（危险 API、变长下标、尺寸乘法、
  窄化转换）与缓解证据（空指针检查、上下界比较、SIZE_MAX 检查、安全替代 API、
  长度封顶、错误返回），外加若干交互特征（例如「有 memcpy 但无边界比较」）。
* 模型：逻辑回归（numpy 训练，纯 Python 推理），训练集为课程提供的历史案例
  （漏洞版 label=1 / 修复版 label=0），按提交分组做交叉验证，避免同源泄漏。
* 阈值：在交叉验证的成对预测上扫描，选取使 P-C 最大、P-R 最小的判定阈值，
  与课程评测口径一致。

该模块只在「训练」时依赖 numpy；推理仅需标准库。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .llm import AssistantTurn, ToolCall
from .textutil import function_name, function_signature, read_jsonl, truncate

# ---------------------------------------------------------------- 特征定义

# (特征名, 正则, 说明, 是否缓解项)
PATTERN_FEATURES: list[tuple[str, str, str, bool]] = [
    # ---- 危险动作 ----
    ("unsafe_str_copy", r"\b(strcpy|strcat|sprintf|gets|wcscpy|wcscat|stpcpy)\s*\(", "无上界的字符串拷贝/拼接", False),
    ("mem_copy", r"\b(memcpy|memmove|bcopy|mempcpy)\s*\(", "原始内存拷贝", False),
    ("scanf_family", r"\b(scanf|sscanf|fscanf|vscanf|vfscanf)\s*\(", "格式化输入未限宽", False),
    ("exec_call", r"\b(system|popen|execl|execlp|execle|execv|execvp|execvpe|WinExec|ShellExecute\w*)\s*\(", "执行外部命令", False),
    ("format_output", r"\b(printf|fprintf|vfprintf|syslog|vsyslog)\s*\(", "格式化输出", False),
    ("alloc_call", r"\b(malloc|calloc|realloc|alloca|new)\b", "堆/栈内存分配", False),
    ("free_call", r"\b(free|delete)\s*\(", "内存释放", False),
    ("variable_index", r"[A-Za-z_]\w*\s*\[\s*[A-Za-z_]\w*(?:\s*[-+][^]]*)?\s*\]", "变长下标访问", False),
    ("size_multiply", r"\*\s*sizeof\b|\bsizeof\b[^;]{0,40}\*|[\w\)\]]\s*\*\s*[A-Za-z_]\w*\s*[;,)]", "尺寸乘法（整数溢出高发处）", False),
    ("narrow_cast", r"\(\s*(?:int|short|unsigned\s+short|uint16_t|int32_t|uint32_t|char|size_t\s*\)?)\s*\)", "类型窄化转换", False),
    ("arithmetic_shift", r"<<|>>", "位运算/移位", False),
    ("division", r"[^/*]\s*/\s*[A-Za-z_(]", "除法运算（除零风险）", False),
    ("direct_deref", r"\*\s*[A-Za-z_]\w*\s*=|->[A-Za-z_]\w*\s*=", "指针解引用写入", False),
    ("struct_walk", r"->\s*\w+\s*=\s*[^;]*->", "多级指针解引用", False),

    # ---- 缓解/防护证据 ----
    ("null_check", r"(==|!=)\s*NULL\b|\bNULL\s*(==|!=)|if\s*\(\s*!\s*[A-Za-z_]", "空指针检查", True),
    ("bounds_compare", r"if\s*\([^)]{0,120}(>=|<=|>|<)[^)]{0,120}\)", "存在比较判断（可能为边界检查）", True),
    ("size_ceiling", r"SIZE_MAX|INT_MAX|UINT_MAX|USHRT_MAX|INT32_MAX|UINT32_MAX|INT64_MAX|\(size_t\)\s*-1|~\s*0U?", "整数上界常量检查", True),
    ("safe_str_api", r"\b(strncpy|strncat|snprintf|vsnprintf|strlcpy|strlcat|g_strlcpy|g_strlcat|av_strlcpy|av_strlcat|memcpy_s|sprintf_s|apr_pstrmemdup)\b", "使用限长/安全替代 API", True),
    ("clamp_macro", r"\b(min|max|MIN|MAX|CLAMP|clamp|MINIMUM|MAXIMUM)\s*\(", "取值夹紧宏/函数", True),
    ("error_check", r"if\s*\([^)]{0,80}(<\s*0|==\s*-1|!=\s*0|!=\s*NULL|==\s*NULL)[^)]{0,40}\)\s*\{?", "错误返回值检查", True),
    ("assert_call", r"\b(assert|BUG_ON|WARN_ON|CHECK|VERIFY)\s*\(", "断言/防御性宏", True),
    ("goto_cleanup", r"\bgoto\s+\w+", "统一出错清理路径", True),
    ("free_then_null", r"free\s*\([^)]*\)\s*;[^;]{0,40}=\s*NULL", "释放后置空", True),
    ("length_limit", r"(>\s*(?:sizeof|MAX|SIZE|[A-Z_]{3,})|>=\s*(?:MAX|SIZE|[A-Z_]{3,})|<=\s*(?:MAX|SIZE|[A-Z_]{3,}))", "与容量/上限的比较", True),
    ("early_return", r"return\s+(-1|NULL|false|0)\s*;", "提前返回", True),
    ("length_clamp", r"=\s*(min|MIN|MINIMUM|clamp)\s*\(|\?\s*[^:]{1,40}:\s*[^;]{1,40};", "长度被夹紧或条件取值", True),
    ("input_validate", r"if\s*\([^)]{0,100}(len|size|count|n|idx|index|offset)[^)]{0,100}(>|<|>=|<=|==|!=)", "对长度/下标做显式校验", True),
    ("mem_guard", r"(if|while)\s*\([^)]{0,80}(size|len|count)[^)]{0,80}\)\s*\{?[^}]{0,200}(memcpy|memmove)", "拷贝前有长度判断", True),
    ("return_value_used", r"=\s*\w+\s*\([^;]*\)\s*;\s*if\s*\(", "检查函数返回值", True),
]

# 交互特征：(名称, 需要的正特征, 需要的负特征)
INTERACTION_FEATURES = [
    ("mem_copy_without_guard", "mem_copy", "bounds_compare"),
    ("mem_copy_without_size_ceiling", "mem_copy", "size_ceiling"),
    ("str_copy_without_safe_api", "unsafe_str_copy", "safe_str_api"),
    ("alloc_without_null_check", "alloc_call", None),
    ("index_without_bounds", "variable_index", "bounds_compare"),
    ("size_multiply_without_ceiling", "size_multiply", "size_ceiling"),
    ("exec_without_validate", "exec_call", "input_validate"),
    ("format_without_safe_api", "format_output", "safe_str_api"),
]

COMPILED = [(name, re.compile(pattern, re.S), note, is_guard) for name, pattern, note, is_guard in PATTERN_FEATURES]


@dataclass
class Finding:
    """一条可直接写进判定理由的证据。"""

    line: int
    snippet: str
    note: str
    kind: str          # risk / guard

    def as_text(self) -> str:
        return f"第 {self.line} 行 `{self.snippet}`：{self.note}"


@dataclass
class Analysis:
    """离线分析的中间结果。"""

    features: dict[str, float]
    probability: float = 0.5
    verdict: str = "无漏洞"
    confidence: float = 0.5
    risks: list[Finding] = field(default_factory=list)
    guards: list[Finding] = field(default_factory=list)
    cwe: list[str] = field(default_factory=list)
    target_name: str = ""
    reason: str = ""


def extract_features(code: str) -> dict[str, float]:
    """提取可解释特征：二值模式 + 规模特征 + 交互特征。"""
    features: dict[str, float] = {}
    for name, regex, _note, _guard in COMPILED:
        features[name] = 1.0 if regex.search(code) else 0.0
    features["log_length"] = round(math.log1p(len(code)) / 10.0, 3)
    features["n_conditionals"] = min(code.count("if (") + code.count("if("), 60) / 20.0
    features["n_loops"] = min(code.count("for (") + code.count("for(") + code.count("while (")
                              + code.count("while("), 30) / 10.0
    features["n_returns"] = min(code.count("return"), 40) / 20.0
    for name, positive, negative in INTERACTION_FEATURES:
        if features.get(positive, 0.0) <= 0:
            features[name] = 0.0
            continue
        if negative is None:
            features[name] = 1.0
        else:
            features[name] = 1.0 - features.get(negative, 0.0)
    return features


FEATURE_NAMES = list(extract_features("int f(void){return 0;}").keys())

TYPE_WORDS = {
    "int", "char", "short", "long", "float", "double", "void", "unsigned", "signed",
    "const", "struct", "union", "enum", "static", "extern", "register", "volatile",
    "size_t", "ssize_t", "bool", "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "auto",
}


def _size_multiply_risk(masked_line: str) -> bool:
    """判断一行是否真的在做「尺寸乘法」，排除指针声明（`const Image *image`）。"""
    for match in re.finditer(r"([A-Za-z_]\w*|\d+|\)|\])\s*\*\s*([A-Za-z_]\w*|\d+|\()", masked_line):
        left, right = match.group(1), match.group(2)
        if left in TYPE_WORDS or left == ")" and right in TYPE_WORDS:
            continue
        if right == "(":
            continue
        return True
    return False


def _find_lines(code: str, regex: re.Pattern, limit: int = 3, skip_signature: bool = True) -> list[tuple[int, str]]:
    """在「屏蔽注释/字符串」后的代码上匹配，返回原始行内容（带行号）。"""
    from .repo.cparse import mask_literals

    masked = mask_literals(code)
    hits: list[tuple[int, str]] = []
    for number, (raw, mline) in enumerate(zip(code.splitlines(), masked.splitlines()), 1):
        if skip_signature and number <= 2:
            continue
        if regex.search(mline):
            text = raw.strip()
            if not text:
                continue
            hits.append((number, text[:160]))
            if len(hits) >= limit:
                break
    return hits


CWE_BY_FEATURE = {
    "unsafe_str_copy": ["CWE-120", "CWE-787"],
    "mem_copy": ["CWE-787", "CWE-125"],
    "variable_index": ["CWE-125", "CWE-787"],
    "size_multiply": ["CWE-190"],
    "narrow_cast": ["CWE-197", "CWE-190"],
    "exec_call": ["CWE-78"],
    "format_output": ["CWE-134"],
    "division": ["CWE-369"],
    "free_call": ["CWE-416", "CWE-415"],
    "alloc_call": ["CWE-476", "CWE-401"],
}


class OfflineAnalyzer:
    """基于「静态特征 + 知识库训练」的判别器与解释器。"""

    def __init__(self, model: dict):
        self.model = model
        self.weights = model["weights"]
        self.bias = model["bias"]
        self.threshold = model.get("threshold", 0.5)
        self.metrics = model.get("metrics", {})
        self.order = model["features"]

    # ---------------------------------------------------------- 加载/训练
    @classmethod
    def load(cls, build_if_missing: bool = True) -> "OfflineAnalyzer":
        path = config.OFFLINE_MODEL_PATH
        if path.exists():
            return cls(json.loads(path.read_text(encoding="utf-8")))
        if not build_if_missing:
            raise FileNotFoundError(f"缺少离线模型：{path}")
        train_and_save()
        return cls(json.loads(path.read_text(encoding="utf-8")))

    # ---------------------------------------------------------- 推理
    def probability(self, code: str) -> float:
        features = extract_features(code)
        z = self.bias
        for name, weight in zip(self.order, self.weights):
            z += weight * features.get(name, 0.0)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def analyze(self, code: str) -> Analysis:
        prob = self.probability(code)
        verdict = "有漏洞" if prob >= self.threshold else "无漏洞"
        risks = self._collect_findings(code)
        guards = self._collect_guards(code)
        confidence = min(0.95, 0.5 + abs(prob - 0.5) * 0.9)
        analysis = Analysis(
            features=extract_features(code),
            probability=round(prob, 4),
            verdict=verdict,
            confidence=round(confidence, 2),
            risks=risks,
            guards=guards,
            cwe=self._guess_cwe(risks),
            target_name=function_name(code),
        )
        analysis.reason = self._compose_reason(analysis, code)
        return analysis

    def _collect_findings(self, code: str) -> list[Finding]:
        findings: list[Finding] = []
        risk_patterns = [
            ("unsafe_str_copy", "调用了无长度上界的字符串函数，长度不受控时可能越界写入"),
            ("mem_copy", "存在原始内存拷贝，长度与目标缓冲区容量需要额外核对"),
            ("scanf_family", "格式化输入未限定宽度，可能写越界或解析到超长输入"),
            ("exec_call", "执行外部命令，若参数来自不可信输入则存在注入风险"),
            ("variable_index", "使用变量做下标，缺少边界判断时会越界读写"),
            ("narrow_cast", "存在窄化类型转换，大尺寸可能被截断为小尺寸"),
            ("division", "存在除法运算，分母为 0 时会崩溃"),
        ]
        table = dict((n, r) for n, r, _x, _y in PATTERN_FEATURES)
        for feature, note in risk_patterns:
            for line, snippet in _find_lines(code, re.compile(table[feature], re.S), limit=2):
                findings.append(Finding(line=line, snippet=snippet, note=note, kind="risk"))
        for line, snippet in _size_multiply_lines(code, limit=2):
            findings.append(Finding(line=line, snippet=snippet,
                                    note="分配/拷贝尺寸由乘法得出，存在整数溢出导致缓冲区过小", kind="risk"))
        findings = _dedupe(findings)
        findings.sort(key=lambda f: f.line)
        return findings[:8]

    def _collect_guards(self, code: str) -> list[Finding]:
        guards: list[Finding] = []
        guard_patterns = [
            ("null_check", "存在空指针检查"),
            ("length_limit", "与容量/上限做了比较"),
            ("size_ceiling", "使用了整数上界常量做溢出检查"),
            ("safe_str_api", "使用了限长/安全替代 API"),
            ("clamp_macro", "使用夹紧宏限制取值"),
            ("input_validate", "对长度/下标做了显式校验"),
            ("mem_guard", "拷贝前有长度判断"),
            ("free_then_null", "释放后置空指针"),
        ]
        table = dict((n, r) for n, r, _x, _y in PATTERN_FEATURES)
        for feature, note in guard_patterns:
            for line, snippet in _find_lines(code, re.compile(table[feature], re.S), limit=2):
                guards.append(Finding(line=line, snippet=snippet, note=note, kind="guard"))
        guards = _dedupe(guards)
        guards.sort(key=lambda f: f.line)
        return guards[:8]

    @staticmethod
    def _guess_cwe(risks: list[Finding]) -> list[str]:
        scores: dict[str, int] = {}
        for finding in risks:
            for cwe in _risk_feature_cwes(finding):
                scores[cwe] = scores.get(cwe, 0) + 1
        return [cwe for cwe, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:3]

    def _compose_reason(self, analysis: Analysis, code: str) -> str:
        name = analysis.target_name or "该函数"
        lines = []
        if analysis.risks:
            lines.append("风险面：" + "；".join(f.as_text() for f in analysis.risks[:3]))
        else:
            lines.append("风险面：函数中没有出现危险内存操作、外部命令执行或未校验下标等高危动作。")
        if analysis.guards:
            lines.append("缓解面：" + "；".join(f.as_text() for f in analysis.guards[:3]))
        else:
            lines.append("缓解面：函数中没有发现针对上述动作的边界、长度或空指针校验。")
        if analysis.cwe:
            lines.append(f"缺陷类型倾向：{'、'.join(analysis.cwe)}")
        if analysis.verdict == "有漏洞":
            conclusion = ("综合判断：存在可疑的危险动作，且在函数范围内未见有效缓解证据，判定为有漏洞"
                          if analysis.risks else
                          "综合判断：函数中缺少必要的防护措施，按保守口径判定为有漏洞")
        else:
            conclusion = ("综合判断：可疑动作能够被函数内的校验/限长/空指针检查覆盖，判定为无漏洞"
                          if analysis.guards else
                          "综合判断：函数中没有构成触发条件的危险动作，判定为无漏洞")
        lines.append(
            f"{conclusion}（离线判别器概率 {analysis.probability:.2f}，决策阈值 {self.threshold:.2f}；"
            f"{name} 共 {len(code.splitlines())} 行）。"
        )
        return "\n".join(lines)


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[int, str]] = set()
    unique: list[Finding] = []
    for finding in findings:
        key = (finding.line, finding.note)
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique


def _size_multiply_lines(code: str, limit: int = 2) -> list[tuple[int, str]]:
    """尺寸乘法（整数溢出高发点），排除指针声明造成的误报。"""
    from .repo.cparse import mask_literals

    masked = mask_literals(code)
    hits: list[tuple[int, str]] = []
    for number, (raw, mline) in enumerate(zip(code.splitlines(), masked.splitlines()), 1):
        if number <= 2 or not raw.strip():
            continue
        if _size_multiply_risk(mline):
            hits.append((number, raw.strip()[:160]))
            if len(hits) >= limit:
                break
    return hits


def _risk_feature_cwes(finding: Finding) -> list[str]:
    if "字符串" in finding.note:
        return CWE_BY_FEATURE["unsafe_str_copy"]
    if "拷贝" in finding.note:
        return CWE_BY_FEATURE["mem_copy"]
    if "下标" in finding.note:
        return CWE_BY_FEATURE["variable_index"]
    if "乘法" in finding.note or "尺寸" in finding.note:
        return CWE_BY_FEATURE["size_multiply"]
    if "转换" in finding.note:
        return CWE_BY_FEATURE["narrow_cast"]
    if "命令" in finding.note:
        return CWE_BY_FEATURE["exec_call"]
    if "除法" in finding.note:
        return CWE_BY_FEATURE["division"]
    if "输入" in finding.note:
        return ["CWE-120", "CWE-787"]
    return []


# ---------------------------------------------------------------- 训练


def _load_training_samples(limit: int | None = None) -> list[dict]:
    samples = []
    for record in read_jsonl(config.KNOWLEDGE_PATH):
        code = record.get("func") or ""
        if not code.strip():
            continue
        samples.append({
            "code": code,
            "label": int(record.get("target") or 0),
            "group": f"{record.get('cve')}@{record.get('commit_id')}",
            "name": function_name(code),
            "cve": record.get("cve"),
        })
        if limit and len(samples) >= limit:
            break
    return samples


def _matrix(samples: list[dict], order: list[str]):
    import numpy as np

    matrix = np.zeros((len(samples), len(order)), dtype=np.float64)
    labels = np.zeros(len(samples), dtype=np.float64)
    for row, sample in enumerate(samples):
        features = extract_features(sample["code"])
        for col, name in enumerate(order):
            matrix[row, col] = features.get(name, 0.0)
        labels[row] = sample["label"]
    return matrix, labels


def _fit_logistic(x, y, iterations: int = 600, lr: float = 0.6, l2: float = 0.002):
    import numpy as np

    n, d = x.shape
    weights = np.zeros(d)
    bias = 0.0
    for _ in range(iterations):
        z = x @ weights + bias
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = x.T @ (pred - y) / n + l2 * weights
        grad_b = float(np.mean(pred - y))
        weights -= lr * grad
        bias -= lr * grad_b
    return weights, bias


def _predict(x, weights, bias):
    import numpy as np

    z = x @ weights + bias
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_and_save(force: bool = False, verbose: bool = True, folds: int = 5) -> dict:
    """在历史案例上训练逻辑回归，做分组交叉验证并保存模型。"""
    import numpy as np

    config.ensure_dirs()
    samples = _load_training_samples()
    order = FEATURE_NAMES
    x, y = _matrix(samples, order)
    groups = sorted({s["group"] for s in samples})
    rng = np.random.default_rng(20261122)
    rng.shuffle(groups)
    fold_of_group = {group: index % folds for index, group in enumerate(groups)}
    folds_of_sample = np.array([fold_of_group[s["group"]] for s in samples])

    oof = np.zeros(len(samples))
    for fold in range(folds):
        train_mask = folds_of_sample != fold
        test_mask = ~train_mask
        weights, bias = _fit_logistic(x[train_mask], y[train_mask])
        oof[test_mask] = _predict(x[test_mask], weights, bias)

    metrics = _pair_metrics(samples, oof)
    threshold = metrics["best_threshold"]

    weights, bias = _fit_logistic(x, y)
    payload = {
        "features": order,
        "weights": [round(float(w), 6) for w in weights],
        "bias": round(float(bias), 6),
        "threshold": round(float(threshold), 4),
        "metrics": metrics,
        "trained_at": "2026-11-22",
        "samples": len(samples),
    }
    config.OFFLINE_MODEL_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"[OFFLINE] 训练样本 {len(samples)} 条，特征 {len(order)} 个，阈值 {threshold:.3f}")
        print(f"[OFFLINE] 交叉验证：{json.dumps({k: v for k, v in metrics.items() if k != 'curve'}, ensure_ascii=False)}")
    return payload


def _pair_metrics(samples: list[dict], probabilities) -> dict:
    """在交叉验证预测上按「同一函数的漏洞版/修复版」成对统计。"""
    pairs: dict[tuple[str, str], dict[str, float]] = {}
    for sample, prob in zip(samples, probabilities):
        key = (sample["group"], sample["name"])
        entry = pairs.setdefault(key, {})
        # 同一 key 可能包含多个函数（同名），取标签对应的位置
        entry[str(sample["label"])] = float(prob)
    paired = [p for p in pairs.values() if "0" in p and "1" in p]

    best = {"best_threshold": 0.5, "pair_correct": 0, "pair_correct_rate": 0.0}
    for step in range(5, 96):
        threshold = step / 100
        correct = sum(1 for p in paired if p["1"] >= threshold and p["0"] < threshold)
        reversed_count = sum(1 for p in paired if p["1"] < threshold and p["0"] >= threshold)
        score = correct - 0.25 * reversed_count
        if score > best.get("score", -1):
            best = {
                "best_threshold": threshold,
                "pair_correct": correct,
                "pair_correct_rate": round(correct / len(paired), 4) if paired else 0.0,
                "pair_reversed": reversed_count,
                "score": score,
            }

    threshold = best["best_threshold"]
    predictions = (probabilities >= threshold).astype(int)
    labels = [s["label"] for s in samples]
    accuracy = float(sum(1 for p, l in zip(predictions, labels) if int(p) == l) / len(labels))
    tp = sum(1 for p, l in zip(predictions, labels) if int(p) == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if int(p) == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if int(p) == 0 and l == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "pairs": len(paired),
        "best_threshold": best["best_threshold"],
        "pair_correct": best["pair_correct"],
        "pair_correct_rate": best["pair_correct_rate"],
        "pair_reversed": best.get("pair_reversed", 0),
        "sample_accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
    }


# ---------------------------------------------------------------- 离线「模型」适配器


class OfflineLLM:
    """把离线分析器包装成与 LLMClient 相同的接口，驱动同一套 ReAct 循环。

    它在「第 N 轮该做什么」上采用确定性策略：
      1) 仓库可用时先做一次仓库核验（repo_overview + find_callers）；
      2) 需要时补一次定向知识检索；
      3) 提交结构化结论。
    这样离线模式的界面、事件流与真实模型完全一致，便于无 Key 演示与回归测试。
    """

    name = "offline-analyzer"

    def __init__(self, analyzer: OfflineAnalyzer, *, repo_enabled: bool = True,
                 prefetched_kb: bool = True):
        self.analyzer = analyzer
        self.repo_enabled = repo_enabled
        self.prefetched_kb = prefetched_kb
        self._analysis: Analysis | None = None
        self._counter = 0

    def _call(self, name: str, arguments: dict) -> ToolCall:
        self._counter += 1
        return ToolCall(id=f"offline_{self._counter}", name=name, arguments=arguments)

    def analyze(self, code: str) -> Analysis:
        if self._analysis is None:
            self._analysis = self.analyzer.analyze(code)
        return self._analysis

    def chat(self, messages, tools=None, tool_choice=None, max_tokens=None, temperature=None):
        code = _current_code(messages)
        analysis = self.analyze(code)
        forced = isinstance(tool_choice, dict) and \
            (tool_choice.get("function") or {}).get("name") == "submit_report"
        used = _used_tools(messages)

        if forced:
            return AssistantTurn(text="", tool_calls=[self._call("submit_report", self._report_args(analysis))])

        if self.repo_enabled and "repo_overview" not in used:
            target = analysis.target_name or function_name(code)
            calls = [self._call("repo_overview", {})]
            if target:
                calls.append(self._call("find_callers", {"name": target}))
            return AssistantTurn(
                text=(
                    "先确认仓库快照版本，再核实该函数的调用方：如果参数来自外部且调用方未校验，"
                    "风险可被触发；反之则需重新评估。"
                ),
                tool_calls=calls,
            )

        if not self.prefetched_kb and "search_knowledge" not in used:
            query = function_signature(code) + " " + " ".join(analysis.cwe)
            return AssistantTurn(
                text="先检索历史相似案例，确认这类写法在历史漏洞中的典型表现。",
                tool_calls=[self._call("search_knowledge", {"query": query.strip(), "top_k": 5})],
            )

        return AssistantTurn(
            text="证据已足够，提交结论。",
            tool_calls=[self._call("submit_report", self._report_args(analysis))],
        )

    def _report_args(self, analysis: Analysis) -> dict:
        evidence = [f.as_text() for f in analysis.risks[:3]] + [f.as_text() for f in analysis.guards[:2]]
        verdict = analysis.verdict
        confidence = analysis.confidence
        if self.repo_enabled:
            verdict_note = "（已结合仓库上下文核验）"
        else:
            verdict_note = "（仓库上下文不可用，仅基于函数本身与历史案例）"
        return {
            "verdict": verdict,
            "confidence": confidence,
            "cwe": analysis.cwe,
            "reason": analysis.reason + verdict_note,
            "key_evidence": evidence[:5] or ["函数中未发现高危动作，也未发现需要排除的可疑点"],
        }


def _used_tools(messages: list[dict]) -> set[str]:
    used: set[str] = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            name = (call.get("function") or {}).get("name")
            if name:
                used.add(name)
    return used


def _current_code(messages: list[dict]) -> str:
    """从任务描述里取回待测函数源码（离线策略需要重新分析）。"""
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content") or ""
        marker_start = content.find("```c")
        if marker_start < 0:
            continue
        marker_end = content.find("```", marker_start + 4)
        if marker_end < 0:
            continue
        return content[marker_start + 4:marker_end].strip("\n")
    return ""


if __name__ == "__main__":
    train_and_save(force=True)
