"""核心模块单元测试（标准库 unittest，无需第三方依赖）。

运行：
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fva import config
from fva.agent import ReActAgent, ToolContext, build_registry, build_user_prompt, normalize_verdict
from fva.evaluation import evaluate
from fva.knowledge import KnowledgeBase, build_kb
from fva.llm import AssistantTurn, ScriptedLLM, ToolCall
from fva.repo import cparse, parse_commit_url
from fva.repo.snapshot import SnapshotError
from fva.repo.tools import RepoTools
from fva.textutil import function_name, function_signature, tokenize

SAMPLE_FUNC = """static int copy_name(char *dst, const char *src, int len)
{
    if (len < 0)
        return -1;
    memcpy(dst, src, len);
    return 0;
}"""


class StubKB:
    """最小知识库替身：避免单元测试依赖完整索引。"""

    def search(self, query, top_k=5, paired_only=True):
        return []


def _context(**overrides) -> ToolContext:
    settings = config.Settings()
    settings.repo.enabled = False
    ctx = ToolContext(
        func_code=SAMPLE_FUNC,
        commit_url="https://github.com/owner/repo/commit/" + "a" * 40,
        target=1,
        kb=StubKB(),
        settings=settings,
    )
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


class TestTextUtil(unittest.TestCase):
    def test_tokenize_splits_identifiers(self):
        tokens = tokenize("int ParseExifHeader(const char *buf, size_t buf_len);")
        self.assertIn("parse", tokens)
        self.assertIn("exif", tokens)
        self.assertIn("header", tokens)
        self.assertNotIn("int", tokens)      # C 关键字被过滤
        self.assertNotIn("const", tokens)

    def test_function_signature_and_name(self):
        self.assertEqual(function_name(SAMPLE_FUNC), "copy_name")
        self.assertIn("copy_name(char *dst", function_signature(SAMPLE_FUNC))

    def test_tokenize_ignores_comments(self):
        tokens = tokenize("// strcpy is dangerous\nint a = 1;")
        self.assertNotIn("strcpy", tokens)


class TestCParser(unittest.TestCase):
    def test_extract_functions_with_nested_block(self):
        code = """
        static int foo(int a) {
            if (a) { bar(a); }
            return 0;
        }
        int bar(int x)
        {
            return baz(x) + 1;
        }
        """
        funcs = cparse.extract_functions(code)
        self.assertEqual([f.name for f in funcs], ["foo", "bar"])
        self.assertEqual((funcs[0].start_line, funcs[0].end_line), (2, 5))

    def test_extract_calls(self):
        calls = cparse.extract_calls(SAMPLE_FUNC)
        self.assertIn("memcpy", calls)
        self.assertNotIn("if", calls)

    def test_cpp_method(self):
        funcs = cparse.extract_functions("void S::run() const { helper(); }\n")
        self.assertEqual(funcs[0].name, "run")


class TestRepoTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "src").mkdir()
        (root / "src" / "util.c").write_text(
            "#include <string.h>\n"
            "static int helper(int n) { return n + 1; }\n"
            "int copy_buf(char *dst, const char *src, int len)\n"
            "{\n"
            "    memcpy(dst, src, len);\n"
            "    return helper(len);\n"
            "}\n",
            encoding="utf-8",
        )
        (root / "src" / "main.c").write_text(
            "int copy_buf(char *dst, const char *src, int len);\n"
            "int main(void) { char b[8]; return copy_buf(b, \"x\", 1); }\n",
            encoding="utf-8",
        )
        self.tools = RepoTools(root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_grep_and_read(self):
        self.assertIn("memcpy(dst, src, len)", self.tools.grep("memcpy"))
        self.assertIn("1|", self.tools.read_file("src/util.c", 1, 2))

    def test_find_function_and_callees(self):
        self.assertIn("src/util.c", self.tools.find_function("copy_buf"))
        self.assertIn("memcpy", self.tools.find_callees("copy_buf"))
        self.assertIn("char *dst", self.tools.get_function("copy_buf"))

    def test_find_callers_excludes_definition(self):
        text = self.tools.find_callers("copy_buf")
        self.assertIn("main.c", text)
        self.assertIn("main", text)
        # 定义所在行不应被当作调用方
        self.assertNotIn("util.c:3", text)

    def test_path_escape_blocked(self):
        self.assertIn("非法路径", self.tools.read_file("../../etc/passwd"))


class TestCommitUrl(unittest.TestCase):
    def test_github(self):
        ref = parse_commit_url("https://github.com/ImageMagick/ImageMagick6/commit/dc070da861a015d3c97488fdcca6063b44d47a7b")
        self.assertEqual(ref.owner, "ImageMagick")
        self.assertEqual(ref.repo, "ImageMagick6")
        self.assertEqual(ref.provider, "github")

    def test_gitweb(self):
        ref = parse_commit_url("https://git.qemu.org/?p=qemu.git;a=commit;h=e6908bfe8e07f2b452e78e677da1b45b1c0f6829")
        self.assertEqual(ref.provider, "gitweb")
        self.assertEqual(ref.sha, "e6908bfe8e07f2b452e78e677da1b45b1c0f6829")

    def test_invalid(self):
        with self.assertRaises(SnapshotError):
            parse_commit_url("https://github.com/owner/repo")


class TestVerdictParsing(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(normalize_verdict("有漏洞"), "有漏洞")
        self.assertEqual(normalize_verdict("无漏洞"), "无漏洞")
        self.assertEqual(normalize_verdict(1), "有漏洞")
        self.assertEqual(normalize_verdict(0), "无漏洞")
        self.assertEqual(normalize_verdict("vulnerable"), "有漏洞")
        self.assertEqual(normalize_verdict("本函数未发现漏洞"), "无漏洞")


class TestKnowledgeBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_kb(force=False, verbose=False)
        cls.kb = KnowledgeBase.load()

    def test_cases_built(self):
        self.assertGreater(len(self.kb.cases), 3000)
        self.assertGreater(sum(1 for c in self.kb.cases if c.paired), 3000)

    def test_search_returns_relevant(self):
        hits = self.kb.search_function(SAMPLE_FUNC, top_k=3)
        self.assertEqual(len(hits), 3)
        # 相似度应当单调不增
        scores = [h.score for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_render_contains_case(self):
        hits = self.kb.search_function(SAMPLE_FUNC, top_k=1)
        text = self.kb.render_hits(hits)
        self.assertIn("CVE", text)


class TestAgentLoop(unittest.TestCase):
    def test_submit_report_terminates_loop(self):
        turns = [
            AssistantTurn(text="先检索知识库", tool_calls=[
                ToolCall(id="c1", name="search_knowledge", arguments={"query": "memcpy length"})]),
            AssistantTurn(text="结论已明确", tool_calls=[
                ToolCall(id="c2", name="submit_report", arguments={
                    "verdict": "有漏洞", "confidence": 0.8, "cwe": ["CWE-787"],
                    "reason": "memcpy 的长度来自参数 len，函数未校验目标缓冲区容量。",
                    "key_evidence": ["第 5 行 memcpy(dst, src, len)"]})]),
        ]
        llm = ScriptedLLM(turns)
        events = []
        ctx = _context()
        agent = ReActAgent(llm, ctx, settings=config.AgentSettings(max_steps=6), on_event=events.append)
        report = agent.run(build_user_prompt(SAMPLE_FUNC, ctx.commit_url, "（无）", repo_available=False))
        self.assertEqual(report.verdict, "有漏洞")
        self.assertEqual(report.prediction, 1)
        self.assertEqual(report.finished_by, "submit")
        self.assertEqual(report.tool_calls, 2)
        self.assertTrue(any(e["type"] == "tool_call" for e in events))
        self.assertTrue(any(e["type"] == "verdict" for e in events))

    def test_forced_report_when_model_never_submits(self):
        """模型一直不调用工具时，应被强制收尾而不是丢弃样本。"""
        turns = [
            AssistantTurn(text="我认为这段代码有问题。"),
            AssistantTurn(text="", tool_calls=[
                ToolCall(id="f1", name="submit_report", arguments={
                    "verdict": "无漏洞", "reason": "已核对，长度在上游被限制。"})]),
        ]
        llm = ScriptedLLM(turns)
        ctx = _context()
        agent = ReActAgent(llm, ctx, settings=config.AgentSettings(max_steps=3, min_repo_checks=0))
        report = agent.run("请判断")
        self.assertEqual(report.verdict, "无漏洞")
        self.assertEqual(report.finished_by, "forced")

    def test_duplicate_tool_calls_are_skipped(self):
        call = ToolCall(id="dup", name="search_knowledge", arguments={"query": "same"})
        turns = [AssistantTurn(tool_calls=[call]), AssistantTurn(tool_calls=[call]),
                 AssistantTurn(tool_calls=[call])]
        llm = ScriptedLLM(turns)
        ctx = _context()
        agent = ReActAgent(llm, ctx, settings=config.AgentSettings(max_steps=3, min_repo_checks=0))
        report = agent.run("请判断")
        self.assertEqual(report.finished_by, "forced")

    def test_llm_error_is_reported(self):
        from fva.llm import LLMError

        class BrokenLLM:
            name = "broken"

            def chat(self, *args, **kwargs):
                raise LLMError("接口不可达")

        ctx = _context()
        agent = ReActAgent(BrokenLLM(), ctx, settings=config.AgentSettings(max_steps=2))
        report = agent.run("请判断")
        self.assertEqual(report.finished_by, "error")
        self.assertIn("接口不可达", report.error)


class TestEvaluation(unittest.TestCase):
    def test_counts_and_grade(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = tmp_path / "test.jsonl"
            result = tmp_path / "result.jsonl"
            records = [
                {"idx": 1, "target": 1}, {"idx": 2, "target": 0},     # P-C
                {"idx": 3, "target": 1}, {"idx": 4, "target": 0},     # P-V
                {"idx": 5, "target": 0}, {"idx": 6, "target": 1},     # P-C（顺序无关）
                {"idx": 7, "target": 1}, {"idx": 8, "target": 0},     # P-R
            ]
            dataset.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            predictions = [
                {"idx": 1, "prediction": 1}, {"idx": 2, "prediction": 0},
                {"idx": 3, "prediction": 1}, {"idx": 4, "prediction": 1},
                {"idx": 5, "prediction": 0}, {"idx": 6, "prediction": 1},
                {"idx": 7, "prediction": 0}, {"idx": 8, "prediction": 1},
            ]
            result.write_text("\n".join(json.dumps(r) for r in predictions), encoding="utf-8")
            report = evaluate(result, dataset)
            self.assertEqual(report["counts"], {"P-C": 2, "P-V": 1, "P-B": 0, "P-R": 1})
            self.assertEqual(report["level"], "D")     # 2 对 < 20 对

    def test_duplicate_prediction_conflict_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = tmp_path / "test.jsonl"
            result = tmp_path / "result.jsonl"
            dataset.write_text(
                "\n".join(json.dumps(r) for r in [{"idx": 1, "target": 1}, {"idx": 2, "target": 0}]),
                encoding="utf-8",
            )
            result.write_text(
                "\n".join(json.dumps(r) for r in [{"idx": 1, "prediction": 1}, {"idx": 1, "prediction": 0}]),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                evaluate(result, dataset)


class TestEnvConfig(unittest.TestCase):
    """`.env` 配置：解析、优先级、合并写入。"""

    def test_parse_env_text(self):
        text = (
            "# 注释行\n"
            "\n"
            "FVA_LLM_API_KEY=sk-abc123\n"
            "export FVA_LLM_MODEL=deepseek-chat\n"
            'QUOTED="value with space"\n'
            "EMPTY=\n"
            "NOT_A_PAIR\n"
        )
        values = config.parse_env_text(text)
        self.assertEqual(values["FVA_LLM_API_KEY"], "sk-abc123")
        self.assertEqual(values["FVA_LLM_MODEL"], "deepseek-chat")
        self.assertEqual(values["QUOTED"], "value with space")
        self.assertNotIn("NOT_A_PAIR", values)

    def test_env_var_overrides_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("FVA_LLM_API_KEY=from-dotenv\nFVA_LLM_MODEL=dotenv-model\n",
                                encoding="utf-8")
            with mock.patch.object(config, "ENV_FILE", env_path):
                config.load_env_file(env_path, reload=True)
                self.assertEqual(config._env("FVA_LLM_API_KEY"), "from-dotenv")
                self.assertIn(".env", config.env_source("FVA_LLM_API_KEY"))
                with mock.patch.dict(os.environ, {"FVA_LLM_API_KEY": "from-env"}):
                    # 环境变量优先于 .env
                    self.assertEqual(config._env("FVA_LLM_API_KEY"), "from-env")
                    self.assertIn("环境变量", config.env_source("FVA_LLM_API_KEY"))

    def test_write_env_file_merges(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("# 手写注释\nFVA_LLM_MODEL=old-model\nOTHER_KEY=keep-me\n",
                                encoding="utf-8")
            config.write_env_file({"FVA_LLM_MODEL": "new-model", "FVA_LLM_API_KEY": "sk-new"},
                                  env_path)
            values = config.load_env_file(env_path, reload=True)
            self.assertEqual(values["FVA_LLM_MODEL"], "new-model")
            self.assertEqual(values["FVA_LLM_API_KEY"], "sk-new")
            self.assertEqual(values["OTHER_KEY"], "keep-me")     # 其它键保留
            self.assertIn("# 手写注释", env_path.read_text(encoding="utf-8"))


class TestPipelineOffline(unittest.TestCase):
    def test_detect_returns_structured_report(self):
        from fva.pipeline import detect_one

        result = detect_one(SAMPLE_FUNC, "", mode="offline", repo_enabled=False)
        self.assertIn(result.verdict, ("有漏洞", "无漏洞"))
        self.assertIn(result.prediction, (0, 1))
        self.assertTrue(result.reason)
        self.assertEqual(result.mode, "offline")
        self.assertIn("判定", result.reason)
        self.assertIn("历史案例检索", result.analysis_record())


if __name__ == "__main__":
    unittest.main(verbosity=2)
