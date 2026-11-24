"""函数漏洞检测智能体 —— 图形界面（Streamlit）。

四个页面：
1. 单样本检测：输入函数与提交链接，实时查看「知识检索 → 自主取证 → 判定」全过程；
2. 批量检测：上传 / 指定 .jsonl，批量跑并下载 result.jsonl；
3. 评测：按课程口径统计 P-C / P-V / P-B / P-R 与等级；
4. 知识库：浏览与检索历史 CVE 案例。

启动：
    python -m fva.cli ui          # 或 streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fva import config                                   # noqa: E402
from fva.batch import BatchRunner, iter_samples          # noqa: E402
from fva.evaluation import evaluate, render_markdown      # noqa: E402
from fva.knowledge import KnowledgeBase                  # noqa: E402
from fva.pipeline import Detector, detect_one             # noqa: E402
from fva.textutil import read_jsonl                       # noqa: E402

st.set_page_config(page_title="函数漏洞检测智能体", page_icon="🛡️", layout="wide")

EVENT_ICON = {
    "stage": "🧭", "thought": "💭", "tool_call": "🔧", "tool_result": "📄",
    "kb_hit": "📚", "snapshot": "🗂️", "verdict": "✅", "error": "⚠️", "final": "🏁",
}

# 事件里的机器阶段名 → 界面上的可读标题
STAGE_LABEL = {
    "input": "① 输入函数",
    "agent_start": "② 智能体启动",
    "nudge": "③ 补充取证",
    "budget": "④ 收敛结论",
}


# ---------------------------------------------------------------- 资源


@st.cache_resource(show_spinner=False)
def load_kb() -> KnowledgeBase:
    return KnowledgeBase.load()


@st.cache_resource(show_spinner=False)
def load_detector(mode: str, repo_enabled: bool) -> Detector:
    return Detector(kb=load_kb(), mode=mode, repo_enabled=repo_enabled)


def load_test_samples() -> list[dict]:
    if not config.TEST_PATH.exists():
        return []
    return list(read_jsonl(config.TEST_PATH))


def settings_sidebar() -> dict:
    st.sidebar.title("🛡️ 函数漏洞检测智能体")
    st.sidebar.caption("LLM + 领域知识库(RAG) + 仓库上下文挖掘")

    settings = config.Settings.load()
    llm = settings.llm
    if llm.configured:
        st.sidebar.success("模型模式：真实大模型")
        st.sidebar.caption(llm.describe())
    else:
        st.sidebar.warning("模型模式：离线分析（降级）")
        st.sidebar.caption("未配置密钥时用离线判别器跑通流程，判定质量有限。")
        with st.sidebar.expander("如何配置大模型？"):
            st.markdown(
                "**方式一（推荐：双击启动也生效）**\n\n"
                "在项目根目录创建 `.env` 文件，写入：\n\n"
                "```ini\nFVA_LLM_API_KEY=sk-你的密钥\nFVA_LLM_MODEL=deepseek-chat\n```\n\n"
                "或执行 `python -m fva.cli setup` 交互式写入并自动自检。\n\n"
                "**方式二**：设置系统环境变量（`setx` 后需重开终端）。\n\n"
                f"配置文件：`{config.ENV_FILE}`\n\n"
                "> 注意：在某个 cmd 窗口里 `set` 的变量只对该窗口有效，"
                "双击 `.bat` 启动的新进程读不到它。"
            )
        if st.sidebar.button("重新加载配置", use_container_width=True):
            load_kb.clear()
            load_detector.clear()
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("检测参数")
    top_k = st.sidebar.slider("知识检索条数", 1, 8, settings.agent.max_kb_hits)
    max_steps = st.sidebar.slider("Agent 最大轮数", 3, 20, settings.agent.max_steps)
    use_repo = st.sidebar.checkbox("启用仓库上下文挖掘", value=settings.repo.enabled,
                                   help="关闭后只依据函数本身与历史案例判断（更快、更省）")
    settings.agent.max_kb_hits = top_k
    settings.agent.max_steps = max_steps

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"知识库：{'已就绪' if config.KB_CORPUS_PATH.exists() else '未构建'}\n\n"
        f"测试集：{config.TEST_PATH.name}\n\n"
        f"结果文件：results/result.jsonl"
    )
    return {"settings": settings, "use_repo": use_repo}


# ---------------------------------------------------------------- 组件


def render_event(event: dict) -> str | None:
    kind = event.get("type")
    icon = EVENT_ICON.get(kind, "•")
    if kind == "stage":
        label = STAGE_LABEL.get(str(event.get("stage")), event.get("stage"))
        return f"{icon} **{label}**：{event.get('detail', '')}"
    if kind == "thought":
        return f"{icon} **思考**：{event.get('text', '')}"
    if kind == "tool_call":
        text = json.dumps(event.get("args", {}), ensure_ascii=False).replace("\\n", " ")
        return f"{icon} **调用工具** `{event.get('tool')}`　参数 `{text[:300]}`"
    if kind == "tool_result":
        flag = "✅" if event.get("ok") else "⚠️"
        return f"{flag} **结果** `{event.get('tool')}`：{event.get('preview', '')}"
    if kind == "kb_hit":
        rows = event.get("data") or []
        text = "、".join(f"{r.get('cve')}（{'/'.join(r.get('cwe') or [])}，{r.get('score')}）" for r in rows[:4])
        return f"{icon} **知识检索**：命中 {event.get('count')} 条 —— {text}"
    if kind == "snapshot":
        return f"{icon} **仓库快照**（{event.get('status')}）：{event.get('detail', '')}"
    if kind == "verdict":
        return (f"{icon} **判定**：{event.get('verdict')}（置信度 {event.get('confidence')}，"
                f"{event.get('steps')} 轮 / {event.get('tool_calls')} 次工具调用）")
    if kind == "error":
        return f"{icon} **错误**：{event.get('detail')}"
    return None


def show_report(result) -> None:
    st.markdown("### 检测报告")
    col1, col2, col3 = st.columns([1, 1, 2])
    col1.metric("函数 idx", str(result.idx) if result.idx is not None else "—")
    col2.metric("检测结果", result.verdict, f"置信度 {result.confidence}")
    col3.metric("模式 / 耗时", f"{result.mode}", f"{result.elapsed}s")

    st.markdown("#### 判定理由")
    # 理由按行渲染，保留「风险面 / 缓解面 / 综合判断」的分段结构
    for line in (result.reason or "").splitlines():
        if line.strip():
            st.markdown(line.strip())

    if result.key_evidence:
        st.markdown("#### 关键证据")
        for item in result.key_evidence:
            st.markdown(f"- {item}")

    with st.expander("分析记录（知识检索 + 仓库读取 + 工具轨迹）", expanded=True):
        st.code(result.analysis_record(), language="text")

    with st.expander("完整事件流（可回溯每一步）"):
        for event in result.events:
            line = render_event(event)
            if line:
                st.markdown(line)


# ---------------------------------------------------------------- 页面


def page_single(state: dict) -> None:
    st.header("单样本检测")
    st.caption("输入一个 C/C++ 函数与它所在仓库的修复提交链接，智能体会自己决定查什么、查到什么程度。")

    samples = load_test_samples()
    with st.expander("📥 从测试集载入一条样本（便于演示）", expanded=not samples == []):
        if samples:
            idx_options = [s["idx"] for s in samples[:40]]
            picked = st.selectbox("选择 idx", idx_options, index=0)
            if st.button("载入到下方输入框", key="load_sample"):
                sample = next(s for s in samples if s["idx"] == picked)
                st.session_state["func_code"] = sample["func"]
                st.session_state["commit_url"] = sample["commit_url"]
                st.session_state["sample_idx"] = sample["idx"]
                st.session_state["sample_target"] = sample.get("target")
                st.rerun()

    left, right = st.columns([3, 2])
    with left:
        code = st.text_area("函数源码", value=st.session_state.get("func_code", ""), height=360,
                            placeholder="粘贴一个 C/C++ 函数……")
    with right:
        commit_url = st.text_input("修复提交链接", value=st.session_state.get("commit_url", ""),
                                   placeholder="https://github.com/owner/repo/commit/<sha>")
        version_hint = st.radio(
            "仓库版本（严格按标签选择，不泄漏另一版本）",
            options=["不读取仓库", "漏洞版：父提交", "修复版：提交后"],
            index=0 if not state["use_repo"] else 1,
            help="漏洞版样本只能取修复提交的父提交；修复版样本只能取提交后版本。",
        )
        target = {"不读取仓库": None, "漏洞版：父提交": 1, "修复版：提交后": 0}[version_hint]
        idx_text = st.text_input("函数 idx（可选）", value=str(st.session_state.get("sample_idx") or ""),
                                 help="用于报告中标识样本；手工输入时可留空")

    if st.button("🚀 开始检测", type="primary", use_container_width=True):
        if not code.strip():
            st.error("请先粘贴函数源码")
            return
        repo_enabled = target is not None and state["use_repo"]
        detector = Detector(kb=load_kb(), mode=state["settings"].llm.configured and "llm" or "offline",
                            settings=state["settings"], repo_enabled=repo_enabled)
        st.markdown("### 分析过程")
        timeline = st.empty()   # 占位符：每次覆盖刷新，而不是追加新元素
        lines: list[str] = []

        def on_event(event: dict) -> None:
            line = render_event(event)
            if line:
                lines.append(line)
                timeline.markdown("\n\n".join(lines[-40:]))

        with st.spinner("智能体正在分析……"):
            result = detect_one(
                code, commit_url, idx=int(idx_text) if idx_text.strip().isdigit() else None, target=target,
                mode="auto" if state["settings"].llm.configured else "offline",
                repo_enabled=repo_enabled, kb=load_kb(), on_event=on_event,
                settings=state["settings"],
            )
        st.divider()
        show_report(result)


def page_batch(state: dict) -> None:
    st.header("批量检测")
    st.caption("上传 JSONL（每行至少包含 func / commit_url，可选 idx / target），批量检测并生成 result.jsonl。")

    col1, col2, col3 = st.columns(3)
    with col1:
        upload = st.file_uploader("上传 .jsonl", type=["jsonl", "json"])
    with col2:
        use_test = st.checkbox("使用内置测试集 dataset/test.jsonl", value=upload is None)
    with col3:
        limit_pairs = st.number_input("只跑前 N 对（0 表示全部）", min_value=0, max_value=150, value=0)

    workers = st.slider("并发数", 1, 8, 4 if not state["settings"].llm.configured else 2)
    mode = "llm" if state["settings"].llm.configured else "offline"
    st.info(f"当前模式：{'真实大模型' if mode == 'llm' else '离线分析'}；"
            f"仓库上下文：{'启用' if state['use_repo'] else '关闭'}；并发 {workers}")

    if st.button("开始批量检测", type="primary"):
        if upload is not None:
            records = []
            for line in upload.getvalue().decode("utf-8-sig").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        elif use_test:
            records = list(iter_samples(config.TEST_PATH, pairs=int(limit_pairs) or None))
        else:
            st.error("请上传文件或勾选使用内置测试集")
            return

        result_path = config.RESULTS_DIR / f"ui_result_{time.strftime('%H%M%S')}.jsonl"
        detector = Detector(kb=load_kb(), mode=mode, settings=state["settings"],
                            repo_enabled=state["use_repo"])
        progress = st.progress(0.0, text="准备中……")
        status = st.empty()

        def on_progress(info: dict) -> None:
            if info["event"] == "progress":
                progress.progress(info["finished"] / max(1, info["total"]),
                                  text=f"{info['finished']}/{info['total']}  idx={info['idx']} → {info['verdict']}")
                status.caption(f"最近一条：idx={info['idx']} {info['verdict']}（{info['confidence']}）"
                               f" 用时 {info['elapsed']}s，速度 {info['rate']}/s")
            elif info["event"] == "done":
                progress.progress(1.0, text="完成")

        runner = BatchRunner(detector, result_path=result_path, workers=workers,
                             resume=False, save_trace=False, on_progress=on_progress)
        summary = runner.run(records)
        st.success(f"完成：{summary['detected']} 条，用时 {summary['wall_time']}s，"
                   f"平均 {summary['avg_elapsed']}s/条")

        predictions = {r["idx"]: r["prediction"] for r in read_jsonl(result_path)}
        rows = [{"idx": s.get("idx"), "target": s.get("target"),
                 "prediction": predictions.get(s.get("idx"))} for s in records]
        st.dataframe(rows, use_container_width=True, height=320)
        st.download_button("⬇️ 下载 result.jsonl", result_path.read_text(encoding="utf-8"),
                           file_name="result.jsonl", mime="application/jsonl")

        try:
            report = evaluate(result_path, config.TEST_PATH)
            st.markdown(render_markdown(report))
        except Exception as exc:  # 自上传数据可能缺字段
            st.caption(f"（评测未执行：{exc}）")


def page_eval(state: dict) -> None:
    st.header("评测")
    st.caption("按课程口径统计：P-C 成对正确 / P-V 均判有洞 / P-B 均判无洞 / P-R 完全反转，并给出等级。")
    candidates = sorted(config.RESULTS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        st.warning("results/ 下还没有结果文件，请先跑批量检测。")
        return
    picked = st.selectbox("结果文件", [str(p.name) for p in candidates])
    path = config.RESULTS_DIR / picked
    try:
        report = evaluate(path, config.TEST_PATH)
    except Exception as exc:
        st.error(f"评测失败：{exc}")
        return
    st.markdown(render_markdown(report))
    cols = st.columns(4)
    for column, key in zip(cols, ["P-C", "P-V", "P-B", "P-R"]):
        column.metric(key, report["counts"][key], f"{report['rates'][key]:.1%}")


def page_kb(state: dict) -> None:
    st.header("领域知识库")
    kb = load_kb()
    st.metric("历史案例数", len(kb.cases), f"其中成对案例 {sum(1 for c in kb.cases if c.paired)}")
    query = st.text_input("检索相似案例（可粘贴一段代码或输入函数名/API）",
                          value="memcpy NULL check size overflow")
    if query:
        hits = kb.search(query, top_k=8)
        for hit in hits:
            case = hit.case
            with st.expander(f"#{hit.rank} {case.cve} | {'/'.join(case.cwe)} | {case.project}::{case.func_name} | 相似度 {hit.score}"):
                st.caption(case.cve_desc)
                if case.fix_hints:
                    st.markdown("**修复要点**：" + "；".join(case.fix_hints))
                st.code(case.code_vuln[:1800], language="c")


def main() -> None:
    state = settings_sidebar()
    page = st.sidebar.radio("功能", ["单样本检测", "批量检测", "评测", "领域知识库"])
    st.sidebar.markdown("---")
    st.sidebar.caption("课程项目 · 高级软件工程 · 2026 秋季")
    if page == "单样本检测":
        page_single(state)
    elif page == "批量检测":
        page_batch(state)
    elif page == "评测":
        page_eval(state)
    else:
        page_kb(state)


if __name__ == "__main__":
    main()
