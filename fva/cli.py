"""命令行入口：配置、构建知识库、单样本检测、批量检测、评测与界面启动。

用法示例：
    python -m fva.cli setup                # 交互式配置大模型（写入 .env）并自检
    python -m fva.cli check                # 检查模型连通性
    python -m fva.cli build-kb --force
    python -m fva.cli detect --idx 194963
    python -m fva.cli batch --mode offline --no-repo --workers 4
    python -m fva.cli eval
    python -m fva.cli ui
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import config
from .textutil import read_jsonl


def cmd_build_kb(args) -> int:
    from .knowledge import build_kb

    stats = build_kb(force=args.force)
    print(json.dumps({k: v for k, v in stats.items() if k != "contamination"}, ensure_ascii=False, indent=2))
    if stats.get("contamination"):
        print("污染校验：", json.dumps(stats["contamination"], ensure_ascii=False))
    return 0


def cmd_train(args) -> int:
    from .offline import train_and_save

    payload = train_and_save(force=True, verbose=True)
    print(json.dumps({k: v for k, v in payload["metrics"].items() if k != "curve"},
                     ensure_ascii=False, indent=2))
    return 0


def _load_sample_by_idx(idx: int) -> dict:
    for record in read_jsonl(config.TEST_PATH):
        if record.get("idx") == idx:
            return record
    raise SystemExit(f"测试集中不存在 idx={idx}")


def cmd_detect(args) -> int:
    from .pipeline import detect_one

    if args.idx is not None:
        sample = _load_sample_by_idx(args.idx)
        code, commit_url, target = sample["func"], sample["commit_url"], sample.get("target")
        idx = sample.get("idx")
    else:
        code = Path(args.func_file).read_text(encoding="utf-8") if args.func_file else (args.code or "")
        commit_url, target, idx = args.commit_url or "", args.target, None
    if not code.strip():
        raise SystemExit("请通过 --idx / --func-file / --code 提供待检测函数")

    def on_event(event: dict) -> None:
        if args.quiet:
            return
        kind = event.get("type")
        if kind == "thought":
            print(f"  · 思考：{event.get('text', '')[:200]}")
        elif kind == "tool_call":
            print(f"  → 调用 {event.get('tool')}({json.dumps(event.get('args', {}), ensure_ascii=False)})")
        elif kind == "tool_result":
            print(f"  ← {event.get('tool')}：{event.get('preview', '')[:160]}")
        elif kind == "kb_hit":
            print(f"  · 知识检索命中 {event.get('count')} 条")
        elif kind == "snapshot":
            print(f"  · 仓库快照：{event.get('status')} {event.get('detail', '')[:160]}")
        elif kind == "verdict":
            print(f"  ✓ 结论：{event.get('verdict')}（置信度 {event.get('confidence')}）")

    result = detect_one(
        code, commit_url, idx=idx, target=target, mode=args.mode,
        repo_enabled=False if args.no_repo else None,
        on_event=on_event, save_trace=bool(idx),
    )
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print()
        print(f"idx: {result.idx}")
        print(f"检测结果: {result.verdict}（prediction={result.prediction}，置信度 {result.confidence}）")
        print(f"判定理由:\n{result.reason}")
        print(f"分析记录:\n{result.analysis_record()}")
        print(f"耗时 {result.elapsed}s / {result.steps} 轮 / {result.n_tool_calls} 次工具调用 / 模式 {result.mode}")
    return 0


def cmd_batch(args) -> int:
    from .batch import BatchRunner, iter_samples
    from .knowledge import KnowledgeBase
    from .pipeline import Detector

    kb = KnowledgeBase.load()
    detector = Detector(kb=kb, mode=args.mode, repo_enabled=False if args.no_repo else None)
    print(f"[batch] 模式={detector.mode_used} 模型={detector.model_name} 仓库上下文={'关闭' if args.no_repo else '按需'}")

    def on_progress(info: dict) -> None:
        if info["event"] == "start":
            print(f"[batch] 共 {info['total']} 条，已完成 {info['skipped']} 条（断点续跑）", flush=True)
        elif info["event"] == "progress":
            flag = "  ! " + info["error"] if info.get("error") else ""
            print(f"[{info['finished']}/{info['total']}] idx={info['idx']} → {info['verdict']}"
                  f"（{info['confidence']}） {info['elapsed']}s{flag}", flush=True)

    runner = BatchRunner(
        detector,
        result_path=Path(args.result),
        workers=args.workers,
        resume=not args.no_resume,
        save_trace=not args.no_trace,
        on_progress=on_progress,
    )
    summary = runner.run(iter_samples(Path(args.dataset), limit=args.limit, pairs=args.pairs))
    print("[batch] 完成：", json.dumps(summary, ensure_ascii=False))
    return 0


def cmd_eval(args) -> int:
    from .evaluation import evaluate, render_markdown

    report = evaluate(args.result, args.dataset)
    print(render_markdown(report))
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已保存 JSON 报告：{out}")
    return 0


def cmd_export(args) -> int:
    from .external import export_prompts, export_sample_prompts

    if args.sample:
        manifest = export_sample_prompts(pairs=args.sample, per_batch=args.per_batch)
        out = config.RESULTS_DIR / "export_sample"
    else:
        manifest = export_prompts(per_batch=args.per_batch, limit_pairs=args.pairs or None)
        out = config.RESULTS_DIR / "export"
    print(f"[export] 已导出 {len(manifest['batches'])} 个批次，共 {manifest['samples']} 条样本")
    print(f"[export] 目录：{out}  （作答说明见 template.md）")
    return 0


def cmd_import(args) -> int:
    from .external import import_answers

    summary = import_answers(args.answers, args.result)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["missing"]:
        print(f"[import] 警告：仍有 {len(summary['missing'])} 条样本未作答")
    return 0


def cmd_info(args) -> int:
    from .knowledge import KnowledgeBase
    from .llm import LLMSettings

    settings = config.Settings.load()
    print("== 运行环境 ==")
    print(f"项目根目录：{config.ROOT}")
    print(f"Python：{sys.version.split()[0]}  （{sys.executable}）")
    print("\n== 大模型 ==")
    print(f"模式：{'真实模型' if settings.llm.configured else '离线（未配置密钥）'}")
    print(f"配置：{settings.llm.describe()}")
    if settings.llm.configured:
        print(f"密钥来源：{settings.llm.key_source or '未知'}")
    else:
        print(f"配置文件：{config.ENV_FILE}（{'存在' if config.ENV_FILE.exists() else '不存在'}）")
        print("配置方法：python -m fva.cli setup     或   在项目根目录创建 .env 写入 FVA_LLM_API_KEY=sk-xxx")
    print("\n== 知识库 ==")
    if config.KB_CORPUS_PATH.exists():
        kb = KnowledgeBase.load()
        print(f"案例数：{len(kb.cases)}；语料：{config.KB_CORPUS_PATH}")
        print(f"索引：{config.KB_INDEX_PATH}（{config.KB_INDEX_PATH.stat().st_size / 1e6:.1f} MB）")
    else:
        print("尚未构建，请运行：python -m fva.cli build-kb")
    print("\n== 离线模型 ==")
    if config.OFFLINE_MODEL_PATH.exists():
        payload = json.loads(config.OFFLINE_MODEL_PATH.read_text(encoding="utf-8"))
        print(f"特征 {len(payload['features'])} 个，阈值 {payload['threshold']}，"
              f"交叉验证成对正确率 {payload['metrics'].get('pair_correct_rate')}")
    else:
        print("尚未训练，请运行：python -m fva.cli train")
    print("\n== 仓库快照缓存 ==")
    snapshots = list(config.REPO_CACHE_DIR.glob("*/.fva_ready"))
    tarballs = list(config.TARBALL_CACHE_DIR.glob("*.tgz"))
    size_mb = sum(p.stat().st_size for p in config.TARBALL_CACHE_DIR.glob("*.tgz")) / (1024 * 1024)
    print(f"已缓存快照 {len(snapshots)} 个，压缩包 {len(tarballs)} 个（{size_mb:.1f} MB）")
    return 0


def cmd_setup(args) -> int:
    """交互式写入 .env 并做一次连通性自检（适合第一次配置的用户）。"""
    import getpass

    from . import config as cfg
    from .llm import LLMSettings, check_connection

    current = LLMSettings.from_env()
    base_url = (args.base_url or current.base_url or "https://api.deepseek.com").rstrip("/")
    model = args.model or current.model or "deepseek-chat"
    api_key = (args.api_key or "").strip()

    print("== 配置大模型（写入项目根目录的 .env，不会提交到 git）==")
    print(f"BASE_URL : {base_url}")
    print(f"MODEL    : {model}")
    if not api_key:
        hint = f"（当前已配置：{current.masked_key}，直接回车沿用）" if current.api_key else ""
        try:
            api_key = getpass.getpass(f"API Key {hint}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[取消] 未写入任何配置")
            return 2
    if not api_key:
        api_key = current.api_key
    if not api_key:
        print("[错误] 没有提供 API Key，已取消。", file=sys.stderr)
        return 2

    path = cfg.write_env_file({
        "FVA_LLM_BASE_URL": base_url,
        "FVA_LLM_API_KEY": api_key,
        "FVA_LLM_MODEL": model,
    })
    print(f"\n✓ 已写入：{path}")

    shadowed = [name for name in ("FVA_LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "LLM_TOKEN")
                if (os.environ.get(name) or "").strip()]
    if shadowed:
        print(f"! 注意：当前进程里存在环境变量 {shadowed}，它的优先级高于 .env。")
        print("  如果它指向旧的密钥，请先关闭那个窗口/取消该变量，否则会用旧值。")

    if args.no_check:
        print("（已跳过连通性检查；可随时执行 python -m fva.cli check 验证）")
        return 0

    print("\n== 连通性自检 ==")
    settings = LLMSettings.from_env()
    ok, message = check_connection(settings)
    print(("✓ " if ok else "✗ ") + message)
    if ok:
        print("\n配置完成。现在双击「启动网页界面.bat」即可使用真实大模型模式。")
        return 0
    print("\n配置已保存，但自检未通过；修正后重试：python -m fva.cli check", file=sys.stderr)
    return 1


def cmd_check(args) -> int:
    from .llm import LLMSettings, check_connection

    settings = LLMSettings.from_env()
    print(f"模型：{settings.model}")
    print(f"地址：{settings.base_url}")
    print(f"密钥：{settings.masked_key or '（空）'}  来源：{settings.key_source or '未配置'}")
    print(f"配置文件：{settings.env_file}")
    print()
    ok, message = check_connection(settings)
    print(("✓ " if ok else "✗ ") + message)
    return 0 if ok else 1


def cmd_ui(args) -> int:
    """启动 Streamlit 界面，并自动打开浏览器。

    以子进程方式启动，因此控制台窗口会一直保留（关闭窗口即停止服务），
    便于用户看到启动日志与报错信息。
    """
    import threading
    import webbrowser

    app = Path(config.ROOT) / "app" / "streamlit_app.py"
    if not app.exists():
        raise SystemExit(f"未找到界面文件：{app}")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app),
           "--server.port", str(args.port), "--server.headless", "true"]
    url = f"http://localhost:{args.port}"
    from .llm import LLMSettings

    llm = LLMSettings.from_env()
    if llm.configured:
        print(f"大模型：{llm.describe()}", flush=True)
    else:
        print("大模型：未配置密钥 —— 将以「离线分析」模式运行。", flush=True)
        print("        配置方法：python -m fva.cli setup   （或编辑项目根目录的 .env）", flush=True)
    print(f"界面地址：{url}", flush=True)
    print("（浏览器未自动打开时请手动访问；按 Ctrl+C 或关闭本窗口即停止服务）", flush=True)
    print(flush=True)
    if not args.no_browser:
        threading.Timer(4.0, lambda: webbrowser.open(url)).start()
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fva",
        description="函数漏洞检测智能体工具（FVA）：基于 LLM + 领域知识库 + 仓库上下文挖掘",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-kb", help="构建领域知识库索引")
    p.add_argument("--force", action="store_true", help="强制重建")
    p.set_defaults(func=cmd_build_kb)

    p = sub.add_parser("train", help="训练离线判别器（消融基线 / 降级模式）")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("detect", help="检测单个函数")
    p.add_argument("--idx", type=int, help="从测试集按 idx 取样本")
    p.add_argument("--func-file", type=str, help="函数源码文件")
    p.add_argument("--code", type=str, help="函数源码字符串")
    p.add_argument("--commit-url", type=str, default="", help="修复提交链接")
    p.add_argument("--target", type=int, choices=[0, 1], help="样本标签（仅用于确定仓库版本，不参与判定）")
    p.add_argument("--mode", choices=["auto", "llm", "offline"], default="auto")
    p.add_argument("--no-repo", action="store_true", help="不读取仓库上下文")
    p.add_argument("--json", action="store_true", help="输出完整 JSON")
    p.add_argument("--quiet", action="store_true", help="不打印过程事件")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("batch", help="批量检测 JSONL 测试集")
    p.add_argument("--dataset", type=str, default=str(config.TEST_PATH))
    p.add_argument("--result", type=str, default=str(config.RESULT_PATH))
    p.add_argument("--limit", type=int, help="最多检测多少条")
    p.add_argument("--pairs", type=int, help="最多检测多少对（2N 条）")
    p.add_argument("--workers", type=int, default=1, help="并发数（离线模式建议 4）")
    p.add_argument("--mode", choices=["auto", "llm", "offline"], default="auto")
    p.add_argument("--no-repo", action="store_true", help="不读取仓库上下文")
    p.add_argument("--no-resume", action="store_true", help="不续跑，清空已有结果")
    p.add_argument("--no-trace", action="store_true", help="不保存每条事件流")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("eval", help="按课程口径评测 result.jsonl")
    p.add_argument("--result", type=str, default=str(config.RESULT_PATH))
    p.add_argument("--dataset", type=str, default=str(config.TEST_PATH))
    p.add_argument("--save", type=str, default=str(config.RESULTS_DIR / "eval.json"))
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("setup", help="配置大模型密钥（写入 .env）并做连通性自检")
    p.add_argument("--api-key", type=str, help="直接指定 API Key（不传则交互输入）")
    p.add_argument("--base-url", type=str, help="接口根地址，如 https://api.deepseek.com")
    p.add_argument("--model", type=str, help="模型名，如 deepseek-chat")
    p.add_argument("--no-check", action="store_true", help="写入后不做联网自检")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("check", help="检查大模型配置与连通性")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("info", help="查看环境、知识库与缓存状态")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("export", help="导出批量提示词（供外部模型/会话作答）")
    p.add_argument("--per-batch", type=int, default=5, help="每批样本数")
    p.add_argument("--pairs", type=int, help="只导出前 N 对（2N 条）")
    p.add_argument("--sample", type=int, default=0,
                   help="抽样评测：等间隔抽取 N 对，输出到 results/export_sample（每批 3 漏洞版 + 2 修复版）")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("import", help="回填外部模型作答，生成 result.jsonl")
    p.add_argument("--answers", type=str, default=str(config.RESULTS_DIR / "export"))
    p.add_argument("--result", type=str, default=str(config.RESULTS_DIR / "result_llm.jsonl"))
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("ui", help="启动图形界面（Streamlit）")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    p.set_defaults(func=cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from .llm import LLMError

    try:
        return args.func(args)
    except LLMError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[中断] 已停止（批量结果已写入，可重新执行以断点续跑）", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
