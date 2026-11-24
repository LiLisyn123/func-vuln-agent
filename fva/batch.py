"""批量检测：读 JSONL 测试集 → 逐条检测 → 汇总 result.jsonl。

工程要点
--------
* **可断点续跑**：已存在结果中的 idx 会跳过，长任务中断后不必从头开始；
* **并发受控**：默认单线程（离线模式可开并发），并发时用锁保护结果文件写入；
* **产物齐全**：除 `result.jsonl` 外，每条样本的完整报告与事件流分别落盘，
  便于事后复核「为什么这么判」；
* **失败隔离**：单条样本异常只影响自己，不会中断整批任务。
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Iterator

from . import config
from .pipeline import Detector, DetectionResult
from .textutil import read_jsonl


def iter_samples(path: Path, limit: int | None = None, pairs: int | None = None) -> Iterator[dict]:
    """按行读取测试集；pairs 表示只取前 N 对（2N 条）。"""
    count = 0
    for record in read_jsonl(path):
        yield record
        count += 1
        if pairs and count >= pairs * 2:
            return
        if limit and count >= limit:
            return


def load_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    for record in read_jsonl(path):
        if isinstance(record.get("idx"), int):
            done.add(record["idx"])
    return done


class BatchRunner:
    """批量检测执行器。"""

    def __init__(
        self,
        detector: Detector,
        result_path: Path = config.RESULT_PATH,
        *,
        workers: int = 1,
        resume: bool = True,
        save_trace: bool = True,
        on_progress: Callable[[dict], None] | None = None,
    ):
        self.detector = detector
        self.result_path = Path(result_path)
        self.workers = max(1, workers)
        self.resume = resume
        self.save_trace = save_trace
        self.on_progress = on_progress or (lambda info: None)
        self._lock = threading.Lock()
        self._results: list[DetectionResult] = []

    # ---------------------------------------------------------- 主流程
    def run(self, samples: Iterable[dict]) -> dict:
        samples = list(samples)
        config.ensure_dirs()
        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        done = load_done(self.result_path) if self.resume else set()
        if not self.resume and self.result_path.exists():
            self.result_path.unlink()
        pending = [s for s in samples if s.get("idx") not in done]
        started = time.time()
        total = len(samples)
        finished = total - len(pending)
        self.on_progress({"event": "start", "total": total, "skipped": finished,
                          "mode": self.detector.mode_used, "workers": self.workers})

        if self.workers == 1:
            for sample in pending:
                result = self._detect(sample)
                finished += 1
                self._emit_progress(finished, total, result, started)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(self._detect, sample): sample for sample in pending}
                for future in as_completed(futures):
                    result = future.result()
                    finished += 1
                    self._emit_progress(finished, total, result, started)

        elapsed = round(time.time() - started, 1)
        summary = self._summary(samples, elapsed)
        self.on_progress({"event": "done", **summary})
        return summary

    # ---------------------------------------------------------- 单条
    def _detect(self, sample: dict) -> DetectionResult:
        try:
            result = self.detector.detect(
                sample.get("func") or "",
                sample.get("commit_url") or "",
                idx=sample.get("idx"),
                target=sample.get("target"),
                save_trace=self.save_trace,
            )
        except Exception as exc:  # 单条失败不影响整批
            result = DetectionResult(
                idx=sample.get("idx"), prediction=1, verdict="有漏洞", confidence=0.0,
                reason=f"检测异常：{type(exc).__name__}: {exc}", commit_url=sample.get("commit_url") or "",
                target=sample.get("target"), mode=self.detector.mode_used,
                model=self.detector.model_name, error=str(exc),
            )
        self._write(result)
        return result

    def _write(self, result: DetectionResult) -> None:
        line = json.dumps({"idx": result.idx, "prediction": result.prediction}, ensure_ascii=False)
        with self._lock:
            with self.result_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self._results.append(result)

    def _emit_progress(self, finished: int, total: int, result: DetectionResult, started: float) -> None:
        elapsed = time.time() - started
        rate = finished / elapsed if elapsed > 0 else 0
        self.on_progress({
            "event": "progress",
            "finished": finished,
            "total": total,
            "idx": result.idx,
            "prediction": result.prediction,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "elapsed": result.elapsed,
            "rate": round(rate, 2),
            "error": result.error,
        })

    # ---------------------------------------------------------- 汇总
    def _summary(self, samples: list[dict], elapsed: float) -> dict:
        by_prediction = {0: 0, 1: 0}
        errors = 0
        durations = []
        tools = 0
        for result in self._results:
            by_prediction[result.prediction] = by_prediction.get(result.prediction, 0) + 1
            if result.error:
                errors += 1
            durations.append(result.elapsed)
            tools += result.n_tool_calls
        n = len(self._results)
        return {
            "samples": len(samples),
            "detected": n,
            "by_prediction": by_prediction,
            "errors": errors,
            "avg_elapsed": round(sum(durations) / n, 2) if n else 0.0,
            "avg_tool_calls": round(tools / n, 2) if n else 0.0,
            "wall_time": elapsed,
            "result_path": str(self.result_path),
            "mode": self.detector.mode_used,
        }
