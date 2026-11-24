"""仓库快照：按修复提交链接获取「与样本标签对应」的那一个版本。

红线（严格版本控制）
--------------------
* 样本标签 target=1（漏洞版）→ 只允许取修复提交的**父提交**版本；
* 样本标签 target=0（修复版）→ 只允许取修复提交**提交后**的版本；
* 不提供 git diff、不提供另一版本的任何内容；快照下载后即为该版本的完整工作区。

实现要点
--------
* 缓存键为 (仓库, 版本 SHA)，同一提交的多条样本复用同一份快照；
* 先用 GitHub API 解析父提交并做仓库体量预估（超大仓库直接拒绝，避免拉取
  TensorFlow / Linux 这类仓库时把磁盘和带宽打满），再用 codeload 下载 tarball；
* 非 GitHub 托管的仓库退化为 git 命令（浅取指定版本），失败即返回明确原因，
  由 Agent 在缺少仓库上下文的情况下继续推理，而不是中断检测。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..textutil import load_json, dump_json

USER_AGENT = "func-vuln-agent/1.0 (+course project)"
SHA_RE = re.compile(r"\b([0-9a-f]{7,40})\b")


class SnapshotError(RuntimeError):
    """快照不可用（网络、体量、托管平台限制等），调用方据此降级。"""


@dataclass
class CommitRef:
    provider: str          # github / gitlab / gitweb / other
    host: str
    owner: str
    repo: str
    sha: str
    url: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}" if self.owner and self.repo else self.host

    @property
    def clone_url(self) -> str:
        if self.provider == "github":
            return f"https://github.com/{self.owner}/{self.repo}.git"
        if self.provider == "gitlab":
            return f"https://gitlab.com/{self.owner}/{self.repo}.git"
        return self.url


@dataclass
class Snapshot:
    root: Path
    ref: CommitRef
    snapshot_sha: str
    rule: str              # parent / self
    source: str            # codeload / git / cache
    cache_hit: bool = False
    meta: dict = field(default_factory=dict)


def parse_commit_url(url: str) -> CommitRef:
    """解析修复提交链接，支持 GitHub / GitLab / gitweb(cgit) 等常见形式。"""
    raw = (url or "").strip()
    if not raw:
        raise SnapshotError("缺少仓库链接")
    parsed = urllib.parse.urlparse(raw)
    host = parsed.netloc.lower()
    path = parsed.path or ""
    sha = ""

    if "github.com" in host:
        m = re.match(r"^/([^/]+)/([^/]+)/commit/([0-9a-f]{7,40})", path)
        if not m:
            raise SnapshotError(f"无法解析 GitHub 提交链接：{raw}")
        return CommitRef("github", host, m.group(1), m.group(2), m.group(3), raw)
    if "gitlab" in host:
        m = re.match(r"^/(.+?)/([^/]+)/-/commit/([0-9a-f]{7,40})", path)
        if m:
            return CommitRef("gitlab", host, m.group(1), m.group(2), m.group(3), raw)
        m = re.match(r"^/(.+?)/([^/]+)/commit/([0-9a-f]{7,40})", path)
        if m:
            return CommitRef("gitlab", host, m.group(1), m.group(2), m.group(3), raw)

    # gitweb / cgit：?p=xxx.git;a=commit;h=<sha> 或 /commit/?id=<sha>
    for pattern in (r"[?&;]h=([0-9a-f]{7,40})", r"[?&]id=([0-9a-f]{7,40})", r"/commit/([0-9a-f]{7,40})"):
        m = re.search(pattern, raw)
        if m:
            sha = m.group(1)
            break
    if not sha:
        m = SHA_RE.search(path)
        sha = m.group(1) if m else ""
    if not sha:
        raise SnapshotError(f"链接中未找到提交 SHA：{raw}")
    # gitweb 的 ?p=qemu.git 给出仓库名
    project = ""
    m = re.search(r"[?&;]p=([^;&]+)", raw)
    if m:
        project = m.group(1).replace(".git", "")
    owner, repo = (project.split("/", 1) + ["", ""])[:2] if "/" in project else ("", project)
    return CommitRef("gitweb" if ("a=commit" in raw or "cgit" in raw) else "other",
                     host, owner, repo, sha, raw)


# ---------------------------------------------------------------- HTTP


def _http_json(url: str, token: str = "", timeout: float = 30.0) -> dict:
    request = urllib.request.Request(url, headers=_headers(token))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _headers(token: str = "") -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


class SnapshotManager:
    """负责解析版本、下载、解压与缓存。"""

    def __init__(self, settings: config.RepoSettings | None = None):
        self.settings = settings or config.RepoSettings.from_env()
        config.ensure_dirs()
        self.meta = load_json(config.SNAPSHOT_META_PATH, default={}) or {}
        self._memory: dict[str, Snapshot] = {}

    # ---------------------------------------------------------- 元数据
    def _save_meta(self) -> None:
        dump_json(config.SNAPSHOT_META_PATH, self.meta)

    # ---------------------------------------------------------- 版本解析
    def resolve_parent_sha(self, ref: CommitRef) -> str:
        """解析父提交 SHA（漏洞版必须使用父提交快照）。"""
        cache_key = f"parent:{ref.slug}@{ref.sha}"
        if cache_key in self.meta:
            return self.meta[cache_key]["sha"]
        if ref.provider != "github":
            raise SnapshotError("非 GitHub 仓库暂不支持自动解析父提交")
        api = f"https://api.github.com/repos/{ref.owner}/{ref.repo}/commits/{ref.sha}"
        payload = _http_json(api, self.settings.github_token)
        parents = payload.get("parents") or []
        if not parents:
            raise SnapshotError(f"提交 {ref.sha[:10]} 没有父提交（仓库首个提交）")
        parent = parents[0]["sha"]
        self.meta[cache_key] = {"sha": parent, "resolved_at": time.time()}
        self._save_meta()
        return parent

    def repo_size_kb(self, ref: CommitRef) -> int | None:
        """通过 GitHub API 估算仓库体量（KB），用于下载前的成本守卫。"""
        if ref.provider != "github":
            return None
        key = f"size:{ref.slug}"
        if key in self.meta:
            return self.meta[key]["kb"]
        try:
            payload = _http_json(f"https://api.github.com/repos/{ref.owner}/{ref.repo}",
                                 self.settings.github_token)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None
        size = int(payload.get("size") or 0)
        self.meta[key] = {"kb": size, "checked_at": time.time()}
        self._save_meta()
        return size

    # ---------------------------------------------------------- 主流程
    def ensure_snapshot(self, commit_url: str, target: int,
                        allow_fetch: bool = True) -> Snapshot:
        """获取与样本标签对应的仓库快照。

        target=1 → 父提交版本；target=0 → 提交后（即该提交）版本。
        """
        ref = parse_commit_url(commit_url)
        memory_key = f"{ref.slug}@{ref.sha}@{target}"
        if memory_key in self._memory:
            return self._memory[memory_key]

        if target == 1:
            rule = "parent"
            if not re.fullmatch(r"[0-9a-f]{40}", ref.sha):
                full = self._expand_sha(ref)
                ref = CommitRef(ref.provider, ref.host, ref.owner, ref.repo, full, ref.url)
            snapshot_sha = self.resolve_parent_sha(ref)
        else:
            rule = "self"
            snapshot_sha = ref.sha
            if not re.fullmatch(r"[0-9a-f]{40}", snapshot_sha):
                snapshot_sha = self._expand_sha(ref)

        root = config.REPO_CACHE_DIR / f"{_safe(ref.slug)}__{snapshot_sha[:16]}"
        if (root / ".fva_ready").exists():
            snap = Snapshot(root, ref, snapshot_sha, rule, "cache", cache_hit=True, meta=self.meta.get(
                f"snap:{ref.slug}@{snapshot_sha}", {}))
            self._memory[memory_key] = snap
            return snap
        if not allow_fetch:
            raise SnapshotError("已禁用仓库下载")

        size = self.repo_size_kb(ref)
        limit_kb = self.settings.max_snapshot_mb * 1024
        if size is not None and size > limit_kb:
            raise SnapshotError(
                f"仓库体量 {size / 1024:.1f} MB 超过上限 {self.settings.max_snapshot_mb} MB，"
                f"为保证成本可控跳过仓库上下文（仅用领域知识分析）"
            )

        start = time.time()
        if ref.provider in ("github", "gitlab"):
            source = "codeload"
            self._fetch_tarball(ref, snapshot_sha, root)
        else:
            source = "git"
            self._fetch_git(ref, snapshot_sha, rule, root)
        (root / ".fva_ready").write_text(json.dumps({
            "slug": ref.slug, "version": snapshot_sha, "rule": rule, "source": source,
        }), encoding="utf-8")
        elapsed = round(time.time() - start, 1)
        meta = {
            "slug": ref.slug, "commit_url": commit_url, "target": target,
            "snapshot_sha": snapshot_sha, "rule": rule, "source": source,
            "elapsed_s": elapsed, "repo_size_kb": size,
        }
        self.meta[f"snap:{ref.slug}@{snapshot_sha}"] = meta
        self._save_meta()
        snap = Snapshot(root, ref, snapshot_sha, rule, source, cache_hit=False, meta=meta)
        self._memory[memory_key] = snap
        return snap

    # ---------------------------------------------------------- 下载实现
    def _expand_sha(self, ref: CommitRef) -> str:
        if ref.provider != "github":
            raise SnapshotError("仅 GitHub 支持短 SHA 展开")
        api = f"https://api.github.com/repos/{ref.owner}/{ref.repo}/commits/{ref.sha}"
        payload = _http_json(api, self.settings.github_token)
        return payload["sha"]

    def _fetch_tarball(self, ref: CommitRef, sha: str, root: Path) -> None:
        if ref.provider == "github":
            url = f"https://codeload.github.com/{ref.owner}/{ref.repo}/tar.gz/{sha}"
        else:
            url = f"https://gitlab.com/{ref.owner}/{ref.repo}/-/archive/{sha}/{ref.repo}-{sha}.tar.gz"
        tarball = config.TARBALL_CACHE_DIR / f"{_safe(ref.slug)}__{sha[:16]}.tgz"
        if not tarball.exists():
            self._download(url, tarball)
        self._extract(tarball, root)

    def _download(self, url: str, dest: Path) -> None:
        limit = self.settings.max_snapshot_mb * 1024 * 1024
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self.settings.download_timeout) as response, \
                    tmp.open("wb") as handle:
                written = 0
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > limit:
                        raise SnapshotError(f"压缩包超过 {self.settings.max_snapshot_mb} MB 上限，已中止下载")
                    handle.write(chunk)
            tmp.replace(dest)
        except urllib.error.HTTPError as exc:
            tmp.unlink(missing_ok=True)
            raise SnapshotError(f"下载失败 HTTP {exc.code}：{url}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            raise SnapshotError(f"下载失败：{exc}") from exc

    def _extract(self, tarball: Path, root: Path) -> None:
        tmp = Path(tempfile.mkdtemp(dir=config.REPO_CACHE_DIR, prefix=".extract-"))
        try:
            with tarfile.open(tarball, "r:gz") as archive:
                if hasattr(tarfile, "data_filter"):
                    archive.extractall(tmp, filter="data")
                else:  # 兼容旧版本：手工过滤绝对路径与上跳路径
                    members = [m for m in archive.getmembers()
                               if not (m.name.startswith("/") or ".." in Path(m.name).parts)]
                    archive.extractall(tmp, members=members)
            entries = [p for p in tmp.iterdir()]
            if len(entries) == 1 and entries[0].is_dir():
                source = entries[0]
            else:
                source = tmp
            root.parent.mkdir(parents=True, exist_ok=True)
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
            shutil.move(str(source), str(root))
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    def _fetch_git(self, ref: CommitRef, sha: str, rule: str, root: Path) -> None:
        if not shutil.which("git"):
            raise SnapshotError("非 GitHub 仓库需要本机 git，但未找到 git 命令")
        tmp = Path(tempfile.mkdtemp(dir=config.REPO_CACHE_DIR, prefix=".git-"))
        try:
            self._run_git(["init", "-q", str(tmp)])
            self._run_git(["-C", str(tmp), "remote", "add", "origin", ref.clone_url])
            # 先取目标提交（深度 2 保证父提交可用），再检出对应版本
            self._run_git(["-C", str(tmp), "fetch", "--depth", "2", "-q", "origin", sha],
                          timeout=max(120.0, self.settings.download_timeout))
            checkout = sha if rule == "self" else f"{sha}^"
            self._run_git(["-C", str(tmp), "-c", "advice.detachedHead=false", "checkout", "-q", checkout])
            git_dir = tmp / ".git"
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
            root.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp), str(root))
            shutil.rmtree(root / ".git", ignore_errors=True)
        except SnapshotError:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def _run_git(self, args: list[str], timeout: float = 120.0) -> None:
        try:
            result = subprocess.run(["git"] + args, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise SnapshotError(f"git 操作超时：{' '.join(args[:3])}") from exc
        if result.returncode != 0:
            raise SnapshotError(f"git 失败：{result.stderr.strip()[:300]}")


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text or "unknown")


def snapshot_dir_size_mb(root: Path) -> float:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return round(total / (1024 * 1024), 1)
