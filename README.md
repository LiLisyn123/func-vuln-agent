# 函数漏洞检测智能体工具（FVA）

> 2026 秋季学期《高级软件工程》课程大作业
>
> 组长：李柏谚　组员：杨聃、王义卓、孙伊凡、徐培栋

输入一个 C/C++ 函数与它所在仓库的修复提交链接，**自主 Agent** 会先检索历史 CVE 漏洞案例（RAG），
再按需到代码仓库里挖掘调用方 / 被调用方 / 全局变量等上下文，最后给出三样东西：
**检测结果 —— 判定理由 —— 完整分析记录**。

- 📄 项目报告：[`docs/项目报告.md`](docs/项目报告.md)
- 🎤 中期汇报：[`中期/中期汇报.pdf`](中期/中期汇报.pdf)（讲稿见 [`中期/汇报讲稿.md`](中期/汇报讲稿.md)）
- 🧭 新手说明：[`交付说明.md`](交付说明.md)
- 🚀 使用方法：[`使用教程.md`](使用教程.md)

---

## 1. 快速开始

```bash
# 1) 安装依赖（核心只依赖标准库；streamlit 用于界面，numpy 仅用于训练离线基线）
python -m pip install -r requirements.txt

# 2) 构建领域知识库（首次约 5 秒，生成 cache/kb_corpus.jsonl 与 cache/kb_index.pkl.gz）
python -m fva.cli build-kb

# 3) 配置大模型（DeepSeek 或任意 OpenAI 兼容端点）
set FVA_LLM_API_KEY=sk-xxxx          # Windows
export FVA_LLM_API_KEY=sk-xxxx       # Linux / macOS

# 4) 启动图形界面
python -m fva.cli ui                 # 浏览器打开 http://localhost:8501
```

Windows 下也可以直接双击 `启动网页界面.bat`（脚本会自动在 `.venv\Scripts\python.exe`、`python`、
`py -3` 中选择可用的解释器，并在失败时停住窗口显示原因；若浏览器没有自动打开，手动访问
`http://localhost:8501`）。批量跑分则双击 `一键批量评测.bat`；配置密钥可以双击 `配置大模型.bat`。

### 命令行用法

```bash
# 检测测试集里的某一条（含仓库上下文挖掘）
python -m fva.cli detect --idx 194963

# 检测自己写的函数
python -m fva.cli detect --func-file my_func.c --commit-url https://github.com/owner/repo/commit/<sha> --target 1

# 批量检测 300 条测试集，结果写入 results/result.jsonl（支持断点续跑）
python -m fva.cli batch --workers 2

# 按课程口径评测
python -m fva.cli eval

# 查看环境、知识库与缓存状态
python -m fva.cli info
```

---

## 2. 系统架构

```
表现层  Streamlit 界面：输入函数/仓库、实时看过程、接收报告
   │
智能体层  自研 ReAct Agent：推理 → 调用工具 → 观察结果 → 再推理，证据充分后才下结论
   │            ├── 预算控制（轮数 / 工具次数 / 观察长度）
   │            └── 过程留痕（每次思考、调用、观察都写入事件流）
工具层    ① 领域知识检索（BM25 over 历史 CVE 成对案例）
          ② 仓库上下文挖掘（grep / 读文件 / 找定义 / 调用方 / 被调用方 / 符号）
数据层    历史漏洞案例库（dataset/knowledge.jsonl）｜仓库版本快照缓存（cache/）
```

| 目录 | 说明 |
|---|---|
| `fva/` | 核心包：配置、知识库、仓库工具、Agent、流水线、评测、命令行 |
| `app/streamlit_app.py` | 图形界面（单样本 / 批量 / 评测 / 知识库） |
| `tests/` | 31 项自动化测试（`python -m unittest discover -s tests`） |
| `dataset/` | 课程提供的数据集与官方评测脚本 |
| `cache/` | 自动生成的索引与仓库快照（可删除重建） |
| `results/` | `result.jsonl`、每条样本的报告与事件流、评测报告 |
| `docs/项目报告.md` | 项目报告（交付物） |

---

## 3. 三条红线的落实方式

| 红线 | 落实 |
|---|---|
| **必须自主 Agent** | `fva/agent/react.py` 实现 ReAct 状态机：模型自主决定调用哪些工具、调用几次、何时停止；框架只做能力边界、预算与留痕，不预置流水线顺序 |
| **严格成对隔离** | 单样本检测只发送一个函数；仓库快照按标签取版本（`target=1` → 父提交 `parent`，`target=0` → 提交后版本 `self`），见 `fva/repo/snapshot.py`；任何工具都不提供 diff |
| **禁止网络搜索** | 唯一对外网络能力是按 `commit_url` 下载指定版本快照（codeload / git），模型端只连配置好的推理端点，无搜索、无任意 URL 抓取 |

此外，知识库只使用课程提供的数据集构建，并在建库时做与测试集的污染校验（见 `build-kb` 输出）。

---

## 4. 配置

配置读取顺序：**环境变量 → 项目根目录 `.env` → 内置默认值**。
日常使用推荐写 `.env`（双击 `.bat` 也能读到；只在某个 cmd 窗口 `set` 的变量，双击启动的新进程读不到）：

```ini
# .env（可复制 .env.example 改名；已被 .gitignore 忽略）
FVA_LLM_BASE_URL=https://api.deepseek.com
FVA_LLM_API_KEY=sk-你的密钥
FVA_LLM_MODEL=deepseek-chat
```

也可以让程序引导写入并自动联网自检：

```bash
python -m fva.cli setup     # 交互式写入 .env + 连通性检查
python -m fva.cli check     # 随时检查「密钥 / 地址 / 模型」是否可用
python -m fva.cli info      # 看当前模式、密钥来源与配置文件路径
```

| 变量 | 默认 | 说明 |
|---|---|---|
| `FVA_LLM_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容端点 |
| `FVA_LLM_API_KEY` | 空 | 未配置时自动进入离线模式 |
| `FVA_LLM_MODEL` | `deepseek-chat` | 模型名 |
| `FVA_LLM_TEMPERATURE` | `0.0` | 低温度降低判定方差 |
| `FVA_AGENT_MAX_STEPS` | `12` | Agent 最多思考轮数 |
| `FVA_AGENT_MAX_TOOL_CALLS` | `20` | 最多工具调用次数 |
| `FVA_SNAPSHOT_MAX_MB` | `300` | 仓库体量上限（超大仓库自动降级为纯知识分析） |
| `FVA_GITHUB_TOKEN` | 空 | 可选，提高 GitHub API 速率 |

**未配置密钥时**会自动使用离线判别器（`fva/offline.py`）：界面与流程照常可跑，
用于演示与消融基线；正式评测请配置密钥后用 `--mode llm` 复跑。

---

## 5. 评测

课程口径按「相邻两条为一对」统计四类，P-C 对数决定等级（A ≥ 30、B 25–29、C 20–24、D < 20），
且 P-R 超过 P-C 一半时降一级。本项目 `python -m fva.cli eval` 直接调用课程提供的
`dataset/evaluation.py`，保证统计口径与官方一致。

```bash
python -m fva.cli batch --workers 2      # 生成 results/result.jsonl
python -m fva.cli eval                   # 输出四类组数与等级，保存 results/eval.json
```

> `results/result.jsonl` 若为离线模式产物，会在报告与 `results/README.md` 中明确标注；
> 配置密钥后重新执行 `batch` 即可覆盖为真实 Agent 的结果。

---

## 6. 无 API Key 时的替代路径

如果暂时拿不到可直连的 API Key，可以导出提示词，用任意可用的模型会话作答，再回填结果：

```bash
python -m fva.cli export --per-batch 5          # 生成 results/export/batch_*.json
# 用外部模型按 results/export/template.md 的格式逐批作答，写出 answer_*.json
python -m fva.cli import --answers results/export --result results/result_llm.jsonl
python -m fva.cli eval --result results/result_llm.jsonl
```

导出的提示词与 API 模式共用同一套系统提示词、知识检索上下文与判定口径，且保证同一对的
两个版本不会出现在同一批次（严格成对隔离）。
