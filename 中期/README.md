# 中期汇报材料

| 文件 | 说明 |
|---|---|
| `中期汇报.pdf` | **汇报用幻灯片**（19 页，4:3，与开题同一套 SINTEF/UCAS 主题） |
| `汇报讲稿.md` | 逐页讲稿（约 11–12 分钟）、问答要点、现场演示脚本 |
| `For beamer/midterm.tex` | 幻灯片源文件（XeLaTeX / Beamer），改完重新编译即可 |
| `For beamer/assets/shots/` | 界面截图（幻灯片第 13 页使用） |
| `课程项目中期汇报要求.pptx` | 课程给的中期汇报要求 |

## 重新编译

```bat
cd 中期\For beamer
xelatex midterm.tex
xelatex midterm.tex
```

生成的 `midterm.pdf` 复制/覆盖到 `中期/中期汇报.pdf` 即可。

> 需要 TeX Live（含 `xeCJK` 与 Microsoft YaHei 字体，Windows 自带字体即可）。

## 汇报前可选调整

1. **（可选）组号**：封面 `\author{...}` 前加 `第 X 组\quad`；
2. **（可选）界面截图**：`assets/shots/*.png` 目前是离线模式截图，
   用真实模型重新截图覆盖同名文件后重新编译即可。

> 代码库地址已填为 `github.com/LiLisyn123/func-vuln-agent`（第 3 页与最后一页）。

## 内容对应课程要求

| 课程要求 | 对应页 |
|---|---|
| 项目名称、第 X 组、组员 | 第 1 页 |
| 项目代码库地址 | 第 3 页（含协作方式） |
| 团队组成及分工 | 第 3 页表格 |
| 开发计划与进度安排 | 第 4 页 |
| 项目意义和目标 | 第 5 页 |
| 实施技术方案（版本控制/协作/数据库/框架/算法） | 第 6、7 页 |
| 系统架构 | 第 8 页 |
| 功能组成及介绍 | 第 9 页 |
| BDD user stories | 第 10 页 |
| low-fi UI | 第 11 页 |
| 开发进展与截图 | 第 12、13 页 |
| 评测结果与分析 | 第 14、15 页 |
| 创新点、特色 | 第 16 页 |
| 当前存在的问题 | 第 17 页 |
| 下一步优化计划 | 第 18 页 |

## 数据来源

- 评测结果：`results/ui_result_212544.jsonl` → `results/eval_ui_212544.json`
  （P-C 28 / P-V 87 / P-B 19 / P-R 16，等级 C）
- 基线对照：离线判别器 `results/eval.json`（P-C 9）；抽样 LLM `results/eval_sample.json`（P-C 5/25）
- 工程指标：知识库 3891 条案例、31 项测试、检索 P50 5.8 ms 等（见 `docs/项目报告.md` 第 5 章）
