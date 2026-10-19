# 中文阅读效率自动评估 · 静态演示版

中山大学心理学系 · 刘彦平 开发

纯静态站点，无需后端。全部计算在浏览器完成，不上传任何数据。

## 功能

- 实验材料模式：296 句 × 4 条件，难度分与 Python 版逐位一致
- 任意文本模式：字级 n-gram 蒸馏近似，留出句 Pearson r = 0.92
- 眼动叠加：按 DataViewer 风格在句子上方绘制空心圆，圆越大注视越久，不遮挡文字
- 读者效率：阅读速度、平均凝视时间、跳读率与 45 人常模的百分位对照
- HTML 报告导出

## 目录

```
docs/                     ← GitHub Pages 发布目录（约 1.4 MB）
  index.html
  app.js
  style.css
  .nojekyll               ← 关闭 Jekyll，确保 data/ 原样发布
  data/
    materials.json        1181 句实验材料的词边界与难度（GPT-2 预计算）
    material_gaze.json    内置实验眼动记录（266 句，被试编号已脱敏）
    ngram.json            字级 n-gram 蒸馏模型（任意文本近似）
    scoring.json          Ridge 参数
    vocab.json            分词词典（4521 词）
    norm.json             45 名被试的三项聚合常模
    sample_text.txt       演示用任意文本
    sample_reading.asc    演示用模拟眼动数据（MOCK，非真实被试）

build_static.py           生成 materials.json / norm.json
build_material_gaze.py    从真实 asc 生成内置眼动 material_gaze.json
build_ngram.py            用 GPT-2 蒸馏 n-gram，生成 ngram/scoring/vocab
make_demo_data.py         生成示例文本与模拟 asc
```

## 本地预览

```powershell
cd docs
python -m http.server 8000
```

打开 http://127.0.0.1:8000/ 。直接双击 `index.html` 会因浏览器同源策略无法加载 JSON。

## 部署到 GitHub Pages

1. 把 `docs/` 目录的内容提交到仓库。
2. 仓库 Settings → Pages → Source 选择 `Deploy from a branch`，分支选 `main`，目录选 `/docs`。
3. 等待构建完成，访问 `https://<用户名>.github.io/<仓库名>/`。

`docs/.nojekyll` 必须保留，否则以 `data/` 开头的路径可能被 Jekyll 忽略。

### 发布清单

运行时只需要以下 12 个文件（约 1.37 MB），没有任何外部 CDN、字体或模型服务：

```
docs/
  .nojekyll
  index.html
  app.js
  style.css
  data/materials.json       1038 KB
  data/material_gaze.json     58 KB
  data/ngram.json            216 KB
  data/scoring.json          0.2 KB
  data/vocab.json             55 KB
  data/norm.json             0.9 KB
  data/sample_text.txt       0.2 KB
  data/sample_reading.asc    1.1 KB
```

以下文件不进 Pages，但建议一并放在仓库根目录以便复现：`README.md`、`requirements.txt`、
`build_static.py`、`build_material_gaze.py`、`build_ngram.py`、`build_standalone.py`、
`make_demo_data.py`、`models/`。

**不要提交** `test_data/`（含 47 MB 被试原始 EDF/ASC）。

### 隐私提醒

`data/material_gaze.json` 是真实被试的逐次注视坐标，`data/norm.json` 是 45 名被试的个体指标。
虽然不含姓名等直接标识，仍属人类被试数据。公开发布前请确认伦理审批与知情同意允许公开；
若不允许，可用 `make_demo_data.py` 生成的模拟数据替换，或把常模改为仅保留分位数。

## 单文件离线版

```powershell
python build_standalone.py
```

生成 `reading_efficiency_standalone.html`（约 1.4 MB），把 HTML、CSS、JS 与全部数据内联。
双击即可运行，不需要服务器、网络或任何外部程序；也可单独拷贝分发。


## 两种模式

| 模式 | 数据来源 | 精度 | 适用 |
|---|---|---|---|
| 实验材料 | `materials.json` 预计算 | 与 Python 版逐位一致 | 296 句实验材料 |
| 任意文本 | `ngram.json` 查表 + Ridge | 留出句 Pearson r = 0.92、MAE = 0.53 | 任意中文文本 |

## 交互

没有独立的"分析"按钮，选择即刷新：

- **实验材料模式**：内置一名被试（编号 sub01）的实验眼动记录，无需上传任何文件。选择句子即显示难度、注视圆点与阅读效率；句子前带 ● 的表示有内置记录，无记录的句子只显示文本难度。该模式下眼动上传模块整体隐藏。
- **任意文本模式**：先上传 `.txt`（每行一句），出现句子下拉框；选择句子显示难度；再上传 `.asc`，按同序号试次叠加眼动并显示效率。未上传 `.txt` 时眼动上传控件禁用。
- 句子与试次双向联动：换句子跳到同序号试次，换试次跳到对应句子。

任意文本模式的词级 entropy 使用训练集均值（其 Ridge 系数接近零），并采用前向最大匹配分词；分词结果与实验材料的分词口径可能不同。

## 眼动数据

- 仅接受扩展名为 `.asc` 的文件。`.edf` 需先用 EyeLink 开发者工具转换：
  `edf2asc -y input.EDF -o output.asc`
- 布局默认 `x = 600 + 字符索引 × 50`，与实验呈现程序一致。其他布局需在页面上修改两个参数。
- 若 asc 含 `SENTENCEID ExxxCy` 标记且能在材料库中匹配，会自动切换到实验材料模式并叠加对应试次的眼动；否则自动切换到任意文本模式，按试次顺序对应输入文本的句子。
- 材料模式只叠加与当前句 SENTENCEID 匹配的试次；没有匹配时不显示圆点，避免错配。

## 重新生成数据

需要项目的 Python 环境与已缓存的 GPT-2 中文模型。

```powershell
cd reading_demo
python build_static.py      # 约 3-5 分钟（逐句 GPT-2 前向）
python build_ngram.py       # 约 5-8 分钟（1547 句语料蒸馏）
python make_demo_data.py    # 数秒
```

## 已知限制

- 文本难度模型仅用 296 句实验材料训练（按句分组 5 折 R² = 0.20），对新文体存在外推误差。
- 句首词的 GPT-2 surprisal 恒为 0（无前文），难度分会系统性偏低。
- 读者效率是实测指标与常模的对照，不是预测。
- 单试次的速度估计受跳读与试次长度影响较大。
- 演示用 `sample_reading.asc` 为模拟数据，仅用于展示流程。
