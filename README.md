# NLP 第一次作业代码

本项目实现作业要求中的 6 个实验，并确保所有方法使用同一份、可复现的 NYT 80%/10%/10% 分层划分：

1. Binary Bag of Words + Logistic Regression
2. Word Frequency + Logistic Regression
3. Pre-trained GloVe 100d + mean pooling + Logistic Regression
4. AG News 训练 Word2Vec 100d + mean pooling + Logistic Regression
5. NYT 训练集训练 Word2Vec 100d + mean pooling + Logistic Regression
6. `google-bert/bert-base-uncased` 微调，`max_length=64`，训练 3 epochs

每个实验都在同一个 NYT Test Set 上计算 Accuracy 和 Macro-F1。默认随机种子为 42。为了避免测试集信息泄漏，NYT Word2Vec 只使用 NYT Training Set 训练。

作业规定的关键参数集中定义在 `nlp_assignment/constants.py`：词向量维度固定为 100，BERT 模型固定为 `google-bert/bert-base-uncased`，最大长度固定为 64，训练轮数固定为 3。自动化测试会检查这些值，防止修改代码时意外偏离题目。

## 提交内容

- 最终实验报告：[2312390_程子晨_第一次实验报告.pdf](output/pdf/2312390_程子晨_第一次实验报告.pdf)
- 完整实验代码：`run.py` 与 `nlp_assignment/`
- 代码运行说明：本 README
- 测试与提交前自检：`tests/` 与 `check_assignment.py`
- 报告分析和绘图代码：`report_analysis.py`、`analyze_all_methods.py`、`generate_report_figures.py`
- 实验结果摘要：`outputs/results.csv`、`outputs/results.json`、`outputs/report_diagnostics.json`

课程提供的数据集、训练后的模型权重、数据划分副本和缓存文件体积较大，不属于提交代码，已通过 `.gitignore` 排除。复现实验时，请按下文说明放置数据并重新运行代码。

## 1. 环境安装

建议使用 Python 3.10 或 3.11，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

BERT 训练建议使用 CUDA GPU。CPU 可以运行，但耗时会明显增加。

## 2. 数据准备

运行前请将课程提供的数据放到以下位置：

- `HW-1/nyt.csv`
- `HW-1/ag.csv`

数据文件不会被 Git 提交；评分者可使用课程原始数据按上述路径运行，或通过命令行参数指定其他位置。

### NYT

支持 CSV、JSON、JSONL、Parquet，默认要求两列：

- `text`：新闻文本
- `label`：类别

默认路径为 `HW-1/nyt.csv`。若列名不同，可使用 `--text-column` 和 `--label-column`。

### AG News

有两种使用方式：

- 不传 `--ag-news`：代码通过 Hugging Face `datasets` 自动下载 `fancyzhx/ag_news`。
- 传本地文件：文件需有 `text` 列，或同时有 `title`、`description` 两列。

### GloVe

从 <https://nlp.stanford.edu/data/glove.6B.zip> 下载。代码可以直接读取 ZIP、解压后的 `glove.6B.100d.txt`，或 Gensim `KeyedVectors` 保存的 `.kv` 文件，并且只保留 NYT 中需要的词向量。

## 3. 运行实验

以下命令均在项目根目录执行。

### Task 1：两种 Bag of Words

```bash
python run.py --task bow
```

### Task 2.1：Pre-trained GloVe

```bash
python run.py --task glove --glove data/glove.6B.zip
```

### Task 2.2：AG News Word2Vec

自动下载 AG News：

```bash
python run.py --task w2v-ag
```

使用本地 AG News：

```bash
python run.py --task w2v-ag --ag-news HW-1/ag.csv
```

### Task 2.3：NYT Word2Vec

```bash
python run.py --task w2v-nyt
```

### Task 3：BERT

```bash
python run.py --task bert --bert-batch-size 16
```

显存不足时可把 batch size 调小，例如 `--bert-batch-size 8`。模型、最大长度和 epochs 的默认值严格按作业要求设置。

### 一次运行全部实验

```bash
python run.py --task all --glove data/glove.6B.zip --ag-news HW-1/ag.csv
```

若已按“数据准备”一节放置课程提供的 AG News，建议显式传入 `--ag-news HW-1/ag.csv`，避免重复下载。

## 4. 已完成的真实实验结果

使用随机种子 42，NYT 划分为 9,215 条训练、1,152 条验证和 1,152 条测试数据。当前 `outputs/results.csv` 中保存了以下测试集结果：

- Binary Bag of Words：Accuracy 0.9887，Macro-F1 0.9718
- Word Frequency：Accuracy 0.9913，Macro-F1 0.9789
- Pre-trained GloVe 100d：Accuracy 0.9852，Macro-F1 0.9646
- AG News Word2Vec 100d：Accuracy 0.9757，Macro-F1 0.9403
- NYT Training Set Word2Vec 100d：Accuracy 0.9852，Macro-F1 0.9635
- BERT-base-uncased：Accuracy 0.9818，Macro-F1 0.9601

## 5. 输出说明

默认输出到 `outputs/`：

- `results.csv`：六种方法的 Accuracy 与 Macro-F1 汇总
- `results.json`：指标和词表大小、词向量覆盖率等详细信息
- `run_config.json`：随机种子、类别和各划分样本数
- `splits/`：本次实验实际使用的 train/validation/test CSV，便于复现
- `models/`：两份 Word2Vec 模型
- `bert_model/`：最终 BERT 模型和 tokenizer
- `bert_checkpoints/`：BERT 训练检查点（最多保留两个）

可用 `--output-dir` 修改输出目录。
分任务多次运行时，程序会按方法名称更新并合并结果，不会覆盖其他已完成实验的指标。

`run_config.json` 还记录 NYT 文件的 SHA-256、每个划分的类别数量和实验指纹。如果数据文件、随机种子或列配置发生变化，程序会拒绝把新结果混入旧输出目录，应改用新的 `--output-dir`。Word2Vec 使用稳定哈希和单 worker 默认设置，确保不同 Python 进程中的初始化可复现。

## 6. 常用可调参数

```text
--seed 42
--min-df 1
--max-features N
--logreg-c 1.0
--w2v-epochs 10
--w2v-min-count 2
--workers 1
--bert-batch-size 16
--bert-learning-rate 2e-5
```

`--workers 1` 最利于 Word2Vec 结果复现；增加 workers 可加速训练，但结果可能有轻微随机差异。

## 7. 运行测试

项目使用标准库 `unittest`，无需额外安装 pytest：

```bash
python -m unittest discover -s tests -v
```

完成真实实验后，可运行提交前自检：

```bash
python check_assignment.py
```

该命令会检查六种方法、两项指标、统一划分、作业规定参数和最终模型文件是否齐全。

## 8. 实验报告与 LaTeX 编译

完整报告源文件为 `latex 模板/作业一实验报告.tex`，`latex 模板/main.tex` 是统一编译入口，会载入完整报告；不要再使用旧的空白模板生成 PDF。在 `latex 模板` 目录下运行两次 `xelatex main.tex`，第二次用于更新引用，即可得到 `main.pdf`。需要安装 XeLaTeX，并保留同目录的 `NKU.png`。

当前已编译的交付版本另存于 `output/pdf/2312390_程子晨_第一次实验报告.pdf`。报告里的 GloVe 实验实际使用 Gensim 发布的 `glove-wiki-gigaword-100`，而非 README 上述命令示例中的本地 `glove.6B.zip`；两者都是符合要求的 100 维预训练 GloVe，但复现实验结果时应保持词向量来源一致。

报告新增的逐类指标、混淆矩阵、重复正文审计可通过 `python report_analysis.py` 从已保存的划分文件复算。进一步运行 `python analyze_all_methods.py`，会从保存的词向量与 BERT 模型复算六组预测，核对原始 Accuracy 与 Macro-F1，并生成 `outputs/report_diagnostics.json`：其中含六组逐类指标、错误类型、配对 McNemar 检验、共同错误统计和去掉完全重叠测试样本后的敏感性指标。该过程不修改六组原始实验结果；BERT 推断建议使用 CUDA GPU。

若需要重绘报告中的四张图，先运行上述两个分析脚本，再安装 `reportlab` 并在项目根目录运行 `python generate_report_figures.py`。总体指标图、两种词袋逐类柱状图、六组逐类 F1 热力图和错误类型分解图会保存为 `latex 模板/figures/` 下的矢量 PDF；随后在 `latex 模板` 中运行两次 `xelatex main.tex`。绘图脚本会先核对复算的总体指标与原始记录一致。
