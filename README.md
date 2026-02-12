# LLM NLP 评估流水线（20 Newsgroups）

本项目实现一个可复现的文本分类评估流水线，对比两类方法在 `20 Newsgroups` 数据集上的表现：
- 传统基线：`TF-IDF + Logistic Regression`
- LLM 基线：`llm_classifier`（zero-shot，模型：`Qwen/Qwen2.5-1.5B-Instruct`）

项目当前支持：
- 训练并评估传统基线
- 运行 LLM zero-shot 推理
- 统一按 20 类别进行评估（Macro-Precision / Macro-Recall / Macro-F1）
- 生成中文评估报告（HTML + PDF）

## 1. 项目结构

```text
.
├─ main.py
├─ requirements.txt
├─ data/
│  ├─ data_loader.py
│  └─ raw/
├─ modeling/
│  ├─ baseline_tfidf_logreg.py
│  ├─ llm_classifier.py
│  └─ configs/
├─ outputs/
│  ├─ baseline_tfidf_logreg_results.txt
│  ├─ results_zero_shot.txt
│  ├─ llm_predictions_zero_shot.jsonl
│  └─ llm_raw_outputs_zero_shot.jsonl
└─ evaluation/
   ├─ evaluate_models.py
   ├─ generate_report.py
   ├─ outputs/
   └─ report_zh.pdf
```

## 2. 环境安装

建议 Python 3.10+。

```bash
pip install -r requirements.txt
```

说明：
- `requirements.txt` 包含 `torch`、`transformers`、`bitsandbytes` 等依赖。
- 运行 LLM 推理建议使用可用 CUDA 的环境。

## 3. 运行主流程

主流程会执行：
1. 加载 `20 Newsgroups` 训练/测试数据
2. 训练并评估 `tfidf_logreg` 基线
3. 运行 `llm_classifier` zero-shot 推理

```bash
python main.py
```

可选参数示例：

```bash
python main.py --llm-max-test-samples 1000 --llm-model Qwen/Qwen2.5-1.5B-Instruct
```

常用参数：
- `--remove {headers,footers,quotes}`：数据去噪选项
- `--llm-max-test-samples`：LLM 推理样本上限
- `--llm-model`：Hugging Face 模型名
- `--llm-max-chars`：单样本输入最大字符数
- `--llm-max-input-tokens`：LLM tokenizer 截断上限

## 4. 输出文件说明

### 4.1 传统基线输出
- `outputs/baseline_tfidf_logreg_results.txt`
- `modeling/configs/run_tfidf_logreg_metadata.json`
- `modeling/configs/baseline_tfidf_logreg_pipeline.joblib`

### 4.2 LLM 输出
- `outputs/results_zero_shot.txt`
- `outputs/llm_predictions_zero_shot.jsonl`
- `outputs/llm_raw_outputs_zero_shot.jsonl`
- `modeling/configs/llm_run_metadata.json`

## 5. 统一评估（20 类别）

`evaluation/evaluate_models.py` 会：
- 对 `tfidf_logreg` 与 `llm_classifier` 统一评估
- 计算 `Macro-Precision`、`Macro-Recall`、`Macro-F1`
- 导出每个类别（20 类）的 Precision/Recall/F1
- 导出混淆矩阵（含 LLM unknown 版本）

运行：

```bash
python evaluation/evaluate_models.py
```

输出目录：`evaluation/outputs/`
- `metrics_summary.json`
- `detailed_metrics.json`
- `per_class_metrics.csv`
- `confusion_matrix_tfidf_logreg_20x20.csv`
- `confusion_matrix_llm_classifier_20x20.csv`
- `confusion_matrix_llm_classifier_20x21_with_unknown.csv`

## 6. 生成中文评估报告（PDF）

`evaluation/generate_report.py` 会基于评估输出生成：
- 中文 HTML 报告：`evaluation/report_zh.html`
- 中文 PDF 报告：`evaluation/report_zh.pdf`

运行：

```bash
python evaluation/generate_report.py
```

报告内容包括：
- 数据集说明
- baseline 说明
- 配置说明
- 全流程结构（pipeline）
- 输出说明
- 指标选择说明
- 混淆矩阵与误差分析

## 7. 指标选择说明

本项目优先使用 Macro 指标：
- `Macro-Precision`
- `Macro-Recall`
- `Macro-F1`

原因：20 类文本分类任务中，类别难度和分布可能不均衡，Macro 指标对每个类别等权重，更适合做模型横向对比。

## 8. 常见问题

1. LLM 出现无法映射到 20 类标签的输出怎么办？  
在评估阶段会被记为 `unknown`，并在输出文件中给出占比。

2. 评估脚本为什么要重新加载测试集？  
为了保证两个模型在同一测试集合上统一计算指标，避免样本不一致。

3. PDF 生成依赖什么？  
当前实现使用本地 Edge headless 打印 HTML 为 PDF。

## 9. 许可与用途

本项目用于学习与实验目的，请在遵守数据集与模型许可条款的前提下使用。

