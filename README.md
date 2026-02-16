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

`evaluation/evaluate_models.py` 模组会对本项目进行统一评估。标准如下：
- 对 `tfidf_logreg` 与 `llm_classifier` 两个模型分别进行评估
- 考虑到不同文本类别的难度和分布可能不均匀，采用了涉及权重配比的 `Macro-Precision`、`Macro-Recall`、`Macro-F1` 指标进行比较
- 生成每个类别（20 类）的 Precision/Recall/F1 与混淆矩阵

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

## 6. 预测结果

下表展示统一评估下两种模型的 Macro 指标：

| 模型 | Macro-Precision | Macro-Recall | Macro-F1 |
| --- | ---: | ---: | ---: |
| `tfidf_logreg` | 0.6653 | 0.6469 | 0.6450 |
| `llm_classifier` | 0.3580 | 0.1865 | 0.2001 |

## 7. 生成中文评估报告（PDF）

通过 `evaluation/generate_report.py` 生成PDF格式报告。报告内容包括：
- 数据集说明
- baseline 说明
- 配置说明
- 全流程结构（pipeline）
- 输出说明
- 指标选择说明
- 混淆矩阵与误差分析

运行：

```bash
python evaluation/generate_report.py
```