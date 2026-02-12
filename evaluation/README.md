# evaluation

- `evaluate_models.py`: 对 `tfidf_logreg` 与 `llm_classifier` 在 20 类别上计算 Macro-Precision / Macro-Recall / Macro-F1，并输出逐类指标与混淆矩阵。
- `generate_report.py`: 基于评估输出生成中文 HTML 报告，并调用 Edge headless 生成 PDF。

执行顺序：
1) `python evaluation/evaluate_models.py`
2) `python evaluation/generate_report.py`
