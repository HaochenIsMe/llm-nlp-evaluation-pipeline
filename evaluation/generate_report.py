from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _build_aliases(labels: List[str]) -> List[str]:
    if len(labels) > 26:
        raise ValueError("Only supports up to 26 labels for A-Z aliases")
    return [chr(ord("A") + i) for i in range(len(labels))]


def _top_confusions(matrix: List[List[int]], labels: List[str], top_n: int = 10) -> List[Tuple[str, str, int]]:
    pairs: List[Tuple[str, str, int]] = []
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            if i == j:
                continue
            if count > 0:
                pairs.append((labels[i], labels[j], int(count)))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:top_n]


def _cell_style(value: int, max_value: int) -> str:
    if value <= 0 or max_value <= 0:
        return "background-color: #ffffff;"
    ratio = value / max_value
    # Larger value -> darker color.
    lightness = 97 - int(52 * ratio)
    return f"background-color: hsl(197, 82%, {lightness}%);"


def _matrix_html(matrix: List[List[int]], labels: List[str], aliases: List[str], title: str) -> str:
    max_value = max((max(row) for row in matrix), default=0)
    header_cells = "".join(f"<th>{html.escape(alias)}</th>" for alias in aliases)

    body_rows = []
    for i, row in enumerate(matrix):
        cells = []
        for value in row:
            style = _cell_style(int(value), max_value)
            cells.append(f"<td style='{style}'>{int(value)}</td>")
        body_rows.append(f"<tr><th>{html.escape(aliases[i])}</th>{''.join(cells)}</tr>")

    legend = (
        "<div class='heat-legend'>"
        "<span>颜色深浅: 小</span>"
        "<span class='box light'></span>"
        "<span class='box mid'></span>"
        "<span class='box dark'></span>"
        "<span>大</span>"
        "</div>"
    )

    return (
        f"<h3>{html.escape(title)}</h3>"
        "<div class='table-wrap'><table class='matrix'>"
        f"<thead><tr><th>真实\\预测</th>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
        f"{legend}"
    )


def _alias_mapping_html(labels: List[str], aliases: List[str]) -> str:
    rows = []
    for alias, label in zip(aliases, labels):
        rows.append(f"<tr><td><b>{html.escape(alias)}</b></td><td>{html.escape(label)}</td></tr>")
    return (
        "<h3>类别缩写映射（A~T）</h3>"
        "<table class='alias-map'>"
        "<thead><tr><th>缩写</th><th>原始类别</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _find_edge() -> Path | None:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _write_html(report_html: Path, detailed: Dict[str, object], summary: Dict[str, object]) -> None:
    labels = list(summary["dataset"]["categories"])
    aliases = _build_aliases(labels)
    label_to_alias = {label: alias for label, alias in zip(labels, aliases)}

    tfidf = detailed["tfidf_logreg"]
    llm = detailed["llm_classifier"]

    llm_meta_path = PROJECT_ROOT / "modeling" / "configs" / "llm_run_metadata.json"
    llm_model_name = str(_load_json(llm_meta_path).get("model", "N/A")) if llm_meta_path.exists() else "N/A"

    tfidf_top_errors = _top_confusions(tfidf["confusion_matrix_20x20"], labels, top_n=12)
    llm_top_errors = _top_confusions(llm["confusion_matrix_20x20"], labels, top_n=12)

    def top_error_list(items: List[Tuple[str, str, int]]) -> str:
        lines = []
        for true_label, pred_label, count in items:
            true_alias = label_to_alias[true_label]
            pred_alias = label_to_alias[pred_label]
            lines.append(
                f"<li><code>{html.escape(true_alias)}</code> ({html.escape(true_label)}) 被预测为 "
                f"<code>{html.escape(pred_alias)}</code> ({html.escape(pred_label)}): <b>{count}</b> 次</li>"
            )
        return "<ol>" + "".join(lines) + "</ol>"

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>20 Newsgroups 分类评估报告</title>
  <style>
    @page {{ size: A4; margin: 14mm; }}
    body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; color: #111; line-height: 1.45; }}
    h1, h2, h3 {{ margin: 10px 0 6px; }}
    h1 {{ font-size: 24px; }}
    h2 {{ font-size: 18px; border-left: 4px solid #0f766e; padding-left: 8px; }}
    h3 {{ font-size: 14px; }}
    p, li {{ font-size: 12px; }}
    code {{ background: #f1f5f9; padding: 1px 4px; border-radius: 4px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 9px; table-layout: fixed; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 2px 3px; text-align: center; }}
    th {{ background: #e2e8f0; }}
    .table-wrap {{ overflow-x: auto; margin-bottom: 6px; }}
    .kpi {{ width: 100%; border-collapse: collapse; margin: 8px 0; }}
    .kpi th, .kpi td {{ font-size: 12px; padding: 6px; }}
    .alias-map th, .alias-map td {{ font-size: 11px; padding: 4px; text-align: left; }}
    .note {{ background: #f8fafc; border-left: 3px solid #0ea5e9; padding: 8px; font-size: 11px; }}
    .heat-legend {{ display: flex; align-items: center; gap: 6px; margin: 2px 0 10px; font-size: 10px; }}
    .heat-legend .box {{ width: 14px; height: 12px; border: 1px solid #94a3b8; display: inline-block; }}
    .heat-legend .light {{ background: hsl(197, 82%, 89%); }}
    .heat-legend .mid {{ background: hsl(197, 82%, 70%); }}
    .heat-legend .dark {{ background: hsl(197, 82%, 50%); }}
  </style>
</head>
<body>
  <h1>20 Newsgroups 文本分类评估报告</h1>
  <p>本报告由 <code>evaluation/evaluate_models.py</code> 与 <code>evaluation/generate_report.py</code> 自动生成。</p>

  <h2>1. 数据集说明</h2>
  <p>数据集为 <b>20 Newsgroups</b>，共 20 个主题类别，测试集样本数为 <b>{summary['dataset']['num_test_samples']}</b>。</p>
  <p>本项目数据加载函数为 <code>data/data_loader.py</code>，使用 <code>fetch_20newsgroups</code> 并移除 headers/footers/quotes 噪声字段。</p>

  <h2>2. Baseline 与模型说明</h2>
  <ul>
    <li><b>tfidf_logreg</b>：<code>TfidfVectorizer + LogisticRegression</code>，模型工件路径见 <code>modeling/configs/run_tfidf_logreg_metadata.json</code>。</li>
    <li><b>llm_classifier</b>：基于 Hugging Face 因果语言模型的 zero-shot 分类，当前记录模型为 <code>{html.escape(llm_model_name)}</code>。</li>
  </ul>

  <h2>3. 配置说明</h2>
  <ul>
    <li>训练/推理入口：<code>main.py</code></li>
    <li>传统基线配置：<code>modeling/configs/run_tfidf_logreg_metadata.json</code></li>
    <li>LLM 运行配置：<code>modeling/configs/llm_run_metadata.json</code></li>
    <li>评估输出目录：<code>evaluation/outputs</code></li>
  </ul>

  <h2>4. 全流程结构（Pipeline）</h2>
  <ol>
    <li>加载 20 Newsgroups 数据（训练/测试）</li>
    <li>训练 TF-IDF + Logistic Regression 基线并保存模型</li>
    <li>运行 LLM zero-shot 预测并保存 JSONL 结果</li>
    <li>统一评估脚本读取两种模型输出，计算宏平均与按类指标</li>
    <li>生成评估工件（JSON/CSV）并产出中文 PDF 报告</li>
  </ol>

  <h2>5. 输出说明</h2>
  <ul>
    <li><code>evaluation/outputs/metrics_summary.json</code>：核心宏平均指标</li>
    <li><code>evaluation/outputs/detailed_metrics.json</code>：按类指标与混淆矩阵</li>
    <li><code>evaluation/outputs/per_class_metrics.csv</code>：20 类别逐类 Precision/Recall/F1</li>
    <li><code>evaluation/outputs/confusion_matrix_*.csv</code>：混淆矩阵数据</li>
  </ul>

  <h2>6. 评估指标选择说明</h2>
  <p>本项目使用 <b>Macro-Precision / Macro-Recall / Macro-F1</b> 作为核心指标。原因：20 类别任务中，不同类别难度差异明显，宏平均能够让每个类别等权重参与评估，避免被高频类别主导。</p>

  <table class="kpi">
    <thead>
      <tr><th>模型</th><th>Macro-Precision</th><th>Macro-Recall</th><th>Macro-F1</th><th>未知预测占比</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>tfidf_logreg</td>
        <td>{_fmt(float(summary['models']['tfidf_logreg']['macro_precision']))}</td>
        <td>{_fmt(float(summary['models']['tfidf_logreg']['macro_recall']))}</td>
        <td>{_fmt(float(summary['models']['tfidf_logreg']['macro_f1']))}</td>
        <td>{_fmt(float(summary['models']['tfidf_logreg']['unknown_prediction_rate']))}</td>
      </tr>
      <tr>
        <td>llm_classifier</td>
        <td>{_fmt(float(summary['models']['llm_classifier']['macro_precision']))}</td>
        <td>{_fmt(float(summary['models']['llm_classifier']['macro_recall']))}</td>
        <td>{_fmt(float(summary['models']['llm_classifier']['macro_f1']))}</td>
        <td>{_fmt(float(summary['models']['llm_classifier']['unknown_prediction_rate']))}</td>
      </tr>
    </tbody>
  </table>

  <h2>7. 混淆矩阵与误差分析</h2>
  <p>为解决类别名称过长导致的显示出界，混淆矩阵统一使用 <b>A~T</b> 表示 20 个类别；完整映射见下方表格。</p>
  {_matrix_html(tfidf['confusion_matrix_20x20'], labels, aliases, 'TF-IDF + LogReg 混淆矩阵 (20x20)')}
  {_matrix_html(llm['confusion_matrix_20x20'], labels, aliases, 'LLM 混淆矩阵 (20x20，仅统计可映射标签)')}
  {_alias_mapping_html(labels, aliases)}

  <h3>TF-IDF + LogReg 主要混淆对</h3>
  {top_error_list(tfidf_top_errors)}

  <h3>LLM 主要混淆对</h3>
  {top_error_list(llm_top_errors)}

  <div class="note">
    说明：LLM 存在部分输出无法映射到 20 个标准标签的情况，评估中记为 unknown；
    在按 20 类混淆矩阵展示时，这部分样本不计入 20x20 方阵，详细数量见 summary 文件中的 unknown_prediction_rate。
  </div>
</body>
</html>
"""

    report_html.write_text(html_doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chinese evaluation report HTML/PDF")
    parser.add_argument(
        "--eval-output-dir",
        default=str(PROJECT_ROOT / "evaluation" / "outputs"),
        help="Directory containing evaluation outputs",
    )
    parser.add_argument(
        "--report-html",
        default=str(PROJECT_ROOT / "evaluation" / "report_zh.html"),
        help="Output HTML report path",
    )
    parser.add_argument(
        "--report-pdf",
        default=str(PROJECT_ROOT / "evaluation" / "report_zh.pdf"),
        help="Output PDF report path",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_output_dir)
    summary_path = eval_dir / "metrics_summary.json"
    detailed_path = eval_dir / "detailed_metrics.json"
    if not summary_path.exists() or not detailed_path.exists():
        raise FileNotFoundError(
            "Missing evaluation outputs. Run evaluation/evaluate_models.py first."
        )

    summary = _load_json(summary_path)
    detailed = _load_json(detailed_path)

    report_html = Path(args.report_html)
    report_html.parent.mkdir(parents=True, exist_ok=True)
    _write_html(report_html, detailed=detailed, summary=summary)
    print(f"Saved HTML report: {report_html}")

    report_pdf = Path(args.report_pdf)
    edge_path = _find_edge()
    if edge_path is None:
        print("Edge not found. Skipped PDF rendering.")
        return

    report_pdf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(edge_path),
        "--headless=new",
        "--disable-gpu",
        f"--print-to-pdf={report_pdf}",
        report_html.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True)
    print(f"Saved PDF report: {report_pdf}")


if __name__ == "__main__":
    main()
