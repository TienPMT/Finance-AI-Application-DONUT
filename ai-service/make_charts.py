"""
Render kết quả KPDL (results/all_results.json) thành ảnh PNG dán slide.

Chạy:  python make_charts.py
Xuất vào  results/charts/ :
  - chart_recall_vs_value.png   (Recall thô vs Value-Recall vs giá TB/gợi ý)
  - chart_value_coverage.png    (tổng VND bắt được / tổng VND truth)
  - chart_classification.png    (Accuracy/F1 của DT/NB/KNN)
  - table_metrics_k5.png        (bảng metric @5)
  - table_significance.png      (bảng Wilcoxon)
  - table_data_quality.png      (bảng làm sạch)
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "charts")

BG = "#ffffff"; PANEL = "#f4f7fa"; TEXT = "#1a1a1a"; GRID = "#c8d2dc"
MINT = "#1f9e7a"; AMBER = "#e0a000"; BLUE = "#2f6fd0"; PINK = "#d23f8c"; PURPLE = "#7a4fd0"

T_BG = "#ffffff"; T_HEAD = "#1f6f5c"; T_HEADTXT = "#ffffff"; T_TXT = "#1a1a1a"
T_GRID = "#c8d2dc"; T_ROW = "#ffffff"; T_ALT = "#eef3f6"; T_HI = "#d9f5ec"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.edgecolor": GRID, "font.size": 11,
})


def _bar_chart(exp):
    vs = exp["valueSummary"]; algos = exp["algos"]
    recall = [vs[a]["recall"] * 100 for a in algos]
    vrec = [vs[a]["valueRecall"] * 100 for a in algos]
    price = [vs[a]["recAvgPrice"] for a in algos]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = range(len(algos)); w = 0.35
    ax1.bar([i - w / 2 for i in x], recall, w, label="Recall thô (%)", color=AMBER)
    ax1.bar([i + w / 2 for i in x], vrec, w, label="Value-Recall (%)", color=MINT)
    ax1.set_ylabel("%"); ax1.set_xticks(list(x)); ax1.set_xticklabels(algos)
    ax1.grid(axis="y", color=GRID, alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(list(x), price, "o-", color=BLUE, lw=2, label="Giá TB/gợi ý (đ)")
    ax2.set_ylabel("đ", color=BLUE); ax2.tick_params(axis="y", colors=BLUE)
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc="upper right", facecolor=PANEL, edgecolor=GRID)
    plt.title("Recall thô vs Value-Recall · đường xanh = giá TB mỗi món gợi ý", color=TEXT)
    fig.tight_layout(); _save(fig, "chart_recall_vs_value.png")


def _coverage_chart(exp):
    vs = exp["valueSummary"]; algos = exp["algos"]
    cov = [vs[a]["valueCoverage"] * 100 for a in algos]
    colors = [MINT if a == "WCM-SPADE" else (AMBER if a == "TOP" else "#3a4d75") for a in algos]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(algos, cov, color=colors)
    for b, v in zip(bars, cov):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", color=TEXT)
    ax.set_ylabel("Value coverage (%)"); ax.grid(axis="y", color=GRID, alpha=0.5)
    ax.set_title("Value Coverage: tổng VND bắt được / tổng VND truth (WCM-SPADE bắt nhiều tiền hơn TOP)", color=TEXT)
    fig.tight_layout(); _save(fig, "chart_value_coverage.png")


def _classification_chart(cl):
    if "models" not in cl:
        return
    names = list(cl["models"].keys())
    acc = [cl["models"][n]["accuracy"] * 100 for n in names]
    f1 = [cl["models"][n]["f1"] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(names)); w = 0.35
    ax.bar([i - w / 2 for i in x], acc, w, label="Accuracy (%)", color=PURPLE)
    ax.bar([i + w / 2 for i in x], f1, w, label="F1 macro (%)", color=MINT)
    ax.set_xticks(list(x)); ax.set_xticklabels(names); ax.set_ylim(0, 105)
    for i, (a, b) in enumerate(zip(acc, f1)):
        ax.text(i - w / 2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", color=GRID, alpha=0.5)
    ax.legend(facecolor=PANEL, edgecolor=GRID)
    ax.set_title("Phân lớp khách chi tiêu cao/thấp (Chương 3)", color=TEXT)
    fig.tight_layout(); _save(fig, "chart_classification.png")


def _table(title, col_labels, rows, fname, highlight_row=None):
    fig, ax = plt.subplots(figsize=(min(2 + 1.6 * len(col_labels), 14), 1.0 + 0.5 * len(rows)))
    fig.patch.set_facecolor(T_BG)
    ax.set_facecolor(T_BG); ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(T_GRID); cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor(T_HEAD); cell.set_text_props(color=T_HEADTXT, weight="bold")
            cell.set_height(cell.get_height() * 1.15)
        elif highlight_row is not None and r == highlight_row:
            cell.set_facecolor(T_HI); cell.set_text_props(color=T_TXT, weight="bold")
        else:
            cell.set_facecolor(T_ROW if r % 2 else T_ALT); cell.set_text_props(color=T_TXT)
    ax.set_title(title, color=T_TXT, pad=14, weight="bold")
    fig.tight_layout(); _save(fig, fname, facecolor=T_BG)


def _save(fig, name, facecolor=BG):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=facecolor); plt.close(fig)
    print(f"  wrote {p}")


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(RES, "all_results.json"), encoding="utf-8") as f:
        d = json.load(f)
    exp = d["experiment"]; dq = d["dataQuality"]; cl = d["classification"]

    _bar_chart(exp)
    _classification_chart(cl)

    vs = exp["valueSummary"]
    cov = {a: f"{vs[a]['valueCoverage']*100:.1f}%" for a in exp["algos"]}
    rec = {a: f"{vs[a]['recall']:.3f}" for a in exp["algos"]}
    cap_rows = [
        ["Sinh luật tuần tự {A}→{B}", "CO", "khong", "khong", "khong", "khong"],
        ["Value-aware (trong so gia)", "CO", "khong", "khong", "khong", "khong"],
        ["Num alpha trade-off tan suat/gia tri", "CO", "khong", "khong", "khong", "khong"],
        ["Ca nhan hoa theo lich su", "CO", "khong", "CO", "CO", "theo cum"],
        ["Value coverage (bat gia tri)"] + [cov[a] for a in exp["algos"]],
        ["Recall tho (khong phai muc tieu)"] + [rec[a] for a in exp["algos"]],
    ]
    _table("Nang luc: WCM-SPADE vs baseline (chi WCM-SPADE cho luat + value-aware)",
           ["Nang luc"] + exp["algos"], cap_rows, "table_capability.png")

    metrics = exp["metrics"]
    rows = [[a] + [f"{exp['results'][a]['5'][m]:.3f}" for m in metrics] for a in exp["algos"]]
    hi = exp["algos"].index("WCM-SPADE") + 1
    _table("Metric next-basket @5 (20k khách)", ["algo"] + metrics, rows,
           "table_metrics_k5.png", highlight_row=hi)

    sig = exp.get("significance", {})
    srows = []
    for a, row in sig.get("vs", {}).items():
        for m, c in row.items():
            mark = "WCM-SPADE thắng" if c["winsRef"] else "thua"
            s = "có ý nghĩa" if c["sig"] else "ns"
            srows.append([a, m, f"{c['meanDiff']:+.3f}", f"{c['p']:.4g}", s, mark])
    _table("Kiểm định Wilcoxon: WCM-SPADE vs baseline (Chương 5)",
           ["vs baseline", "metric", "meanDiff", "p", "ý nghĩa", "kết luận"],
           srows, "table_significance.png")

    r = dq["rows"]; c = dq["customers"]; it = dq["items"]
    qrows = [
        ["Tổng dòng đọc", f"{r['totalRead']:,}"],
        ["Loại thiếu ID/item", f"{r['dropMissingId']:,}"],
        ["Loại sai ngày", f"{r['dropBadDate']:,}"],
        ["Loại giá ≤ 0", f"{r['dropBadPrice']:,}"],
        ["Loại số lượng ≤ 0", f"{r['dropBadAmount']:,}"],
        ["Giữ lại", f"{r['kept']:,} ({r['keptPct']}%)"],
        ["Dòng trùng → dedup", f"{r['duplicateLines']:,} ({r['duplicatePct']}%)"],
        ["Khách hàng", f"{c['nCustomers']:,}"],
        ["Khách ≥2 basket (holdout)", f"{c['nWithGe2Baskets']:,} ({c['ge2Pct']}%)"],
        ["Item phân biệt", f"{it['nItems']:,}"],
        ["Giá item min→max", f"{it['priceMin']} → {it['priceMax']}"],
        ["Chi tiêu/khách median", f"{c['spendMedian']:,}"],
    ]
    _table("Làm sạch & EDA dữ liệu Ta-Feng (full 817k dòng)",
           ["Chỉ số", "Giá trị"], qrows, "table_data_quality.png")

    print(f"DONE → {OUT}")


if __name__ == "__main__":
    main()
