"""reader.py — 读者效率：实测指标与人群常模比较
================================================================
常模来自 l3_modeling 的 45 名被试（句子级中位阅读速度、中位平均凝视时间、跳读率）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
NORM_PATH = BASE / "python_analysis" / "l3_modeling" / "l3_dataset.pkl"


def load_norm() -> pd.DataFrame:
    df = pd.read_pickle(NORM_PATH)
    return df[["subid", "speed_wps", "mean_gd", "mean_skip"]].copy()


def percentile(value: float, reference: np.ndarray) -> float:
    """value 在 reference 中的百分位（0-100）。"""
    ref = np.asarray(reference, dtype=float)
    ref = ref[~np.isnan(ref)]
    if len(ref) == 0:
        return float("nan")
    return float((ref < value).mean() * 100)


def evaluate(metrics: dict, norm: pd.DataFrame) -> dict:
    """把实测指标换算成常模百分位与文字判断。"""
    speed = metrics.get("reading_speed_wps", 0.0)
    gd = metrics.get("mean_gd_ms", 0.0)
    skip = metrics.get("skipping_rate", 0.0)

    p_speed = percentile(speed, norm.speed_wps.values)
    p_gd = percentile(gd, norm.mean_gd.values)
    p_skip = percentile(skip, norm.mean_skip.values)

    if p_speed >= 70:
        band = "偏快"
        note = f"该读者的阅读速度高于约 {p_speed:.0f}% 的常模被试。"
    elif p_speed >= 30:
        band = "中等"
        note = f"该读者的阅读速度处于常模中间区间（{p_speed:.0f} 分位）。"
    else:
        band = "偏慢"
        note = f"该读者的阅读速度低于约 {100 - p_speed:.0f}% 的常模被试。"

    return {
        "speed_wps": speed,
        "mean_gd_ms": gd,
        "skipping_rate": skip,
        "percentile_speed": p_speed,
        "percentile_gd": p_gd,
        "percentile_skip": p_skip,
        "band": band,
        "note": note,
        "norm_n": int(len(norm)),
    }


def norm_table(norm: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "指标": ["阅读速度（词/秒）", "平均凝视时间（ms）", "跳读率"],
        "常模中位数": [norm.speed_wps.median(), norm.mean_gd.median(),
                    norm.mean_skip.median()],
        "常模范围（10–90 分位）": [
            f"{norm.speed_wps.quantile(.1):.2f} – {norm.speed_wps.quantile(.9):.2f}",
            f"{norm.mean_gd.quantile(.1):.0f} – {norm.mean_gd.quantile(.9):.0f}",
            f"{norm.mean_skip.quantile(.1):.2f} – {norm.mean_skip.quantile(.9):.2f}",
        ],
    })
