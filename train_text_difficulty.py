"""train_text_difficulty.py — 训练"文本 → 人群阅读难度"模型
================================================================
用途
    演示程序的文本层：给定一段中文文本，预测每个词对一般读者的加工难度。

标签
    每个词的跨被试平均凝视时间（仅统计非跳读、GD ∈ [60, 1500] ms 的观测），
    取对数后作为"人群难度"。要求该词至少被 10 名被试读过。

特征
    GPT-2 词级 surprisal、entropy，以及词长。

验证
    GroupKFold(5)，按句子分组——同一句子的不同词不会同时出现在训练与测试折，
    避免同句上下文造成的乐观偏差。

输出
    models/text_difficulty.pkl     (scaler + ridge + 难度分映射参数)
    models/text_difficulty_metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

BASE = Path(r"d:\Working\EEG\direct lexical control")
V3 = BASE / "python_analysis" / "full_v3"
OUT = BASE / "reading_demo"
MODEL_DIR = OUT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["t_surprisal", "t_entropy", "length"]
GD_MIN, GD_MAX = 60.0, 1500.0
MIN_SUBJECTS_PER_WORD = 10


def build_training_table() -> pd.DataFrame:
    em = pd.read_pickle(V3 / "cache" / "em_word.pkl")
    em = em.rename(columns={"subjectid": "subid"})
    em = em[(em.skipping == 0) & (em.gd >= GD_MIN) & (em.gd <= GD_MAX)]

    # 词级人群聚合
    g = em.groupby(["sentid", "wfcond", "wordid"], as_index=False).agg(
        mean_gd=("gd", "mean"),
        n_subj=("subid", "nunique"),
        length=("length", "first"),
    )
    g = g[g.n_subj >= MIN_SUBJECTS_PER_WORD].copy()
    g["log_gd"] = np.log(g.mean_gd)

    fc = pd.read_csv(V3 / "output" / "forecast_features_word_slim.csv")
    fc = fc.rename(columns={"sentenceid": "sentid", "word_pos": "wordid"})
    fc["wfcond"] = fc.sent_cond.map({"hf": 1, "lf": 0})
    fc = fc[["sentid", "wfcond", "wordid", "t_surprisal", "t_entropy"]]

    df = g.merge(fc, on=["sentid", "wfcond", "wordid"], how="inner")
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df


def main() -> None:
    df = build_training_table()
    print(f"training words: {len(df):,} | sentences: {df.sentid.nunique()}")
    print(f"label log_gd: mean={df.log_gd.mean():.3f}, sd={df.log_gd.std():.3f}")
    print(df[FEATURES].describe().round(3).to_string())

    X = df[FEATURES].values.astype(float)
    y = df.log_gd.values.astype(float)
    groups = df.sentid.values

    gkf = GroupKFold(n_splits=5)
    oof = np.full(len(y), np.nan)
    alphas = [0.1, 1.0, 10.0, 100.0]
    best_alpha_scores = {a: [] for a in alphas}

    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        # 内层选择 alpha
        inner = GroupKFold(n_splits=3)
        scores = {}
        for a in alphas:
            s = []
            for itr, iva in inner.split(Xtr, y[tr], groups[tr]):
                m = Ridge(alpha=a).fit(Xtr[itr], y[tr][itr])
                s.append(r2_score(y[tr][iva], m.predict(Xtr[iva])))
            scores[a] = float(np.mean(s))
        a_best = max(scores, key=scores.get)
        best_alpha_scores[a_best].append(1)
        m = Ridge(alpha=a_best).fit(Xtr, y[tr])
        oof[te] = m.predict(Xte)

    metrics = {
        "n_words": int(len(y)),
        "n_sentences": int(df.sentid.nunique()),
        "r2": float(r2_score(y, oof)),
        "pearson_r": float(np.corrcoef(y, oof)[0, 1]),
        "spearman_rho": float(spearmanr(y, oof).statistic),
        "mae_log": float(mean_absolute_error(y, oof)),
        "features": FEATURES,
        "label": "cross-subject mean log(GD)",
        "cv": "GroupKFold(5) by sentence",
    }
    print("\n--- cross-validated performance ---")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # 最终模型：全量拟合
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    final = Ridge(alpha=10.0).fit(Xs, y)
    coefs = dict(zip(FEATURES, final.coef_.round(5).tolist()))

    # 难度分映射：OOF 预测的 1%/99% 分位 → 0-100
    lo, hi = np.quantile(oof, [0.01, 0.99])
    artifact = {
        "scaler_mean": sc.mean_.tolist(),
        "scaler_scale": sc.scale_.tolist(),
        "ridge_coef": final.coef_.tolist(),
        "ridge_intercept": float(final.intercept_),
        "features": FEATURES,
        "score_lo": float(lo),
        "score_hi": float(hi),
        "label_sd": float(y.std()),
    }
    pd.to_pickle(artifact, MODEL_DIR / "text_difficulty.pkl")
    with open(MODEL_DIR / "text_difficulty_metrics.json", "w", encoding="utf-8") as f:
        json.dump({**metrics, "coefficients": coefs,
                   "score_range": [float(lo), float(hi)]},
                  f, ensure_ascii=False, indent=2)

    print("\n系数（标准化后）：", coefs)
    print(f"难度分映射区间: [{lo:.3f}, {hi:.3f}] → 0-100")
    print("saved:", MODEL_DIR / "text_difficulty.pkl")


if __name__ == "__main__":
    main()
